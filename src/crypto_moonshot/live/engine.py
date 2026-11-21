"""The trading engine: ticks in; batches, books, checks, orders and the state file out.

One asyncio component with an injected clock, driven by any `Feed` (live or replay). It owns
the broker (an `ExecutionBroker`: the paper broker behind `PaperExecution` unless another is
given) and the strategy books and writes everything to the `StateStore` (used from the
event-loop thread only, and never inside a transaction across an `await`). See
`docs/architecture.md`, "Engine", for the design; the exact rules are here.

Per tick (interval *I*, candle *T*, entry time *E = T + I*)
----------------------------------------------------------
1. **Broker.** Orders left `unknown` are resolved with `query` first. A broker that
   `needs_candles` gets them for every series it needs on *I* (a position from its entry
   candle, a pending order from its time): the tick's candle when the previous one was handed
   over, else the feed's window since the last `HISTORY_SLACK` candles (so late candles that
   filled a hole are picked up). Then `events(E, interval=I)`, and every event is recorded in
   one transaction (fills, positions, bracket order rows, cancellations, rejections). For a
   `needs_candles` broker, pending buys whose entry candle has still not closed `entry_grace`
   after it should have are cancelled (`no_candle`). For a broker that does not manage time
   exits, every open position on *I* whose `entry time + max_hold` is at or before *E* is sold
   now: its bracket is cancelled, and only once the cancel is confirmed (a fill that raced it
   wins) a market sell with reason `time` is placed.
2. **Signals.** Tick listeners run (M6 broadcasts the tick to external providers). In-process
   plugins get their windows (read on the event-loop thread: the traded symbols, minus those
   whose latest candle is more than `stale_after` intervals old, as `rolling_signals` does,
   and the context symbols) and run in a worker thread; a plugin whose traded symbols have no
   candle in this tick gets an empty batch without being called. External batches arrive
   through `submit_batch`. Every batch is recorded when it arrives. The engine waits until
   every strategy of *I* has reported or `signal_deadline` after the close (by the clock);
   in replay it waits for every batch, up to `replay_timeout` of real time. A plugin that
   raises, returns a bad frame or is still running then misses its batch and an error event
   is written; the engine never crashes on a plugin, and stops waiting for one that failed.
   `stop()` abandons a tick still collecting: it is decided again after a restart.
3. **Decisions** (one transaction, ending with the tick marker). Missed batches are recorded;
   the controls are read (a requested flatten sells every open position for *E*, cancels
   pending buys and pauses); exit signals become market sells for *E*; then entries, across
   strategies, in order of signal priority, then strategy name, then symbol. Each entry passes,
   in order (the first refusal is the signal's recorded reason):

   `paused` -> `unknown_symbol` (not one of the strategy's symbols) -> `symbol_held` (the
   account's one position per symbol, pending entries included) -> `stale` (`feed.stale_symbols`)
   -> sizing on the strategy's book (the signal's `weight` for weighted sizing) -> the book's
   risk checks (`risk.first_rejection`, unchanged: `daily_loss_limit`, `day_cap`,
   `slots_full`, `symbol_held`, `exposure`) -> `cash` (the book's) -> `max_order_usdt` ->
   `max_total_exposure` (open cost over realised equity, both summed over the books).

   An accepted entry is a market buy for *E* with a deterministic client order ID, carrying
   the strategy's exit plan with the signal's partial override. Entries are placed one at a
   time, in that order, so a rejected one frees its slot and cash for the next.

Every order goes **intent, then action, then outcome**: its row (`new`, deterministic client
order ID) and, for an entry, its pending position (stake reserved, `placed_at = E`) and the
signal's outcome are committed before `broker.submit` is awaited (`Step.BEFORE_BROKER_CALL`);
the result is recorded in a second transaction (`Step.AFTER_BROKER_CALL` lies between):
accepted (`pending`, plus any immediate fill), rejected (the order `rejected`, an entry's
position cancelled and its signal `rejected` with `broker_<reason>`), or unknown (the order
`unknown`, resolved by `query` at the next broker step). A whole-position sell (an exit
signal, a flatten, a time exit) cancels the position's bracket first when the broker's sell
does not (`sell_cancels_bracket`). The decision step is therefore several transactions; the
last one writes the tick's refusal summary and the tick marker.

Books are kept in memory for the decisions and re-derived from the state file
(`StateStore.derive_book`) after every step that changed it and whenever the UTC day changes,
so they always equal `derive_book` and a restarted engine computes exactly the same numbers.

**Between ticks** a broker that does not need candles (an exchange) is drained every
`engine.drain_interval`: while no tick is being processed, `events(now)` is called and the
events are recorded in their own transaction, so a fill reaches the state file (and the
dashboard) within seconds instead of at the next tick. The paper broker is never drained
between ticks.

A batch for a candle already processed is *late*: it is recorded, its entries are expired and
its exits become market sells for the next candle open. Duplicates are refused.

Recovery
--------
At start the engine checks the config against the stored strategies (a changed capital or
initial cash, or a removed strategy with open positions, is refused), restores a paper broker
from `StateStore.broker_snapshot` (which holds orders a crash left `new`: the paper broker is
deterministic) and derives the books. Then (`run`, before the first tick) every `new` or
`unknown` order is resolved with `broker.query`: one the broker holds is accepted; one it never
placed is rejected (`not_placed`: an entry's position is cancelled, so only a later tick may
enter that symbol again), except a sell, which finishes the exit it was meant for (its bracket
cancelled first); `reconcile` then compares the broker with the state file (orphaned orders
and balance mismatches become events and alerts, and a mismatch pauses entries). On the first
tick of each interval the engine gives a `needs_candles` broker every candle it needs from each
position's entry (and each pending order's time): the feed's window if it reaches back far
enough, else the `history` fetcher. The tick's broker step then applies exits that happened
while the engine was down at their real times. Ticks missed while down are recorded as an
event; their signals are not replayed. Because every write is in a transaction, client order
IDs are deterministic, intents are committed before the broker is called and the tick marker
commits last, a restart after a crash at any point neither loses nor duplicates an order or a
position: a tick whose marker was not committed is processed again (its broker step finds
nothing new, recorded batches and signal outcomes are reused).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import polars as pl

from crypto_moonshot.broker.base import (
    ExitPlan,
    Order,
    OrderType,
    Side,
    child_order_id,
    client_order_id,
)
from crypto_moonshot.broker.candles import Candles
from crypto_moonshot.broker.execution import (
    BracketFailed,
    BracketPlaced,
    BrokerEvent,
    BrokerHalted,
    CancelStatus,
    EntryFilled,
    ExecutionBroker,
    ExitFilled,
    OrderCancelled,
    OrderRejection,
    OrderState,
    PaperExecution,
    Reconciliation,
    RestoresState,
    SubmitResult,
    SubmitStatus,
    WorkingOrder,
)
from crypto_moonshot.live.config import ConfigError, RunConfig, StrategyConfig
from crypto_moonshot.live.contract import ExitPlanIn, SignalBatch, SignalIn, frame_to_batch
from crypto_moonshot.live.feed import (
    HISTORY_SLACK,
    BackfillFailed,
    Feed,
    FeedDisconnected,
    FeedEvent,
    KlineFetcher,
    SeedFailed,
    StaleSymbolsChanged,
    SubscriptionFailed,
    Tick,
)
from crypto_moonshot.live.plugin import Strategy, load_strategy
from crypto_moonshot.live.state import (
    BatchRecord,
    BatchStatus,
    Control,
    Controls,
    EventLevel,
    OrderRecord,
    OrderStatus,
    PositionRecord,
    PositionStatus,
    ProviderRecord,
    SignalOutcome,
    SignalRecord,
    SignalRow,
    StateStore,
)
from crypto_moonshot.models import KLINE_SCHEMA, Interval, as_utc
from crypto_moonshot.portfolio import Book, Sizing, WeightedSizing, utc_day
from crypto_moonshot.risk import RiskCheck, first_rejection

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]
TickListener = Callable[[Tick], Awaitable[None] | None]
StepHook = Callable[[str, Tick | None], None]
"""Called at each step boundary (`Step`) with the tick: tests inject crashes here."""

_TOLERANCE = 1e-9
_EXPOSURE_TOLERANCE = 1e-12


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Step(StrEnum):
    """Step boundaries of a tick, where `step_hook` is called."""

    AFTER_BROKER = "after_broker"
    """The broker step is committed."""
    AFTER_SIGNALS = "after_signals"
    """Every batch is in (or the deadline passed); nothing of step 3 is written yet."""
    BEFORE_BROKER_CALL = "before_broker_call"
    """An order's intent is committed; the broker is not called yet (for a sell, not even to
    cancel the bracket)."""
    AFTER_BROKER_CALL = "after_broker_call"
    """`submit` returned; its outcome is not recorded yet."""
    BEFORE_TICK_MARKER = "before_tick_marker"
    """Inside the last transaction of step 3: every decision is written, the tick marker not
    yet."""
    TICK_DONE = "tick_done"
    """The tick is committed and the books re-derived."""


# -- strategies -------------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategySpec:
    """A strategy instance as the engine runs it: its config, the loaded plugin (`None` for an
    external provider) and everything resolved from both."""

    config: StrategyConfig
    interval: Interval
    warmup: int
    symbols: tuple[str, ...]
    context_symbols: tuple[str, ...]
    exit: ExitPlan | None
    plugin: Strategy | None
    sizing: Sizing
    checks: tuple[RiskCheck, ...]

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def capital(self) -> float:
        return self.config.capital

    @property
    def external(self) -> bool:
        return self.plugin is None


def build_specs(
    config: RunConfig,
    data_root: Path | None,
    *,
    plugins: Mapping[str, Strategy] | None = None,
) -> list[StrategySpec]:
    """Load every in-process plugin (`load_strategy`; `plugins` supplies ready instances by
    strategy name instead) and resolve intervals, warm-ups, symbols, exit plans, sizing and
    risk checks. Raises `PluginError` or `ConfigError`."""
    specs: list[StrategySpec] = []
    for sc in config.strategies:
        if sc.external:
            assert sc.interval is not None
            assert sc.warmup is not None
            plugin: Strategy | None = None
            context: tuple[str, ...] = tuple(sc.context_symbols)
            interval, warmup, base = sc.interval, sc.warmup, None
        else:
            given = (plugins or {}).get(sc.name)
            plugin = (
                given
                if given is not None
                else load_strategy(sc.source, name=sc.name, params=sc.params)
            )
            if sc.interval is not None and sc.interval is not plugin.interval:
                msg = (
                    f"strategy {sc.name}: interval {sc.interval.value} given, but the plugin "
                    f"trades {plugin.interval.value}"
                )
                raise ConfigError(msg)
            interval = plugin.interval
            warmup = sc.warmup or plugin.warmup
            context = tuple(plugin.context_symbols)
            base = plugin.exit
        specs.append(
            StrategySpec(
                config=sc,
                interval=interval,
                warmup=warmup,
                symbols=tuple(sc.resolve_symbols(data_root, interval)),
                context_symbols=context,
                exit=sc.exit_plan(base),
                plugin=plugin,
                sizing=sc.sizing.to_sizing(),
                checks=tuple(sc.risk.checks()),
            )
        )
    return specs


def feed_subscriptions(
    specs: Sequence[StrategySpec], positions: Sequence[PositionRecord] = ()
) -> dict[Interval, list[str]]:
    """The symbols the feed must follow per interval: traded and context symbols, plus the
    symbols of `positions` (pass `StateStore.live_positions()`: a position whose symbol left
    its strategy's list still needs candles to exit)."""
    out: dict[Interval, dict[str, None]] = defaultdict(dict)
    for spec in specs:
        for symbol in (*spec.symbols, *spec.context_symbols):
            out[spec.interval][symbol] = None
    for position in positions:
        out[position.interval][position.symbol] = None
    return {interval: list(symbols) for interval, symbols in out.items()}


def feed_warmups(specs: Sequence[StrategySpec]) -> dict[Interval, int]:
    """The window length the feed must hold per interval (the largest warm-up)."""
    out: dict[Interval, int] = {}
    for spec in specs:
        out[spec.interval] = max(out.get(spec.interval, 1), spec.warmup)
    return out


# -- results and status -----------------------------------------------------------------------


@dataclass(frozen=True)
class BatchOutcome:
    """What `submit_batch` did with a batch.

    `status`: `on_time` and `late` (accepted and recorded), `duplicate` (a batch for that
    strategy and candle exists) or `refused` (`reason` says why).
    """

    status: str
    reason: str | None = None
    batch_id: int | None = None
    lateness: timedelta | None = None

    @property
    def accepted(self) -> bool:
        return self.status in ("on_time", "late")


@dataclass(frozen=True)
class BookStatus:
    strategy: str
    capital: float
    cash: float
    equity: float
    open_cost: float
    open_positions: int
    day_pnl: float
    day_entries: int


@dataclass(frozen=True)
class EngineStatus:
    """A snapshot for the API: mode, controls, books, positions, open orders, stale symbols
    per interval, the last processed candle per interval, providers and health."""

    mode: str
    running: bool
    controls: Controls
    books: dict[str, BookStatus]
    positions: list[PositionRecord]
    open_orders: list[OrderRecord]
    stale_symbols: dict[str, list[str]]
    last_ticks: dict[str, datetime]
    providers: list[ProviderRecord]
    health: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """The snapshot as JSON-ready values (ISO times, plain numbers and strings)."""
        value = to_json(self)
        assert isinstance(value, dict)
        return value


def to_json(value: object) -> object:
    """JSON-ready form of engine records: dataclasses become dicts, datetimes ISO strings,
    durations seconds, enums their values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Mapping):
        return {str(to_json(k)): to_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [to_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


_EMPTY_WINDOW = pl.DataFrame(schema=KLINE_SCHEMA)


def _signal_row(signal: SignalIn) -> SignalRow:
    override = None if signal.exit is None else signal.exit.model_dump(exclude_none=True)
    return SignalRow(
        symbol=signal.symbol,
        action=signal.action,
        priority=signal.priority,
        weight=signal.weight,
        exit_override=override or None,
        info=dict(signal.info),
    )


def time_exit_id(entry_order_id: str, when: datetime) -> str:
    """The client order ID of the engine's time exit of a position for `when` (one per tick, so
    a failed attempt can be retried at the next)."""
    return child_order_id(entry_order_id, f"time:{as_utc(when).isoformat()}")


def needs_bracket(plan: ExitPlan | None) -> bool:
    """Whether `plan` has a price exit an exchange must hold as an order."""
    return plan is not None and (
        plan.take_profit is not None or plan.stop is not None or plan.trailing is not None
    )


def _is_market_sell(order: OrderRecord) -> bool:
    return order.side is Side.SELL and order.type is OrderType.MARKET


def _effective(plan: ExitPlan | None) -> ExitPlan | None:
    """`None` for a plan without any exit (no bracket is opened for it)."""
    if plan is None or plan == ExitPlan(tp_fill=plan.tp_fill):
        return None
    return plan


# -- the engine -------------------------------------------------------------------------------


class Engine:
    """The trading engine. Build with the run config, the strategies (`build_specs`) and an
    open state store, then `await run(feed)`. See the module docstring for the rules.

    Args:
        clock: aware-UTC "now" (wall clock live; the replay feed's clock in replay).
        replay: wait for every batch instead of the wall-clock deadline.
        history: fetches candles a restarted broker needs that the feed's windows lack (a
            REST client live, the Parquet store in replay).
        step_hook: called at each `Step` boundary (tests inject crashes there).
        broker: the broker to trade through (default: a `PaperExecution` restored from the
            state file).
    """

    def __init__(
        self,
        config: RunConfig,
        specs: Sequence[StrategySpec],
        store: StateStore,
        *,
        clock: Clock = _utc_now,
        replay: bool = False,
        history: KlineFetcher | None = None,
        step_hook: StepHook | None = None,
        broker: ExecutionBroker | None = None,
    ) -> None:
        config.require_runnable()
        names = [s.name for s in specs]
        if sorted(names) != sorted(s.name for s in config.strategies):
            msg = "the specs must be exactly the config's strategies"
            raise ValueError(msg)
        self.config = config
        self.specs: dict[str, StrategySpec] = {s.name: s for s in specs}
        self.store = store
        self.clock = clock
        self.replay = replay
        self.history = history
        self._hook = step_hook
        self._costs = config.account.costs.to_costs()
        self._by_interval: dict[Interval, list[StrategySpec]] = defaultdict(list)
        for spec in sorted(specs, key=lambda s: s.name):
            self._by_interval[spec.interval].append(spec)
        self._listeners: list[TickListener] = []
        self._feed: Feed | None = None
        self._broker: ExecutionBroker = (
            broker
            if broker is not None
            else PaperExecution.create(config.account.initial_cash, self._costs)
        )
        self._reconciliation: Reconciliation | None = None
        self._recovered = False
        self._books: dict[str, Book] = {}
        self._book_day: dict[str, int] = {}
        self._dirty: set[str] = set()
        self._last_ticks: dict[Interval, datetime] = {}
        self._frontier: dict[Interval, datetime] = {}
        self._collecting: dict[Interval, datetime] = {}
        self._received: dict[tuple[Interval, datetime], set[str]] = {}
        self._failed: dict[tuple[Interval, datetime], set[str]] = {}
        """In-process plugins that finished (or could not start) without a batch."""
        self._arrived = asyncio.Event()
        self._busy: dict[str, asyncio.Task[None]] = {}
        self._handed: dict[tuple[str, Interval], datetime] = {}
        self._caught_up: set[Interval] = set()
        self._started = False
        self._running = False
        self._stopping = False
        self._last_tick_at: datetime | None = None
        self._tick_done_at: dict[Interval, datetime] = {}
        self._error: str | None = None
        self._tick_lock = asyncio.Lock()
        """Held while a tick is processed (and while the between-ticks drain runs)."""
        self._drain_task: asyncio.Task[None] | None = None
        self._drain_failure: BaseException | None = None
        self._halt: BrokerHalted | None = None

    # -- public API ---------------------------------------------------------------------------

    @property
    def books(self) -> dict[str, Book]:
        """Each strategy's book (always equal to `StateStore.derive_book` between steps)."""
        return dict(self._books)

    @property
    def broker(self) -> ExecutionBroker:
        return self._broker

    @property
    def reconciliation(self) -> Reconciliation | None:
        """The latest reconciliation report (at start-up, or from the broker after a
        reconnect); `None` before the first."""
        return self._reconciliation

    @property
    def broker_halt(self) -> BrokerHalted | None:
        """Set once the broker reported that it stopped placing orders."""
        return self._halt

    def unprotected_positions(self) -> list[PositionRecord]:
        """Open positions whose exit plan needs a bracket the broker does not hold (it failed,
        or was cancelled for a sell that did not happen), with no sell working."""
        store = self.store
        working = {o.id: o for o in store.pending_orders()}
        selling = {o.symbol for o in working.values() if _is_market_sell(o)}
        return [
            p
            for p in store.live_positions()
            if p.status is PositionStatus.OPEN
            and needs_bracket(p.exit_plan)
            and (p.bracket_id is None or p.bracket_id not in working)
            and p.symbol not in selling
        ]

    @property
    def running(self) -> bool:
        """Whether `run` is processing ticks."""
        return self._running

    @property
    def last_tick_at(self) -> datetime | None:
        """The clock time at which the latest tick (of any interval) was decided, since this
        engine started; `None` before the first."""
        return self._last_tick_at

    def tick_times(self) -> dict[Interval, datetime]:
        """Per interval, the clock time at which its latest tick was decided since this engine
        started (an interval without a tick yet is absent)."""
        return dict(self._tick_done_at)

    def collecting(self, interval: Interval) -> datetime | None:
        """The candle whose batches the engine is collecting now (its tick was published to the
        listeners and is not decided yet), or `None`."""
        return self._collecting.get(interval)

    def add_tick_listener(self, listener: TickListener) -> None:
        """Called with every tick after the broker step, before the engine waits for batches
        (M6 broadcasts it to external providers). May be async; errors become events."""
        self._listeners.append(listener)

    def start(self) -> None:
        """Check the config against the state file, restore the broker and derive the books.
        `run` calls it; idempotent. Raises `ConfigError` for a config the state contradicts."""
        if self._started:
            return
        store = self.store
        stored = {s.name: s for s in store.strategies()}
        restart = bool(stored) or bool(store.last_ticks())
        initial = self.config.account.initial_cash
        known = store.meta("initial_cash")
        if known is not None and float(known) != initial:
            msg = (
                f"account.initial_cash is {initial:g} but the state file {store.path} was "
                f"started with {float(known):g}; use a new state file to change it"
            )
            raise ConfigError(msg)
        for name, record in stored.items():
            spec = self.specs.get(name)
            if spec is None:
                live = store.live_positions(name)
                if live:
                    msg = (
                        f"strategy {name} has {len(live)} open or pending position(s) in "
                        f"{store.path} but is not in the config; put it back (or flatten "
                        "first) before removing it"
                    )
                    raise ConfigError(msg)
            elif record.capital != spec.capital:
                msg = (
                    f"strategy {name}: capital changed from {record.capital:g} to "
                    f"{spec.capital:g}; books are fixed allocations, so use a new state file "
                    "(or a new strategy name) to change it"
                )
                raise ConfigError(msg)
        with store.transaction():
            store.set_meta("initial_cash", repr(initial))
            for spec in self.specs.values():
                store.upsert_strategy(spec.name, spec.config.model_dump(mode="json"), spec.capital)
            if restart:
                ticks = {i.value: t.isoformat() for i, t in store.last_ticks().items()}
                self._event(
                    EventLevel.INFO, "restart", "engine restarted from the state file", ticks
                )
        if isinstance(self._broker, PaperExecution):
            self._broker.restore(store.broker_snapshot(initial))
        elif isinstance(self._broker, RestoresState):
            self._broker.load(
                [
                    WorkingOrder(o.to_order(), o.interval, o.strategy)
                    for o in store.pending_orders()
                ],
                [p.to_paper() for p in store.live_positions() if p.status is PositionStatus.OPEN],
            )
        self._last_ticks = store.last_ticks()
        self._derive(self.specs, self.clock())
        self._started = True

    async def run(self, feed: Feed) -> None:
        """Process the feed's ticks until it ends or `stop` is called. An exception other
        than a plugin's propagates (the daemon then restarts from the state file)."""
        self._feed = feed
        self.start()
        self._running = True
        try:
            await self._recover()
            if self._drains():
                self._drain_task = asyncio.create_task(self._drain_loop(), name="engine-drain")
            async for tick in feed.ticks():
                async with self._tick_lock:
                    await self._process(tick)
                if self._stopping:
                    break
            if self._drain_failure is not None:
                raise self._drain_failure
        except BaseException as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._running = False
            drain, self._drain_task = self._drain_task, None
            if drain is not None:
                drain.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await drain
            for task in self._busy.values():
                task.cancel()
            await feed.aclose()

    def _drains(self) -> bool:
        return not self._broker.needs_candles and self.config.engine.drain_interval is not None

    async def _drain_loop(self) -> None:
        """Between ticks: record what the broker heard (see the module docstring)."""
        interval = self.config.engine.drain_interval
        assert interval is not None
        wait = interval.total_seconds()
        try:
            while not self._stopping:
                await asyncio.sleep(wait)
                if self._tick_lock.locked():
                    continue  # a tick is being processed: its broker step reads the events
                async with self._tick_lock:
                    await self.drain()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._drain_failure = exc
            await self.stop()

    async def drain(self) -> int:
        """Read the broker's events now and record them (one transaction); returns how many.
        The engine calls it between ticks for brokers that do not need candles."""
        events = await self._broker.events(self.clock())
        if events:
            with self.store.transaction():
                for event in events:
                    self._record(event)
            if self._dirty:
                self._derive(sorted(self._dirty), self.clock())
        return len(events)

    async def stop(self) -> None:
        """Stop after the tick in progress."""
        self._stopping = True
        self._arrived.set()
        if self._feed is not None:
            await self._feed.aclose()

    def pause(self) -> None:
        """No new entries from the next decision on; exits keep working."""
        self.store.set_control(Control.PAUSED, True)
        self._event(EventLevel.INFO, "paused", "entries paused")

    def resume(self) -> None:
        self.store.set_control(Control.PAUSED, False)
        self._event(EventLevel.INFO, "resumed", "entries resumed")

    def flatten(self) -> None:
        """At the next tick: sell every open position at market, cancel pending entries, and
        pause."""
        self.store.set_control(Control.FLATTEN_REQUESTED, True)
        self._event(EventLevel.WARNING, "flatten_requested", "flatten requested")

    def record_event(
        self,
        level: EventLevel,
        kind: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Write an event to the state file's event feed (the API reports dropped clients)."""
        self._event(level, kind, message, data)

    def on_feed_event(self, event: FeedEvent) -> None:
        """The feed's event sink: writes the event to the state file's event feed."""
        warning = (FeedDisconnected, SeedFailed, BackfillFailed, SubscriptionFailed)
        level = EventLevel.WARNING if isinstance(event, warning) else EventLevel.INFO
        if isinstance(event, StaleSymbolsChanged) and event.symbols:
            level = EventLevel.WARNING
        data = to_json(event)
        assert isinstance(data, dict)
        message = ", ".join(f"{k}={v}" for k, v in data.items())
        try:
            self._event(level, _snake(type(event).__name__), message, data)
        except Exception:
            logger.exception("could not record a feed event")

    def status(self) -> EngineStatus:
        store = self.store
        now = self.clock()
        stale: dict[str, list[str]] = {}
        health: dict[str, Any] = {
            "running": self._running,
            "error": self._error,
            "last_tick_at": self._last_tick_at,
            "feed": None,
        }
        if self._feed is not None:
            for interval in self._by_interval:
                found = self._feed.stale_symbols(interval, now, self.config.account.stale_after)
                stale[interval.value] = sorted(found)
            health["feed"] = to_json(self._feed.health())
        books = {
            name: BookStatus(
                strategy=name,
                capital=self.specs[name].capital,
                cash=book.cash,
                equity=book.equity,
                open_cost=book.open_cost,
                open_positions=book.open_positions,
                day_pnl=book.day_pnl,
                day_entries=book.day_entries,
            )
            for name, book in self._books.items()
        }
        return EngineStatus(
            mode="replay" if self.replay else self.config.mode,
            running=self._running,
            controls=store.controls(),
            books=books,
            positions=store.live_positions(),
            open_orders=store.pending_orders(),
            stale_symbols=stale,
            last_ticks={i.value: t for i, t in store.last_ticks().items()},
            providers=store.providers(),
            health=health,
        )

    async def submit_batch(
        self, batch: SignalBatch, received_at: datetime | None = None
    ) -> BatchOutcome:
        """Accept an external provider's batch (the API calls this).

        Refused: an unknown or in-process strategy, another interval, a candle that has not
        closed by the clock, or the engine not started. A second batch for the same candle is
        a `duplicate`. A batch for a candle already processed is `late` (entries expired,
        exits sold at the next candle open); any other is `on_time` and joins its candle's
        decisions (it may arrive before the engine has seen the tick).
        """
        received = self.clock() if received_at is None else as_utc(received_at)
        if not self._started:
            return BatchOutcome("refused", "the engine is not running")
        spec = self.specs.get(batch.strategy)
        if spec is None:
            return BatchOutcome("refused", f"unknown strategy {batch.strategy!r}")
        if not spec.external:
            return BatchOutcome("refused", f"strategy {spec.name} runs in process")
        if batch.interval is not spec.interval:
            msg = f"strategy {spec.name} trades {spec.interval.value}, not {batch.interval.value}"
            return BatchOutcome("refused", msg)
        interval, time = spec.interval, batch.candle_time
        if time + interval.length > received:
            return BatchOutcome("refused", f"candle {time.isoformat()} has not closed yet")
        existing = self.store.batch_for(spec.name, interval, time)
        if existing is not None and existing.status is not BatchStatus.MISSED:
            return BatchOutcome("duplicate", "a batch for this candle was already received")
        if existing is not None or self._processed(interval, time):
            record = await self._record_late(spec, batch, received, existing)
            return BatchOutcome("late", None, record.id, record.lateness)
        record = self._record_on_time(spec, batch, received, external=True)
        return BatchOutcome("on_time", None, record.id, record.lateness)

    # -- the tick -----------------------------------------------------------------------------

    def _step(self, step: Step, tick: Tick | None) -> None:
        if self._hook is not None:
            self._hook(step.value, tick)

    async def _process(self, tick: Tick) -> None:
        interval, time = tick.interval, tick.open_time
        last = self._last_ticks.get(interval)
        if last is not None and time <= last:
            return  # already processed (a resumed replay, or a repeated tick)
        if interval not in self._caught_up:
            await self._catch_up(tick)
        await self._broker_step(tick)
        self._step(Step.AFTER_BROKER, tick)
        specs = self._by_interval.get(interval, [])
        await self._collect(tick, specs)
        if self._stopping:
            return  # abandoned: decided again after a restart (its batches are kept)
        self._step(Step.AFTER_SIGNALS, tick)
        await self._decide(tick, specs)
        self._last_tick_at = self._tick_done_at[interval] = self.clock()
        self._step(Step.TICK_DONE, tick)

    async def _catch_up(self, tick: Tick) -> None:
        """First tick of an interval: candles a restored broker needs, and missed ticks."""
        interval, time = tick.interval, tick.open_time
        self._caught_up.add(interval)
        last = self._last_ticks.get(interval)
        if last is not None and time > interval.next(last):
            first = interval.next(last)
            count = interval.count_open_times(first, time)
            self._event(
                EventLevel.WARNING,
                "ticks_missed",
                f"{count} {interval.value} tick(s) missed while down",
                {"interval": interval.value, "from": first.isoformat(), "to": time.isoformat()},
            )
        if not self._broker.needs_candles:
            return
        # A restored broker holds no candles: every series it needs from its start.
        needs = {s: t for s, t in self._broker.candle_needs(interval).items() if t <= time}
        if not needs:
            return
        feed = self._feed
        assert feed is not None
        end = time + interval.length
        for symbol, start in sorted(needs.items()):
            count = interval.count_open_times(start, end) + 1
            frame = feed.windows(interval, [symbol], time, count).get(symbol)
            if frame is None or frame.height == 0 or frame["open_time"][0] > start:
                if self.history is not None:
                    fetched = await self.history.fetch_klines(symbol, interval, start, end)
                    frame = fetched if frame is None else pl.concat([fetched, frame])
                else:
                    self._event(
                        EventLevel.WARNING,
                        "history_short",
                        f"{symbol}: the feed has no candles back to {start.isoformat()}",
                    )
            if frame is None:
                continue
            frame = (
                frame.filter((pl.col("open_time") >= start) & (pl.col("open_time") <= time))
                .unique("open_time", keep="first", maintain_order=True)
                .sort("open_time")
            )
            if frame.height:
                self._broker.add_candles(symbol, Candles.from_frame(frame, interval, ended=False))
                self._handed[symbol, interval] = frame["open_time"][-1]

    def _hand_over(self, tick: Tick, symbol: str, start: datetime) -> None:
        interval, time = tick.interval, tick.open_time
        key = (symbol, interval)
        if start > time:
            return  # an order for a later candle: nothing of it can have closed
        candle = tick.candles.get(symbol)
        previous = self._handed.get(key)
        if (
            candle is not None
            and previous is not None
            and previous == time - interval.length
            and start <= previous
        ):
            self._broker.add_candles(symbol, Candles.from_bars([candle], interval, ended=False))
            self._handed[key] = time
            return
        feed = self._feed
        assert feed is not None
        count = min(interval.count_open_times(start, time + interval.length), HISTORY_SLACK)
        frame = feed.windows(interval, [symbol], time, count).get(symbol)
        if frame is None:
            return
        frame = frame.filter(pl.col("open_time") >= start)
        if frame.height:
            self._broker.add_candles(symbol, Candles.from_frame(frame, interval, ended=False))
            self._handed[key] = frame["open_time"][-1]

    async def _broker_step(self, tick: Tick) -> None:
        interval, time = tick.interval, tick.open_time
        for order in self.store.pending_orders():
            if order.status is OrderStatus.UNKNOWN:
                await self._resolve(order, tick)
        broker = self._broker
        if broker.needs_candles:
            needed = broker.candle_needs(interval)
            for symbol, start in needed.items():
                self._hand_over(tick, symbol, start)
            for key in [k for k in self._handed if k[1] is interval and k[0] not in needed]:
                del self._handed[key]
        events = await broker.events(time + interval.length, interval=interval)
        with self.store.transaction():
            for event in events:
                self._record(event)
        if broker.needs_candles:
            await self._cancel_overdue(interval)
        if not broker.manages_time_exits:
            await self._time_exits(tick)
        if self._dirty:
            self._derive(sorted(self._dirty), self.clock())

    def _record(self, event: BrokerEvent) -> None:
        """Write one broker event to the state file (inside the caller's transaction). An event
        for an order already final there (reported twice) is ignored."""
        store = self.store
        if isinstance(event, EntryFilled):
            if not self._still_working(event.fill.order_id, event):
                return
            position = store.position_for_order(event.fill.order_id)
            assert position is not None
            store.mark_order_filled(event.fill.order_id)
            store.record_fill(event.fill)
            paper = event.position
            bracket = paper.bracket()
            if bracket is not None:
                store.insert_order(
                    OrderRecord.from_order(bracket, position.strategy, paper.interval)
                )
            store.open_position(
                position.id,
                entry_time=paper.entry_time,
                entry_ref=paper.entry_ref,
                entry_price=paper.entry_price,
                quantity=paper.quantity,
                cost=paper.cost,
                bracket_id=paper.bracket_id,
            )
            self._dirty.add(position.strategy)
        elif isinstance(event, ExitFilled):
            if not event.closed:  # pragma: no cover - the engine only sells whole positions
                msg = f"unexpected partial sell of {event.fill.symbol}"
                raise AssertionError(msg)
            if not self._still_working(event.fill.order_id, event):
                return
            position = store.live_position(event.fill.symbol)
            assert position is not None
            assert position.status is PositionStatus.OPEN
            order = store.order(event.fill.order_id)
            assert order is not None
            store.mark_order_filled(order.id)
            store.record_fill(event.fill)
            bracket_id = position.bracket_id
            if bracket_id is not None and bracket_id != order.id:
                held = store.order(bracket_id)
                if held is not None and held.status.working:
                    store.mark_order_cancelled(bracket_id, "position_sold")
            store.close_position(
                position.id,
                exit_time=event.fill.time,
                exit_price=event.fill.price,
                exit_reason=self._exit_reason(event, order, position),
                proceeds=event.proceeds,
            )
            self._dirty.add(position.strategy)
        elif isinstance(event, OrderCancelled):
            if not self._still_working(event.order.id, event):
                return
            store.mark_order_cancelled(event.order.id, event.reason.value)
            if event.order.side is Side.BUY:
                position = store.position_for_order(event.order.id)
                assert position is not None
                store.cancel_position(position.id, event.time, event.reason.value)
                self._dirty.add(position.strategy)
        elif isinstance(event, OrderRejection):
            order = store.order(event.order_id)
            if order is None or not self._still_working(event.order_id, event):
                return
            self._rejected(order, event.reason, event.time)
            self._event(
                EventLevel.WARNING,
                "order_rejected",
                f"{order.symbol}: the broker rejected {order.side.value} order {order.id} "
                f"after accepting it: {event.reason}",
                {"order_id": order.id, "symbol": order.symbol, "reason": event.reason},
            )
        elif isinstance(event, BracketFailed):
            self._event(
                EventLevel.ERROR,
                "bracket_failed",
                f"{event.symbol}: the bracket could not be placed ({event.reason}); the "
                "position is unprotected until it is",
                {"symbol": event.symbol, "entry_order_id": event.entry_order_id},
            )
        elif isinstance(event, BracketPlaced):
            position = store.position_for_order(event.position.entry_order_id)
            bracket = event.position.bracket()
            if position is None or position.status is not PositionStatus.OPEN or bracket is None:
                logger.warning("ignored a bracket for %s: no open position", event.position)
                return
            if store.order(bracket.id) is None:
                store.insert_order(
                    OrderRecord.from_order(bracket, position.strategy, position.interval)
                )
            store.set_position_bracket(position.id, bracket.id)
            self._event(
                EventLevel.INFO,
                "bracket_placed",
                f"{position.symbol}: bracket {bracket.id} placed",
                {"symbol": position.symbol, "bracket_id": bracket.id},
            )
        elif isinstance(event, BrokerHalted):
            self._halt = event
            self._event(
                EventLevel.ERROR,
                "broker_halted",
                f"the broker stopped placing orders ({event.reason}): {event.detail}",
                {"reason": event.reason},
            )
        else:
            self._apply_reconciliation(event)

    def _still_working(self, order_id: str, event: BrokerEvent) -> bool:
        order = self.store.order(order_id)
        if order is not None and order.status.working:
            return True
        logger.warning("ignored a broker event for an order that is not working: %s", event)
        return False

    def _exit_reason(self, event: ExitFilled, order: OrderRecord, position: PositionRecord) -> str:
        if event.fill.reason is not None:
            return event.fill.reason.value
        if order.signal_id is not None:
            return "signal"
        if order.id == time_exit_id(position.entry_order_id, order.time):
            return "time"
        return "flatten"

    async def _cancel_overdue(self, interval: Interval) -> None:
        """Cancel pending buys whose entry candle is overdue by more than `entry_grace`."""
        now = self.clock()
        grace = self.config.engine.entry_grace
        cancelled: list[str] = []
        for order in self.store.pending_orders():
            if order.interval is not interval or order.side is not Side.BUY:
                continue
            if order.status is not OrderStatus.PENDING:
                continue
            if now < order.time + interval.length + grace:
                continue
            result = await self._broker.cancel(order.to_order())
            with self.store.transaction():
                for event in result.events:
                    self._record(event)
                if result.status is not CancelStatus.CANCELLED:
                    continue
                self.store.mark_order_cancelled(order.id, "no_candle")
                position = self.store.position_for_order(order.id)
                assert position is not None
                self.store.cancel_position(position.id, now, "no_candle")
                self._dirty.add(position.strategy)
            cancelled.append(order.symbol)
        if cancelled:
            self._event(
                EventLevel.WARNING,
                "entries_cancelled",
                f"{len(cancelled)} pending entr(ies) cancelled: no candle after the grace",
                {"symbols": cancelled},
            )

    async def _time_exits(self, tick: Tick) -> None:
        """For a broker without time exits: sell every open position on the tick's interval
        whose holding time is up by the entry time *E*."""
        interval = tick.interval
        when = tick.open_time + interval.length
        for position in self.store.live_positions():
            plan = position.exit_plan
            if position.status is not PositionStatus.OPEN or position.interval is not interval:
                continue
            if plan is None or plan.max_hold is None:
                continue
            assert position.entry_time is not None
            if when < position.entry_time + plan.max_hold or self._sell_pending(position.symbol):
                continue
            order = Order(
                position.symbol,
                Side.SELL,
                when,
                id=time_exit_id(position.entry_order_id, when),
            )
            if self.store.order(order.id) is not None:
                continue  # tried at this tick already
            record = replace(
                OrderRecord.from_order(order, position.strategy, position.interval),
                status=OrderStatus.NEW,
            )
            self.store.insert_order(record)
            await self._sell(record, tick)

    # -- orders: intent, action, outcome ------------------------------------------------------

    async def _submit(self, record: OrderRecord, tick: Tick | None) -> SubmitResult:
        """Submit an order whose intent is committed, and record the outcome."""
        result = await self._broker.submit(record.to_order(), interval=record.interval)
        self._step(Step.AFTER_BROKER_CALL, tick)
        with self.store.transaction():
            if result.status is SubmitStatus.ACCEPTED:
                self.store.mark_order_accepted(record.id)
                for event in result.events:
                    self._record(event)
            elif result.status is SubmitStatus.REJECTED:
                self._rejected(record, result.reason or "rejected", self.clock())
                if record.side is Side.SELL:
                    self._event(
                        EventLevel.WARNING,
                        "order_rejected",
                        f"{record.symbol}: the broker rejected sell {record.id}: {result.reason}",
                        {"order_id": record.id, "symbol": record.symbol, "reason": result.reason},
                    )
            else:
                self.store.mark_order_unknown(record.id)
                self._event(
                    EventLevel.WARNING,
                    "order_unknown",
                    f"{record.symbol}: no answer for {record.side.value} order {record.id}; "
                    "it is queried at the next tick",
                    {"order_id": record.id, "symbol": record.symbol},
                )
        return result

    def _rejected(self, order: OrderRecord, reason: str, time: datetime) -> None:
        """The order was refused (at submit or later): an entry's position is cancelled (its
        stake released) and its signal rejected with `broker_<reason>`."""
        store = self.store
        store.mark_order_rejected(order.id, reason)
        if order.side is Side.BUY:
            position = store.position_for_order(order.id)
            if position is not None and position.status is PositionStatus.PENDING:
                store.cancel_position(position.id, time, f"broker_{reason}")
                self._dirty.add(position.strategy)
        if order.signal_id is not None:
            store.set_signal_outcome(order.signal_id, SignalOutcome.REJECTED, f"broker_{reason}")

    async def _sell(self, record: OrderRecord, tick: Tick | None) -> None:
        """Sell a whole position, its intent (`record`) committed: cancel the bracket first if
        the broker's sell does not, then submit."""
        self._step(Step.BEFORE_BROKER_CALL, tick)
        store = self.store
        if not self._broker.sell_cancels_bracket:
            position = store.live_position(record.symbol)
            bracket = self._live_bracket(position)
            if bracket is not None:
                result = await self._broker.cancel(bracket.to_order())
                with store.transaction():
                    for event in result.events:
                        self._record(event)
                    if result.status is CancelStatus.CANCELLED:
                        store.mark_order_cancelled(bracket.id, "position_sold")
                position = store.live_position(record.symbol)
                if position is None or position.status is not PositionStatus.OPEN:
                    with store.transaction():
                        store.mark_order_cancelled(record.id, "position_closed")
                    return  # the bracket filled first: nothing left to sell
                if self._live_bracket(position) is not None:
                    with store.transaction():
                        self._rejected(record, "bracket_open", self.clock())
                        self._event(
                            EventLevel.WARNING,
                            "bracket_cancel_failed",
                            f"{record.symbol}: the bracket could not be cancelled "
                            f"({result.status.value}); not sold",
                            {"order_id": record.id, "symbol": record.symbol},
                        )
                    return
        await self._submit(record, tick)

    def _live_bracket(self, position: PositionRecord | None) -> OrderRecord | None:
        if position is None or position.bracket_id is None:
            return None
        if position.status is not PositionStatus.OPEN:
            return None
        bracket = self.store.order(position.bracket_id)
        return bracket if bracket is not None and bracket.status.working else None

    async def _resolve(self, order: OrderRecord, tick: Tick | None) -> None:
        """Find out what became of a `new` or `unknown` order (`broker.query`)."""
        store = self.store
        result = await self._broker.query(order.to_order())
        if result.state is OrderState.NOT_FOUND and _is_market_sell(order):
            position = store.live_position(order.symbol)
            if (
                position is not None
                and position.status is PositionStatus.OPEN
                and position.strategy == order.strategy
            ):
                self._event(
                    EventLevel.WARNING,
                    "order_resubmitted",
                    f"{order.symbol}: sell {order.id} was never placed; placing it now",
                    {"order_id": order.id, "symbol": order.symbol},
                )
                await self._sell(order, tick)  # finish the exit it was meant for
                return
        with store.transaction():
            state = result.state
            if state in (OrderState.OPEN, OrderState.FILLED, OrderState.CANCELLED):
                store.mark_order_accepted(order.id)
                for event in result.events:
                    self._record(event)
            elif state is OrderState.REJECTED:
                self._rejected(order, result.reason or "rejected", self.clock())
            elif state is OrderState.NOT_FOUND:
                self._rejected(order, "not_placed", self.clock())
                self._event(
                    EventLevel.WARNING,
                    "order_not_placed",
                    f"{order.symbol}: {order.side.value} order {order.id} was never placed",
                    {"order_id": order.id, "symbol": order.symbol},
                )
            elif order.status is OrderStatus.NEW:
                store.mark_order_unknown(order.id)
        if self._dirty:
            self._derive(sorted(self._dirty), self.clock())

    async def _recover(self) -> None:
        """Before the first tick: resolve orders a crash or a lost answer left `new` or
        `unknown`, then reconcile the broker with the state file."""
        if self._recovered:
            return
        self._recovered = True
        store = self.store
        for order in store.pending_orders():
            if order.status in (OrderStatus.NEW, OrderStatus.UNKNOWN):
                await self._resolve(order, None)
        working = [
            WorkingOrder(o.to_order(), o.interval, o.strategy)
            for o in store.pending_orders()
            if o.status is OrderStatus.PENDING
        ]
        positions = [
            p.to_paper() for p in store.live_positions() if p.status is PositionStatus.OPEN
        ]
        report = await self._broker.reconcile(working, positions, now=self.clock())
        with store.transaction():
            self._apply_reconciliation(report)

    def _apply_reconciliation(self, report: Reconciliation) -> None:
        """Record a reconciliation's findings (inside a transaction); a balance mismatch pauses
        entries. The health monitor raises the alerts while the latest report has them."""
        self._reconciliation = report
        for orphan in report.orphans:
            self._event(
                EventLevel.WARNING,
                "orphan_order",
                f"{orphan.symbol}: open {orphan.side.value} order {orphan.order_id} on the "
                f"exchange is not in the state file; left alone",
                {"order_id": orphan.order_id, "symbol": orphan.symbol, "detail": orphan.detail},
            )
        for mismatch in report.mismatches:
            self._event(
                EventLevel.ERROR,
                "balance_mismatch",
                f"{mismatch.asset}: the exchange holds {mismatch.actual:g}, the open positions "
                f"need {mismatch.expected:g}; entries paused",
                {
                    "asset": mismatch.asset,
                    "expected": mismatch.expected,
                    "actual": mismatch.actual,
                },
            )
        if report.mismatches and not self.store.controls().paused:
            self.store.set_control(Control.PAUSED, True)
            self._event(EventLevel.WARNING, "paused", "entries paused: balance mismatch")

    # -- signals ------------------------------------------------------------------------------

    async def _notify(self, tick: Tick) -> None:
        for listener in list(self._listeners):
            try:
                result = listener(tick)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self._event(
                    EventLevel.ERROR,
                    "listener_error",
                    f"a tick listener failed: {type(exc).__name__}: {exc}",
                )

    def _plugin_inputs(
        self, spec: StrategySpec, time: datetime
    ) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
        if not getattr(spec.plugin, "needs_windows", True):
            return {}, {}
        feed = self._feed
        assert feed is not None
        interval = spec.interval
        limit = time - self.config.account.stale_after * interval.length
        windows = {
            symbol: window
            for symbol, window in feed.windows(interval, spec.symbols, time, spec.warmup).items()
            if window["open_time"][-1] >= limit
        }
        raw = feed.windows(interval, spec.context_symbols, time, spec.warmup)
        context = {s: raw.get(s, _EMPTY_WINDOW) for s in spec.context_symbols}
        return windows, context

    async def _call_plugin(
        self,
        spec: StrategySpec,
        time: datetime,
        windows: dict[str, pl.DataFrame],
        context: dict[str, pl.DataFrame],
    ) -> None:
        plugin = spec.plugin
        assert plugin is not None
        try:
            frame = await asyncio.to_thread(plugin.on_candles, windows, time, context)
            batch = frame_to_batch(
                frame, strategy=spec.name, interval=spec.interval, candle_time=time
            )
        except Exception as exc:
            self._event(
                EventLevel.ERROR,
                "plugin_error",
                f"{spec.name}: on_candles for {time.isoformat()} failed: "
                f"{type(exc).__name__}: {exc}",
                {"strategy": spec.name, "candle_time": time.isoformat()},
            )
            self._give_up(spec, time)
            return
        if self._collecting.get(spec.interval) != time:
            self._event(
                EventLevel.WARNING,
                "plugin_late",
                f"{spec.name}: signals for {time.isoformat()} came after the deadline; dropped",
                {"strategy": spec.name, "candle_time": time.isoformat()},
            )
            return
        self._record_on_time(spec, batch, self.clock())

    def _give_up(self, spec: StrategySpec, time: datetime) -> None:
        """No batch will come from this plugin for `time`: stop waiting for it."""
        self._failed.setdefault((spec.interval, time), set()).add(spec.name)
        self._arrived.set()

    def _start_plugin(self, spec: StrategySpec, tick: Tick) -> None:
        time = tick.open_time
        busy = self._busy.get(spec.name)
        if busy is not None and not busy.done():
            self._event(
                EventLevel.ERROR,
                "plugin_busy",
                f"{spec.name}: still computing an earlier candle; {time.isoformat()} missed",
                {"strategy": spec.name, "candle_time": time.isoformat()},
            )
            self._give_up(spec, time)
            return
        if not any(symbol in tick.candles for symbol in spec.symbols):
            empty = SignalBatch(
                contract=1, strategy=spec.name, interval=spec.interval, candle_time=time
            )
            self._record_on_time(spec, empty, self.clock())
            return
        windows, context = self._plugin_inputs(spec, time)
        task = asyncio.create_task(self._call_plugin(spec, time, windows, context))
        self._busy[spec.name] = task

        def done(finished: asyncio.Task[None], name: str = spec.name) -> None:
            if self._busy.get(name) is finished:
                del self._busy[name]

        task.add_done_callback(done)

    async def _collect(self, tick: Tick, specs: Sequence[StrategySpec]) -> None:
        interval, time = tick.interval, tick.open_time
        key = (interval, time)
        self._frontier[interval] = time
        self._collecting[interval] = time
        have = self._received.setdefault(key, set())
        have.update(b.strategy for b in self.store.batches_for(interval, time))
        failed = self._failed.setdefault(key, set())
        try:
            await self._notify(tick)
            for spec in specs:
                if spec.plugin is not None and spec.name not in have:
                    self._start_plugin(spec, tick)
            expected = {s.name for s in specs}
            loop = asyncio.get_running_loop()
            if self.replay:
                wait = self.config.engine.replay_timeout.total_seconds()
            else:
                deadline = time + interval.length + self.config.engine.signal_deadline
                wait = (deadline - self.clock()).total_seconds()
            until = loop.time() + max(0.0, wait)
            while not expected <= have | failed and not self._stopping:
                remaining = until - loop.time()
                if remaining <= 0:
                    break
                self._arrived.clear()
                if expected <= have | failed:
                    break
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._arrived.wait(), remaining)
        finally:
            del self._collecting[interval]
        for spec in specs:
            task = self._busy.get(spec.name)
            if spec.name not in have and task is not None and not task.done():
                self._event(
                    EventLevel.ERROR,
                    "plugin_timeout",
                    f"{spec.name}: no signals for {time.isoformat()} by the deadline",
                    {"strategy": spec.name, "candle_time": time.isoformat()},
                )

    def _processed(self, interval: Interval, time: datetime) -> bool:
        last = self._last_ticks.get(interval)
        if last is not None and time <= last:
            return True
        frontier = self._frontier.get(interval)
        if frontier is None:
            return False
        return time < frontier or (time == frontier and interval not in self._collecting)

    def _record_on_time(
        self, spec: StrategySpec, batch: SignalBatch, received: datetime, *, external: bool = False
    ) -> BatchRecord:
        interval, time = spec.interval, batch.candle_time
        with self.store.transaction():
            record = self.store.record_batch(
                spec.name,
                interval,
                time,
                BatchStatus.ON_TIME,
                received_at=received,
                lateness=received - (time + interval.length),
                signals=[_signal_row(s) for s in batch.signals],
            )
            if external:
                self.store.provider_seen(spec.name, received, time)
        self._received.setdefault((interval, time), set()).add(spec.name)
        self._arrived.set()
        return record

    async def _record_late(
        self,
        spec: StrategySpec,
        batch: SignalBatch,
        received: datetime,
        missed: BatchRecord | None,
    ) -> BatchRecord:
        """Record a late batch (filling in its `missed` record if there is one): entries
        expire, exits become market sells for the next candle open (their intents are written
        with the batch, then submitted)."""
        interval, time = spec.interval, batch.candle_time
        when = interval.ceil(received)
        lateness = received - (time + interval.length)
        rows = [_signal_row(s) for s in batch.signals]
        sells: list[OrderRecord] = []
        with self.store.transaction():
            if missed is not None:
                record = self.store.fill_missed_batch(
                    missed.id, received_at=received, lateness=lateness, signals=rows
                )
            else:
                record = self.store.record_batch(
                    spec.name,
                    interval,
                    time,
                    BatchStatus.LATE,
                    received_at=received,
                    lateness=lateness,
                    signals=rows,
                )
            self.store.provider_seen(spec.name, received, time)
            for signal in record.signals:
                if signal.action == "enter":
                    self.store.set_signal_outcome(signal.id, SignalOutcome.EXPIRED, "late")
                else:
                    outcome, reason, order = self._exit_intent(spec, signal, time, when)
                    self.store.set_signal_outcome(signal.id, outcome, reason)
                    if order is not None:
                        sells.append(order)
            self._event(
                EventLevel.WARNING,
                "batch_late",
                f"{spec.name}: batch for {time.isoformat()} arrived late; entries expired",
                {
                    "strategy": spec.name,
                    "candle_time": time.isoformat(),
                    "lateness_s": (received - (time + interval.length)).total_seconds(),
                },
            )
        for order in sells:
            await self._sell(order, None)
        return record

    # -- decisions ----------------------------------------------------------------------------

    async def _decide(self, tick: Tick, specs: Sequence[StrategySpec]) -> None:
        store = self.store
        interval, time = tick.interval, tick.open_time
        entry_time = time + interval.length
        feed = self._feed
        assert feed is not None
        with store.transaction():
            batches = store.batches_for(interval, time)
            have = {b.strategy for b in batches}
            missing = [s.name for s in specs if s.name not in have]
            for name in missing:
                store.record_batch(name, interval, time, BatchStatus.MISSED)
            if missing:
                self._event(
                    EventLevel.WARNING,
                    "batches_missed",
                    f"no batch for {time.isoformat()} from {', '.join(missing)}",
                    {"strategies": missing, "candle_time": time.isoformat()},
                )
            controls = store.controls()
        paused = controls.paused
        if controls.flatten_requested:
            await self._flatten(tick, time, entry_time)
            paused = True
        self._derive_for(entry_time)
        on_time = [b for b in batches if b.status is BatchStatus.ON_TIME]
        self._resume_entries(on_time, time, entry_time)
        for batch in on_time:
            spec = self.specs[batch.strategy]
            for signal in batch.signals:
                if signal.action == "exit" and signal.outcome is None:
                    with store.transaction():
                        outcome, reason, order = self._exit_intent(spec, signal, time, entry_time)
                        store.set_signal_outcome(signal.id, outcome, reason)
                    if order is not None:
                        await self._sell(order, tick)
        entries = sorted(
            (
                (signal.priority, batch.strategy, signal.symbol, signal.id, signal)
                for batch in on_time
                for signal in batch.signals
                if signal.action == "enter" and signal.outcome is None
            ),
            key=lambda e: e[:4],
        )
        stale = feed.stale_symbols(interval, self.clock(), self.config.account.stale_after)
        for _, name, _, _, signal in entries:
            reason = await self._enter(self.specs[name], signal, tick, paused, stale)
            if reason is not None:
                store.set_signal_outcome(signal.id, SignalOutcome.REJECTED, reason)
        with store.transaction():
            # The refusals of this tick's entries, read back so a decision step resumed after
            # a crash still sums all of them.
            refused: Counter[str] = Counter(
                signal.reason or ""
                for batch in store.batches_for(interval, time)
                if batch.status is BatchStatus.ON_TIME
                for signal in batch.signals
                if signal.action == "enter" and signal.outcome is SignalOutcome.REJECTED
            )
            if refused:
                self._event(
                    EventLevel.INFO,
                    "entries_refused",
                    f"{sum(refused.values())} entr(ies) refused for {time.isoformat()}",
                    {"candle_time": time.isoformat(), "reasons": dict(sorted(refused.items()))},
                )
            self._step(Step.BEFORE_TICK_MARKER, tick)
            store.set_last_tick(interval, time)
        self._last_ticks[interval] = time
        self._received.pop((interval, time), None)
        self._failed.pop((interval, time), None)
        if self._dirty:
            self._derive(sorted(self._dirty), entry_time)

    def _resume_entries(
        self, batches: Sequence[BatchRecord], time: datetime, entry_time: datetime
    ) -> None:
        """A decision step resumed after a crash finds some of the tick's entries placed: give
        each such book exactly the numbers the interrupted step had (derived without those
        entries, which are then opened one by one in decision order), so the remaining entries
        are sized to the last bit as in an uninterrupted run."""
        placed = sorted(
            (signal.priority, batch.strategy, signal.symbol, signal.id)
            for batch in batches
            for signal in batch.signals
            if signal.action == "enter" and signal.outcome is SignalOutcome.PLACED
        )
        held: dict[str, list[PositionRecord]] = {}
        for _, name, symbol, _ in placed:
            position = self.store.position_for_order(client_order_id(name, symbol, time, Side.BUY))
            if position is not None and position.status in (
                PositionStatus.PENDING,
                PositionStatus.OPEN,
            ):
                held.setdefault(name, []).append(position)
        for name, positions in held.items():
            spec = self.specs[name]
            book = self.store.derive_book(
                name, spec.capital, entry_time, exclude={p.id for p in positions}
            )
            for position in positions:
                book.open(position.symbol, position.cost, entry_time)
            self._books[name] = book
            self._book_day[name] = utc_day(entry_time)
            self._dirty.add(name)

    def _exit_intent(
        self,
        spec: StrategySpec,
        signal: SignalRecord,
        time: datetime,
        when: datetime,
    ) -> tuple[SignalOutcome, str | None, OrderRecord | None]:
        """The intent of a market sell of the strategy's position in the signal's symbol for
        `when` (written; the caller sells it), or why there is none."""
        store = self.store
        position = store.live_position(signal.symbol)
        if position is None or position.strategy != spec.name:
            return SignalOutcome.IGNORED, "not_held", None
        if position.status is PositionStatus.PENDING:
            return SignalOutcome.IGNORED, "entry_pending", None
        if self._sell_pending(signal.symbol):
            return SignalOutcome.IGNORED, "sell_pending", None
        order = Order(
            signal.symbol,
            Side.SELL,
            when,
            id=client_order_id(spec.name, signal.symbol, time, Side.SELL),
        )
        if store.order(order.id) is not None:
            return SignalOutcome.IGNORED, "duplicate", None
        record = replace(
            OrderRecord.from_order(order, spec.name, position.interval, signal.id),
            status=OrderStatus.NEW,
        )
        store.insert_order(record)
        return SignalOutcome.PLACED, None, record

    def _sell_pending(self, symbol: str) -> bool:
        """Whether a market sell of `symbol` is working (new, pending or unknown)."""
        return any(o.symbol == symbol and _is_market_sell(o) for o in self.store.pending_orders())

    async def _flatten(self, tick: Tick, time: datetime, entry_time: datetime) -> None:
        """Cancel pending entries and sell every open position at market; then pause and clear
        the request (the last transaction, so a crash in between flattens again)."""
        store = self.store
        sold: list[str] = []
        cancelled: list[str] = []
        for position in store.live_positions():
            if position.status is PositionStatus.PENDING:
                entry = store.order(position.entry_order_id)
                assert entry is not None
                if entry.status is not OrderStatus.PENDING:
                    continue  # new or unknown: resolved first, flattened again then
                result = await self._broker.cancel(entry.to_order())
                with store.transaction():
                    for event in result.events:
                        self._record(event)
                    if result.status is CancelStatus.CANCELLED:
                        store.mark_order_cancelled(position.entry_order_id, "flatten")
                        store.cancel_position(position.id, entry_time, "flatten")
                        self._dirty.add(position.strategy)
                        cancelled.append(position.symbol)
                current = store.position(position.id)
                if current is None or current.status is not PositionStatus.OPEN:
                    continue
                position = current  # it filled before the cancel: sell it too
            if self._sell_pending(position.symbol):
                continue
            base = client_order_id(position.strategy, position.symbol, time, Side.SELL)
            order = Order(
                position.symbol, Side.SELL, entry_time, id=child_order_id(base, "flatten")
            )
            if store.order(order.id) is not None:
                continue  # tried at this tick already
            record = replace(
                OrderRecord.from_order(order, position.strategy, position.interval),
                status=OrderStatus.NEW,
            )
            store.insert_order(record)
            await self._sell(record, tick)
            sold.append(position.symbol)
        with store.transaction():
            self._event(
                EventLevel.WARNING,
                "flatten",
                f"flatten: {len(sold)} sell(s) for {entry_time.isoformat()}, "
                f"{len(cancelled)} pending entr(ies) cancelled; entries paused",
                {"sold": sold, "cancelled": cancelled},
            )
            store.set_control(Control.PAUSED, True)
            store.set_control(Control.FLATTEN_REQUESTED, False)

    def _entry_plan(self, spec: StrategySpec, signal: SignalRecord) -> ExitPlan | None:
        if not signal.exit_override:
            return spec.exit
        return _effective(ExitPlanIn.model_validate(signal.exit_override).to_plan(spec.exit))

    async def _enter(
        self,
        spec: StrategySpec,
        signal: SignalRecord,
        tick: Tick,
        paused: bool,
        stale: set[str],
    ) -> str | None:
        """Place an entry, or return the first check that refuses it (or the broker's
        `broker_<reason>` when it rejects the order)."""
        store = self.store
        symbol = signal.symbol
        time = tick.open_time
        entry_time = time + spec.interval.length
        if paused:
            return "paused"
        if symbol not in spec.symbols:
            return "unknown_symbol"
        if store.live_position(symbol) is not None:
            return "symbol_held"
        if symbol in stale:
            return "stale"
        book = self._books[spec.name]
        book.roll_day(entry_time)
        if isinstance(spec.sizing, WeightedSizing):
            weight = 1.0 if signal.weight is None else signal.weight
            stake = spec.sizing.stake_weighted(book, weight)
        else:
            stake = spec.sizing.stake(book)
        reason = first_rejection(spec.checks, book, symbol, stake)
        if reason is not None:
            return reason
        if not 0 < stake <= book.cash * (1 + _TOLERANCE):
            return "cash"
        stake = min(stake, book.cash)
        account = self.config.account
        if account.max_order_usdt is not None and stake > account.max_order_usdt * (
            1 + _EXPOSURE_TOLERANCE
        ):
            return "max_order_usdt"
        if account.max_total_exposure is not None:
            open_cost = sum(b.open_cost for b in self._books.values())
            equity = sum(b.equity for b in self._books.values())
            limit = account.max_total_exposure * equity * (1 + _EXPOSURE_TOLERANCE)
            if equity <= 0 or open_cost + stake > limit:
                return "max_total_exposure"
        plan = self._entry_plan(spec, signal)
        order = Order(
            symbol,
            Side.BUY,
            entry_time,
            quote_amount=stake,
            exit_plan=plan,
            id=client_order_id(spec.name, symbol, time, Side.BUY),
        )
        record = replace(
            OrderRecord.from_order(order, spec.name, spec.interval, signal.id),
            status=OrderStatus.NEW,
        )
        with store.transaction():
            store.insert_order(record)
            store.insert_position(
                spec.name,
                symbol,
                spec.interval,
                entry_time,
                order.id,
                stake,
                exit_plan=plan,
                signal_info=signal.info or None,
            )
            store.set_signal_outcome(signal.id, SignalOutcome.PLACED)
        self._step(Step.BEFORE_BROKER_CALL, tick)
        result = await self._submit(record, tick)
        if result.status is SubmitStatus.REJECTED:
            return f"broker_{result.reason or 'rejected'}"
        if result.events:
            self._derive([spec.name], entry_time)  # it filled already: its real cost
        else:
            book.open(symbol, stake, entry_time)
            self._dirty.add(spec.name)
        return None

    # -- books and events ---------------------------------------------------------------------

    def _derive(self, names: Sequence[str] | Mapping[str, object], now: datetime) -> None:
        for name in names:
            spec = self.specs[name]
            self._books[name] = self.store.derive_book(name, spec.capital, now)
            self._book_day[name] = utc_day(now)
            self._dirty.discard(name)

    def _derive_for(self, now: datetime) -> None:
        """Before decisions at `now`: re-derive books that changed or are of another day."""
        day = utc_day(now)
        stale = [n for n in self.specs if n in self._dirty or self._book_day.get(n) != day]
        if stale:
            self._derive(stale, now)

    def _event(
        self,
        level: EventLevel,
        kind: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.store.add_event(self.clock(), level, kind, message, data)
