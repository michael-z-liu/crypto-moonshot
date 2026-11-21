"""The Binance spot broker: an `ExecutionBroker` that trades on the exchange (built for the
testnet; see `docs/architecture.md`, "Exchange broker", and `docs/binance-spot-api.md`).

It places real orders through the signed REST client and learns what happens to them from the
WebSocket API user data stream, with REST polling behind it. Prices and quantities are `Decimal`
inside; they become floats only in the events the engine records.

Orders
------
- **Entries**: a `MARKET` buy by `quoteOrderQty` (the stake, rounded down to 8 decimals), with
  the engine's order ID as `newClientOrderId`, refused below the symbol's minimum notional
  (`min_notional`). Its fill becomes `EntryFilled`: the average fill price is the position's
  reference price (exit levels are relative to where the exchange actually filled); the
  quantity is what was bought minus a commission charged in the base asset, rounded down to
  the lot step; the rest is dust, kept on the account and never sold; the cost is the USDT
  spent. A commission in another asset (BNB) is logged and kept in `other_commissions`, not
  converted. The position's entry time is the order's (the engine's entry time), so time
  exits fall on the same tick as in paper mode.
- **Brackets**: right after an entry fills, its exit plan is placed: an OCO sell list
  (`LIMIT_MAKER` take profit above, rounded up to the tick; `STOP_LOSS` below, the stop rounded
  down, or `trailingDelta` basis points for a trailing stop), or a single `LIMIT_MAKER` /
  `STOP_LOSS` when only one level exists. The list's `listClientOrderId` (or the single
  order's ID) is the position's bracket ID, `child_order_id(entry, "bracket")` (then
  `bracket:<n>` for later attempts), and the legs are `child_order_id(bracket, "take_profit" |
  "stop")`. A level already crossed (`-2010` would take, `-2021` would trigger, or the price
  relationship refused while the last price is beyond a level) sells the position at market
  at once, under the bracket's ID, with that level's exit reason. Any other failure (a filter
  such as `PERCENT_PRICE_BY_SIDE`, the balance, the rate limit) reports `EntryFilled` without
  a bracket and `BracketFailed`; `events` retries with backoff and reports `BracketPlaced`.
  Plans the exchange cannot hold are refused (`exit_plan_problems`): `tp_fill = "close"`, and
  a stop tighter than the trailing stop (only one of them can be live).
- **Exits**: a bracket leg's fill is `ExitFilled` with the bracket's ID and the leg's reason; a
  market sell (exit signal, flatten, time exit; always the whole position) is `ExitFilled`
  under its own ID with no reason. Proceeds are the USDT received minus USDT commission. A
  partially filled leg closes the position only once it is filled, or once its list is done
  (what did not sell stays behind as dust); when the engine cancels a partly filled bracket,
  what sold is carried into the sell that follows, which reports the whole exit.
- **Cancels**: a bracket cancels its order list (or single order); `not_open` when it already
  filled or was cancelled, with its outcome in the result's events.

Fills
-----
The user data stream's `executionReport`s are matched by client order ID (`c`, or `C` on
cancels) and `listStatus` closes a list. An order's progress only moves forward: the executed
quantity and quote are cumulative (`z`, `Z`), commissions are kept per trade ID, so a report
heard twice (stream and poll, or a FULL response and the stream) changes nothing. While the
stream is down (every `poll_interval_down`), after every reconnect, and at least every
`poll_interval`, `events` polls every working order over REST (and fetches the trades of one
whose commissions are incomplete), so a missed event delays a fill but never loses it. After
a reconnect it also reports a `Reconciliation`.

Failures are results, never exceptions: an order request without an answer is `unknown`
(queried later by client ID), a filter or balance rejection is `rejected` with a reason
(`filter_<name>`, `insufficient_balance`, ...), a duplicate ID is queried, and an
authentication failure or a ban stops all order placement (`BrokerHalted`, then every order
`rejected` with `auth` / `banned`). Read-only failures inside `events` are logged and retried
at the next call.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Literal

from crypto_moonshot.broker.base import (
    QUOTE_ASSET,
    ExitPlan,
    ExitReason,
    Fill,
    Order,
    OrderType,
    Side,
    base_asset,
    child_order_id,
)
from crypto_moonshot.broker.candles import Candles
from crypto_moonshot.broker.execution import (
    BalanceMismatch,
    BracketFailed,
    BracketPlaced,
    BrokerEvent,
    BrokerHalted,
    CancelReason,
    CancelResult,
    CancelStatus,
    EntryFilled,
    ExitFilled,
    OrderCancelled,
    OrderRejection,
    OrderState,
    OrphanOrder,
    Position,
    QueryResult,
    Reconciliation,
    SubmitResult,
    WorkingOrder,
)
from crypto_moonshot.exchange.client import SignedClient
from crypto_moonshot.exchange.errors import (
    AuthError,
    Banned,
    DuplicateOrder,
    ExchangeError,
    FilterRejected,
    InsufficientBalance,
    OrderNotAccepted,
    OrderStatusUnknown,
    RateLimited,
    UnknownOrder,
    WouldTake,
    WouldTrigger,
)
from crypto_moonshot.exchange.models import Account, ExchangeOrder, OcoLeg, OrderList
from crypto_moonshot.exchange.rules import ExchangeRules, SymbolRules
from crypto_moonshot.exchange.ws_api import (
    AccountPosition,
    ExecutionReport,
    ListStatus,
    StreamDown,
    StreamMessage,
    StreamUp,
    UserDataStream,
)
from crypto_moonshot.models import Interval

logger = logging.getLogger(__name__)

ZERO = Decimal(0)
QUOTE_STEP = Decimal("0.00000001")
"""`quoteOrderQty` precision (*unverified*: USDT's quote precision on Binance is 8)."""
ID_PREFIXES = ("cmb-", "cms-", "cmx-")
"""Client order IDs this toolkit creates (`client_order_id`, `child_order_id`)."""
BRACKET_ROLE = "bracket"
TP_ROLE = "take_profit"
STOP_ROLE = "stop"
MAX_ATTEMPT_SCAN = 20
"""Bracket attempts a reconciliation looks for when adopting a bracket the state lost."""
_FINAL = frozenset({"FILLED", "CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH"})
_RANK = {"NEW": 0, "PENDING_NEW": 0, "PARTIALLY_FILLED": 1, "PENDING_CANCEL": 1}

StreamFactory = Callable[[Callable[[StreamMessage], None]], UserDataStream]
Clock = Callable[[], datetime]
BracketKind = Literal["oco", "single", "market"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _d(value: float) -> Decimal:
    """A float from the engine as the Decimal it prints as (never its binary expansion)."""
    return Decimal(repr(value))


class ExitPlanError(ValueError):
    """An exit plan the exchange cannot hold (see `exit_plan_problems`)."""


def exit_plan_problems(plan: ExitPlan | None) -> list[str]:
    """Why `plan` cannot be mirrored on Binance (empty when it can)."""
    if plan is None:
        return []
    problems: list[str] = []
    if plan.take_profit is not None and plan.tp_fill == "close":
        problems.append(
            "tp_fill = 'close' cannot be mirrored on an exchange (a resting take-profit limit "
            "fills on touch); use tp_fill = 'touch'"
        )
    if plan.stop is not None and plan.trailing is not None and plan.stop < plan.trailing:
        problems.append(
            f"stop {plan.stop:g} is tighter than trailing {plan.trailing:g}: Binance holds only "
            "one stop per position, so a plan with both needs stop >= trailing (the trailing "
            "stop alone is then equivalent)"
        )
    return problems


def check_exit_plan(plan: ExitPlan | None, *, name: str = "exit plan") -> None:
    """Raise `ExitPlanError` for a plan the exchange cannot hold."""
    problems = exit_plan_problems(plan)
    if problems:
        raise ExitPlanError(f"{name}: " + "; ".join(problems))


def needs_bracket(plan: ExitPlan | None) -> bool:
    return plan is not None and (
        plan.take_profit is not None or plan.stop is not None or plan.trailing is not None
    )


def bracket_id(entry_id: str, attempt: int = 0) -> str:
    """The bracket ID of an entry's `attempt`-th bracket (the first: `child_order_id(entry,
    "bracket")`, as the paper broker's)."""
    return child_order_id(entry_id, BRACKET_ROLE if attempt == 0 else f"{BRACKET_ROLE}:{attempt}")


def leg_ids(bracket: str) -> tuple[str, str]:
    """The client IDs of an OCO bracket's take-profit and stop legs."""
    return child_order_id(bracket, TP_ROLE), child_order_id(bracket, STOP_ROLE)


def _stop_reason(plan: ExitPlan | None) -> ExitReason:
    return ExitReason.TRAILING_STOP if plan is not None and plan.trailing else ExitReason.STOP


def _filter_reason(exc: FilterRejected) -> str:
    return f"filter_{exc.filter_name.lower()}" if exc.filter_name else "filter"


# -- bookkeeping --------------------------------------------------------------------------------


@dataclass(slots=True)
class _Trade:
    qty: Decimal
    quote: Decimal
    commission: Decimal
    asset: str


@dataclass(slots=True)
class _Progress:
    """What the exchange did with one order (by client ID). Only moves forward."""

    client_id: str
    symbol: str
    exchange_id: int | None = None
    status: str = "NEW"
    order_type: str = ""
    executed: Decimal = ZERO
    quote: Decimal = ZERO
    trades: dict[int, _Trade] = field(default_factory=dict)
    time: datetime | None = None
    seen: bool = False
    """The exchange confirmed the order exists."""

    @property
    def final(self) -> bool:
        return self.status in _FINAL

    @property
    def is_open(self) -> bool:
        return self.seen and not self.final

    @property
    def trades_complete(self) -> bool:
        return sum((t.qty for t in self.trades.values()), ZERO) >= self.executed

    def commission(self, asset: str) -> Decimal:
        return sum((t.commission for t in self.trades.values() if t.asset == asset), ZERO)

    def other_commissions(self, *known: str) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for t in self.trades.values():
            if t.asset not in known and t.commission:
                out[t.asset] = out.get(t.asset, ZERO) + t.commission
        return out

    def update(
        self,
        status: str,
        executed: Decimal,
        quote: Decimal,
        time: datetime | None,
        exchange_id: int | None = None,
        order_type: str = "",
    ) -> None:
        self.seen = True
        if exchange_id is not None:
            self.exchange_id = exchange_id
        if order_type:
            self.order_type = order_type
        if executed > self.executed:
            self.executed = executed
        if quote > self.quote:
            self.quote = quote
        if not self.final and (
            status in _FINAL or (status in _RANK and _RANK[status] >= _RANK.get(self.status, 0))
        ):
            self.status = status
        if time is not None and (self.time is None or time > self.time):
            self.time = time

    def apply(self, order: ExchangeOrder) -> None:
        """A REST view of the order (placement, query or cancel response)."""
        self.update(
            order.status,
            order.executed_quantity,
            order.cumulative_quote_quantity,
            order.update_time or order.time,
            order.order_id,
            order.order_type,
        )
        for fill in order.fills:
            if fill.trade_id is not None:
                self.trades.setdefault(
                    fill.trade_id,
                    _Trade(
                        fill.quantity,
                        fill.quantity * fill.price,
                        fill.commission,
                        fill.commission_asset,
                    ),
                )


@dataclass(slots=True)
class _Entry:
    order: Order
    interval: Interval
    progress: _Progress
    done: bool = False
    cancel_requested: bool = False
    recovered: bool = False
    """Learned from a query or the state file: its bracket may already exist."""


@dataclass(slots=True)
class _Sell:
    order: Order
    progress: _Progress
    done: bool = False
    cancel_requested: bool = False


@dataclass(slots=True)
class _Bracket:
    id: str
    symbol: str
    entry_id: str
    kind: BracketKind
    legs: dict[str, ExitReason]
    """Client ID of each exchange order -> the exit reason its fill means."""
    progress: dict[str, _Progress]
    list_done: bool = False
    cancel_requested: bool = False
    done: bool = False
    verified: bool = False
    """The kind is known from the exchange (not guessed from the plan after a restart)."""


@dataclass(slots=True)
class _Holding:
    """An open position the broker manages."""

    position: Position
    ref: Decimal
    quantity: Decimal
    """Net base quantity the position holds (what its bracket or sell may sell)."""
    opened: datetime
    """When the entry filled (events about the position are never dated before)."""
    dust: Decimal = ZERO
    sold_qty: Decimal = ZERO
    sold_quote: Decimal = ZERO
    sold_fee: Decimal = ZERO
    """What a cancelled, partly filled bracket sold (carried into the closing sell)."""
    protect: bool = True
    """Keep a bracket working (off once the engine cancelled it to sell)."""
    failures: int = 0
    retry_at: datetime | None = None
    attempt: int = 0

    @property
    def remaining(self) -> Decimal:
        return self.quantity - self.sold_qty

    @property
    def unprotected(self) -> bool:
        return (
            self.protect
            and self.position.bracket_id is None
            and needs_bracket(self.position.exit_plan)
        )


# -- the broker ---------------------------------------------------------------------------------


class BinanceBroker:
    """See the module docstring.

    Args:
        client: the signed REST client (testnet URL and keys).
        rules: symbol rules (default: built on `client`).
        stream: builds the user data stream given the broker's sink (`None`: REST polling
            only).
        clock: aware-UTC "now" for polling and retry timing and for event times the exchange
            does not give.
        poll_interval: poll working orders at least this often while the stream is up...
        poll_interval_down: ...and this often while it is down (or absent).
        retry_base, retry_max: backoff of bracket retries (doubling per failure).
        default_interval: the interval of an order the broker learns of without one (a query
            for an order `load` did not give it).
        plans: exit plans to check now (`ExitPlanError` for one the exchange cannot hold).
    """

    needs_candles = False
    manages_time_exits = False
    sell_cancels_bracket = False

    def __init__(
        self,
        client: SignedClient,
        *,
        rules: ExchangeRules | None = None,
        stream: StreamFactory | None = None,
        clock: Clock = _utc_now,
        poll_interval: timedelta = timedelta(minutes=1),
        poll_interval_down: timedelta = timedelta(seconds=10),
        retry_base: timedelta = timedelta(seconds=5),
        retry_max: timedelta = timedelta(minutes=5),
        default_interval: Interval = Interval.H1,
        plans: Iterable[ExitPlan | None] = (),
    ) -> None:
        for plan in plans:
            check_exit_plan(plan)
        self._client = client
        self._rules = rules if rules is not None else ExchangeRules(client, now=clock)
        self._stream_factory = stream
        self._stream: UserDataStream | None = None
        self._clock = clock
        self._poll_interval = poll_interval
        self._poll_interval_down = poll_interval_down
        self._retry_base = retry_base
        self._retry_max = retry_max
        self._default_interval = default_interval
        self._entries: dict[str, _Entry] = {}
        self._sells: dict[str, _Sell] = {}
        self._brackets: dict[str, _Bracket] = {}
        self._legs: dict[str, str] = {}
        """Leg client ID -> bracket ID (single and market brackets map to themselves)."""
        self._holdings: dict[str, _Holding] = {}
        self._bracket_entry: dict[str, str] = {}
        """Bracket ID -> its entry's ID (bracket events are queued under the entry's ID)."""
        self._queue: list[tuple[str | None, BrokerEvent]] = []
        """Events not reported yet, tagged with the order they belong to."""
        self._inbox: list[StreamMessage] = []
        self._lock = asyncio.Lock()
        self._started = False
        self._last_poll: datetime | None = None
        self._force_poll = False
        self._reconcile_due = False
        self._halted: str | None = None
        self.balances: dict[str, tuple[Decimal, Decimal]] = {}
        """Latest balances the stream reported (`asset -> (free, locked)`)."""
        self.other_commissions: list[tuple[str, str, Decimal]] = []
        """(order ID, asset, amount) of commissions paid in neither the base asset nor USDT."""
        self.dust: dict[str, Decimal] = {}
        """Base quantity left on the account per symbol (commission rounding, partial exits)."""

    def __repr__(self) -> str:
        return f"BinanceBroker(client={self._client!r}, positions={len(self._holdings)})"

    __str__ = __repr__

    # -- lifecycle ----------------------------------------------------------------------------

    @property
    def stream_connected(self) -> bool:
        return self._stream is not None and self._stream.connected

    @property
    def halted(self) -> str | None:
        """Why the broker stopped placing orders, or None."""
        return self._halted

    @property
    def positions(self) -> dict[str, Position]:
        return {s: h.position for s, h in self._holdings.items()}

    async def start(self) -> None:
        """Load the symbol rules and start the user data stream (idempotent; the first async
        call does it too)."""
        if self._started:
            return
        self._started = True
        if self._stream_factory is not None:
            self._stream = self._stream_factory(self.on_stream)
            self._stream.start()
        await self._rules.refresh()

    async def aclose(self) -> None:
        if self._stream is not None:
            await self._stream.aclose()

    def check_plans(self, plans: Iterable[ExitPlan | None]) -> None:
        """Raise `ExitPlanError` for the first plan the exchange cannot hold."""
        for plan in plans:
            check_exit_plan(plan)

    def on_stream(self, message: StreamMessage) -> None:
        """The user data stream's sink: kept until the next async call processes it."""
        if isinstance(message, StreamUp):
            if message.reconnect:
                self._force_poll = True
                self._reconcile_due = True
            return
        if isinstance(message, StreamDown):
            logger.warning("user data stream down (%s): polling over REST", message.reason)
            return
        self._inbox.append(message)

    def load(self, orders: Sequence[WorkingOrder], positions: Sequence[Position]) -> None:
        """Adopt the state file's working orders and open positions (`RestoresState`)."""
        for position in positions:
            if position.symbol in self._holdings:
                continue
            holding = _Holding(
                position, _d(position.entry_ref), _d(position.quantity), position.entry_time
            )
            holding.protect = position.bracket_id is not None or needs_bracket(position.exit_plan)
            self._holdings[position.symbol] = holding
            if position.bracket_id is not None:
                self._track_bracket(position, position.bracket_id, verified=False)
        for working in orders:
            order = working.order
            if order.id in self._entries or order.id in self._sells or order.id in self._legs:
                continue
            if order.side is Side.BUY:
                entry = _Entry(
                    order, working.interval, _Progress(order.id, order.symbol), recovered=True
                )
                self._entries[order.id] = entry
            elif order.type is OrderType.MARKET:
                self._sells[order.id] = _Sell(order, _Progress(order.id, order.symbol))

    # -- ExecutionBroker ----------------------------------------------------------------------

    def candle_needs(self, interval: Interval) -> dict[str, datetime]:
        return {}

    def add_candles(self, symbol: str, candles: Candles) -> None:
        pass

    async def submit(self, order: Order, *, interval: Interval) -> SubmitResult:
        async with self._lock:
            await self.start()
            self._process_inbox()
            if self._halted is not None:
                return SubmitResult.rejected(self._halted)
            if order.side is Side.BUY:
                return await self._submit_entry(order, interval)
            if order.type is not OrderType.MARKET:
                return SubmitResult.rejected("unsupported_order")
            return await self._submit_sell(order)

    async def cancel(self, order: Order) -> CancelResult:
        async with self._lock:
            await self.start()
            self._process_inbox()
            if order.type is OrderType.BRACKET or order.id in self._brackets:
                return await self._cancel_bracket(order)
            return await self._cancel_order(order)

    async def query(self, order: Order) -> QueryResult:
        async with self._lock:
            await self.start()
            self._process_inbox()
            try:
                if order.type is OrderType.BRACKET or order.id in self._brackets:
                    state = await self._query_bracket(order)
                else:
                    state = await self._query_order(order)
                await self._settle()
            except UnknownOrder:
                self._forget(order.id)
                return QueryResult(OrderState.NOT_FOUND)
            except (AuthError, Banned) as exc:
                self._halt(exc)
                return QueryResult(OrderState.UNKNOWN)
            except ExchangeError as exc:
                logger.warning("query of %s failed: %s", order.id, exc)
                return QueryResult(OrderState.UNKNOWN)
            reason = None
            if state is OrderState.REJECTED:
                reason = "rejected"
            events = self._take(order.id) + self._take(self._entry_of_bracket(order.id))
            return QueryResult(state, reason, tuple(events))

    async def events(
        self, until: datetime, *, interval: Interval | None = None
    ) -> list[BrokerEvent]:
        async with self._lock:
            await self.start()
            self._process_inbox()
            now = self._clock()
            report: Reconciliation | None = None
            try:
                if self._reconcile_due:
                    self._reconcile_due = False
                    report = await self._reconciliation(now)
                elif self._poll_due(now):
                    await self._poll()
                await self._retry_brackets()
                await self._settle()
            except (AuthError, Banned) as exc:
                self._halt(exc)
            except ExchangeError as exc:
                logger.warning("exchange broker: catching up failed (%s); retrying later", exc)
            if report is not None:
                self._queue.append((None, report))
            out = [event for _, event in self._queue]
            self._queue.clear()
            out.sort(key=lambda e: e.time)
            return out

    async def reconcile(
        self,
        orders: Sequence[WorkingOrder],
        positions: Sequence[Position],
        *,
        now: datetime,
    ) -> Reconciliation:
        async with self._lock:
            self.load(orders, positions)
            await self.start()
            self._process_inbox()
            return await self._reconciliation(now)

    # -- entries ------------------------------------------------------------------------------

    def _forget(self, order_id: str) -> None:
        """An order the exchange never received: stop following it (the engine may submit it
        again under the same ID)."""
        self._entries.pop(order_id, None)
        self._sells.pop(order_id, None)

    def _selling(self, symbol: str) -> bool:
        return any(s.order.symbol == symbol and not s.done for s in self._sells.values())

    async def _rules_for(self, symbol: str) -> SymbolRules | None:
        try:
            return await self._rules.get(symbol)
        except ExchangeError:
            return None

    async def _submit_entry(self, order: Order, interval: Interval) -> SubmitResult:
        if exit_plan_problems(order.exit_plan):
            return SubmitResult.rejected("unsupported_exit_plan")
        if order.id in self._entries:
            return SubmitResult.rejected("duplicate_id")
        rules = await self._rules_for(order.symbol)
        if rules is None:
            return SubmitResult.rejected("unknown_symbol")
        if not rules.is_trading:
            return SubmitResult.rejected("symbol_not_trading")
        assert order.quote_amount is not None
        quote = _d(order.quote_amount).quantize(QUOTE_STEP, ROUND_DOWN)
        if rules.min_notional_applies_to_market and not rules.meets_min_notional(quote):
            return SubmitResult.rejected("min_notional")
        entry = _Entry(order, interval, _Progress(order.id, order.symbol))
        self._entries[order.id] = entry  # before the call: the stream may report it first
        try:
            response = await self._client.new_order(
                order.symbol, "BUY", "MARKET", client_order_id=order.id, quote_order_qty=quote
            )
        except OrderStatusUnknown:
            return SubmitResult.unknown()
        except DuplicateOrder:
            return await self._submit_from_query(order)
        except ExchangeError as exc:
            del self._entries[order.id]
            return SubmitResult.rejected(self._rejection(exc))
        entry.progress.apply(response)
        await self._settle()
        return SubmitResult.accepted(str(response.order_id), self._take(order.id))

    async def _submit_from_query(self, order: Order) -> SubmitResult:
        """A duplicate client ID: the order exists already (a retry after a lost answer)."""
        try:
            state = await self._query_order(order)
            await self._settle()
        except UnknownOrder:
            return SubmitResult.rejected("duplicate_id")
        except ExchangeError:
            return SubmitResult.unknown()
        if state is OrderState.REJECTED:
            return SubmitResult.rejected("rejected")
        return SubmitResult.accepted(None, self._take(order.id))

    def _rejection(self, exc: ExchangeError) -> str:
        if isinstance(exc, FilterRejected):
            self._rules.note_filter_rejection()
            return _filter_reason(exc)
        if isinstance(exc, InsufficientBalance):
            return "insufficient_balance"
        if isinstance(exc, RateLimited):
            return "rate_limited"
        if isinstance(exc, AuthError | Banned):
            return self._halt(exc)
        if isinstance(exc, OrderNotAccepted):
            return "rejected"
        return "exchange_error"

    def _halt(self, exc: AuthError | Banned) -> str:
        reason = "banned" if isinstance(exc, Banned) else "auth"
        if self._halted is None:
            self._halted = reason
            logger.error("exchange broker halted (%s): %s", reason, exc.message)
            self._queue.append((None, BrokerHalted(self._clock(), reason, exc.message)))
        return reason

    async def _finish_entry(self, entry: _Entry) -> None:
        progress, order = entry.progress, entry.order
        entry.done = True
        time = progress.time or self._clock()
        if progress.executed <= 0:
            if progress.status == "REJECTED":
                self._queue.append(
                    (order.id, OrderRejection(order.id, order.symbol, Side.BUY, time, "rejected"))
                )
            elif not entry.cancel_requested:
                reason = (
                    CancelReason.EXPIRED if progress.status == "EXPIRED" else CancelReason.CANCELLED
                )
                self._queue.append((order.id, OrderCancelled(order, time, reason)))
            return
        rules = await self._rules_for(order.symbol)
        base = base_asset(order.symbol)
        avg = progress.quote / progress.executed
        base_fee = progress.commission(base)
        usdt_fee = progress.commission(QUOTE_ASSET)
        for asset, amount in progress.other_commissions(base, QUOTE_ASSET).items():
            self.other_commissions.append((order.id, asset, amount))
            logger.info("%s: commission %s %s (not converted)", order.id, amount, asset)
        net = progress.executed - base_fee
        quantity = rules.round_quantity(net) if rules is not None else net
        dust = net - quantity
        if dust:
            self.dust[order.symbol] = self.dust.get(order.symbol, ZERO) + dust
        fill = Fill(
            order.id,
            order.symbol,
            Side.BUY,
            time,
            float(avg),
            float(progress.executed),
            float(usdt_fee + base_fee * avg),
        )
        position = Position(
            symbol=order.symbol,
            interval=entry.interval,
            quantity=float(quantity),
            entry_time=order.time,
            entry_ref=float(avg),
            entry_price=float(avg),
            cost=float(progress.quote),
            entry_order_id=order.id,
            exit_plan=order.exit_plan,
        )
        holding = _Holding(position, avg, quantity, time, dust)
        self._holdings[order.symbol] = holding
        failure: str | None = None
        if needs_bracket(order.exit_plan):
            if entry.recovered and await self._adopt_bracket(holding):
                failure = None
            else:
                failure = await self._protect(holding)
        self._queue.append((order.id, EntryFilled(fill, holding.position)))
        if failure is not None:
            self._schedule_retry(holding, failure)
            when = max(self._clock(), time)
            self._queue.append((order.id, BracketFailed(order.symbol, order.id, when, failure)))

    # -- brackets -----------------------------------------------------------------------------

    def _track_bracket(
        self,
        position: Position,
        bid: str,
        *,
        kind: BracketKind | None = None,
        reason: ExitReason | None = None,
        verified: bool = True,
    ) -> _Bracket:
        plan = position.exit_plan
        if kind is None:
            both = (
                plan is not None
                and plan.take_profit is not None
                and (plan.stop is not None or plan.trailing is not None)
            )
            kind = "oco" if both else "single"
        if kind == "oco":
            tp, sl = leg_ids(bid)
            legs = {tp: ExitReason.TAKE_PROFIT, sl: _stop_reason(plan)}
        else:
            if reason is None:
                tp_only = plan is not None and plan.take_profit is not None
                reason = ExitReason.TAKE_PROFIT if tp_only else _stop_reason(plan)
            legs = {bid: reason}
        bracket = _Bracket(
            bid,
            position.symbol,
            position.entry_order_id,
            kind,
            legs,
            {leg: _Progress(leg, position.symbol) for leg in legs},
            verified=verified,
        )
        for leg in legs:
            self._legs[leg] = bid
        self._brackets[bid] = bracket
        if position.entry_order_id:
            self._bracket_entry[bid] = position.entry_order_id
        return bracket

    def _untrack_bracket(self, bracket: _Bracket) -> None:
        for leg in bracket.legs:
            self._legs.pop(leg, None)
        self._brackets.pop(bracket.id, None)

    def _entry_of_bracket(self, bid: str) -> str:
        return self._bracket_entry.get(bid, "")

    async def _protect(self, holding: _Holding) -> str | None:
        """Place the holding's bracket; None once it is placed (or the levels were crossed
        and it sold), else why it failed."""
        position = holding.position
        plan = position.exit_plan
        assert plan is not None
        symbol = position.symbol
        rules = await self._rules_for(symbol)
        if rules is None:
            return "unknown_symbol"
        bid = bracket_id(position.entry_order_id, holding.attempt)
        ref = holding.ref
        tp = (
            rules.round_price(ref * (1 + _d(plan.take_profit)), "up")
            if plan.take_profit is not None
            else None
        )
        trailing = (
            int((_d(plan.trailing) * 10_000).to_integral_value(ROUND_HALF_UP))
            if plan.trailing is not None
            else None
        )
        stop = (
            rules.round_price(ref * (1 - _d(plan.stop)), "down")
            if plan.stop is not None and trailing is None
            else None
        )
        qty = holding.remaining
        has_stop = stop is not None or trailing is not None
        kind: BracketKind = "oco" if tp is not None and has_stop else "single"
        placed: OrderList | ExchangeOrder
        try:
            if kind == "oco":
                assert tp is not None
                tp_id, sl_id = leg_ids(bid)
                placed = await self._client.new_oco(
                    symbol,
                    "SELL",
                    qty,
                    OcoLeg("LIMIT_MAKER", tp_id, price=tp),
                    OcoLeg("STOP_LOSS", sl_id, stop_price=stop, trailing_delta=trailing),
                    list_client_order_id=bid,
                )
            elif tp is not None:
                placed = await self._client.new_order(
                    symbol, "SELL", "LIMIT_MAKER", client_order_id=bid, quantity=qty, price=tp
                )
            else:
                placed = await self._client.new_order(
                    symbol,
                    "SELL",
                    "STOP_LOSS",
                    client_order_id=bid,
                    quantity=qty,
                    stop_price=stop,
                    trailing_delta=trailing,
                )
        except WouldTake:
            return await self._sell_crossed(holding, bid, ExitReason.TAKE_PROFIT)
        except WouldTrigger:
            return await self._sell_crossed(holding, bid, _stop_reason(plan))
        except DuplicateOrder:
            if await self._adopt_bracket(holding, bid):
                return None
            holding.attempt += 1
            return "duplicate_id"
        except OrderStatusUnknown:
            holding.attempt += 1  # it may still appear: the next attempt uses a fresh ID
            if await self._adopt_bracket(holding, bid):
                return None
            return "unknown"
        except (FilterRejected, OrderNotAccepted) as exc:
            crossed = await self._crossed_level(symbol, tp, stop, plan, exc)
            if crossed is not None:
                return await self._sell_crossed(holding, bid, crossed)
            return self._rejection(exc)
        except ExchangeError as exc:
            return self._rejection(exc)
        bracket = self._track_bracket(position, bid, kind=kind)
        if isinstance(placed, OrderList):
            self._apply_list(bracket, placed)
            for report in placed.reports:
                self._apply_leg(bracket, report)
        else:
            bracket.progress[bid].apply(placed)
        holding.position = replace(position, bracket_id=bid)
        holding.failures, holding.retry_at = 0, None
        return None

    async def _crossed_level(
        self,
        symbol: str,
        tp: Decimal | None,
        stop: Decimal | None,
        plan: ExitPlan,
        exc: ExchangeError,
    ) -> ExitReason | None:
        """For a rejection that may mean a level was already crossed (the OCO price
        relationship, a generic -2010), the level the last price is beyond, if any."""
        if isinstance(exc, FilterRejected) and exc.filter_name is not None:
            return None
        if isinstance(exc, InsufficientBalance):
            return None
        try:
            price = await self._client.ticker_price(symbol)
        except ExchangeError:
            return None
        if tp is not None and price >= tp:
            return ExitReason.TAKE_PROFIT
        if stop is not None and price <= stop:
            return _stop_reason(plan)
        return None

    async def _sell_crossed(self, holding: _Holding, bid: str, reason: ExitReason) -> str | None:
        """A level is already crossed: sell at market at once, under the bracket's ID."""
        position = holding.position
        symbol = position.symbol
        bracket = self._track_bracket(position, bid, kind="market", reason=reason)
        holding.position = replace(position, bracket_id=bid)
        try:
            response = await self._client.new_order(
                symbol, "SELL", "MARKET", client_order_id=bid, quantity=holding.remaining
            )
        except OrderStatusUnknown:
            return None  # the poll finds out what happened to it
        except ExchangeError as exc:
            self._untrack_bracket(bracket)
            holding.position = position
            holding.attempt += 1
            return self._rejection(exc)
        bracket.progress[bid].apply(response)
        logger.info("%s: %s level crossed at placement; sold at market", symbol, reason.value)
        return None

    async def _adopt_bracket(self, holding: _Holding, bid: str | None = None) -> bool:
        """Find a bracket of the holding's entry already on the exchange (`bid`, or the first
        attempt's ID) and track it."""
        position = holding.position
        wanted = bid or bracket_id(position.entry_order_id, 0)
        try:
            listed = await self._client.get_order_list(wanted)
        except UnknownOrder:
            listed = None
        if listed is not None:
            bracket = self._track_bracket(position, wanted, kind="oco")
            self._apply_list(bracket, listed)
            await self._refresh_legs(bracket)
        else:
            try:
                single = await self._client.get_order(position.symbol, wanted)
            except UnknownOrder:
                return False
            bracket = self._track_single(position, wanted, single)
        holding.position = replace(position, bracket_id=wanted)
        holding.failures, holding.retry_at = 0, None
        return True

    def _track_single(self, position: Position, bid: str, order: ExchangeOrder) -> _Bracket:
        kind: BracketKind = "market" if order.order_type == "MARKET" else "single"
        reason: ExitReason
        if order.order_type == "LIMIT_MAKER":
            reason = ExitReason.TAKE_PROFIT
        elif order.order_type == "STOP_LOSS":
            reason = ExitReason.TRAILING_STOP if order.trailing_delta else ExitReason.STOP
        else:  # a market sell after a crossed level: which one, from where it filled
            avg = order.average_price
            reason = (
                ExitReason.TAKE_PROFIT
                if avg is not None and avg >= _d(position.entry_ref)
                else _stop_reason(position.exit_plan)
            )
        bracket = self._track_bracket(position, bid, kind=kind, reason=reason)
        bracket.progress[bid].apply(order)
        return bracket

    async def _retry_brackets(self) -> None:
        now = self._clock()
        for holding in list(self._holdings.values()):
            if not holding.unprotected or self._selling(holding.position.symbol):
                continue
            if holding.retry_at is not None and now < holding.retry_at:
                continue
            failure = await self._protect(holding)
            if failure is None:
                position = holding.position
                logger.info("%s: bracket %s placed", position.symbol, position.bracket_id)
                when = max(self._clock(), holding.opened)
                self._queue.append((position.entry_order_id, BracketPlaced(position, when)))
            else:
                self._schedule_retry(holding, failure)

    def _schedule_retry(self, holding: _Holding, failure: str) -> None:
        holding.failures += 1
        delay = min(self._retry_max, self._retry_base * 2 ** (holding.failures - 1))
        holding.retry_at = self._clock() + delay
        logger.warning(
            "%s: bracket not placed (%s); retrying in %s",
            holding.position.symbol,
            failure,
            delay,
        )

    async def _cancel_bracket(self, order: Order) -> CancelResult:
        bracket = self._brackets.get(order.id)
        holding = self._holdings.get(order.symbol)
        if bracket is None:
            if holding is None or holding.position.bracket_id != order.id:
                # Final already (its outcome is queued, or was reported).
                events = self._take(order.id) + self._take(self._entry_of_bracket(order.id))
                return CancelResult(CancelStatus.NOT_OPEN, tuple(events))
            bracket = self._track_bracket(holding.position, order.id, verified=False)
        if bracket.done:
            return CancelResult(CancelStatus.NOT_OPEN, tuple(self._take_bracket(bracket)))
        bracket.cancel_requested = True
        try:
            if bracket.kind == "market":
                raise UnknownOrder("a market order cannot be cancelled")
            if bracket.kind == "oco":
                try:
                    listed = await self._client.cancel_order_list(bracket.symbol, bracket.id)
                except UnknownOrder:
                    if bracket.verified:
                        raise
                    response = await self._client.cancel_order(bracket.symbol, bracket.id)
                    bracket = self._retype_single(bracket, holding, response)
                else:
                    self._apply_list(bracket, listed)
                    for report in listed.reports:
                        self._apply_leg(bracket, report)
            else:
                response = await self._client.cancel_order(bracket.symbol, bracket.id)
                bracket.progress[bracket.id].apply(response)
        except UnknownOrder:
            # Not open: it filled, was cancelled, or (after a restart) never existed.
            try:
                await self._refresh_bracket(bracket)
            except UnknownOrder:
                self._untrack_bracket(bracket)
                if holding is not None:
                    holding.position = replace(holding.position, bracket_id=None)
                return CancelResult(CancelStatus.NOT_OPEN)
            except ExchangeError:
                bracket.cancel_requested = False
                return CancelResult(CancelStatus.UNKNOWN)
            await self._settle()
            return CancelResult(CancelStatus.NOT_OPEN, tuple(self._take_bracket(bracket)))
        except (AuthError, Banned) as exc:
            self._halt(exc)
            bracket.cancel_requested = False
            return CancelResult(CancelStatus.UNKNOWN)
        except ExchangeError as exc:
            logger.warning("cancel of bracket %s failed: %s", bracket.id, exc)
            bracket.cancel_requested = False
            return CancelResult(CancelStatus.UNKNOWN)
        await self._fetch_trades(bracket)
        await self._settle()
        events = self._take_bracket(bracket)
        if events:  # a leg filled completely before the cancel
            return CancelResult(CancelStatus.NOT_OPEN, tuple(events))
        return CancelResult(CancelStatus.CANCELLED)

    def _retype_single(
        self, bracket: _Bracket, holding: _Holding | None, response: ExchangeOrder
    ) -> _Bracket:
        self._untrack_bracket(bracket)
        position = holding.position if holding is not None else None
        assert position is not None
        fresh = self._track_single(position, bracket.id, response)
        fresh.cancel_requested = True
        return fresh

    def _take_bracket(self, bracket: _Bracket) -> list[BrokerEvent]:
        return self._take(bracket.id) + self._take(bracket.entry_id)

    def _apply_list(self, bracket: _Bracket, listed: OrderList) -> None:
        bracket.verified = True
        if listed.list_order_status == "ALL_DONE":
            bracket.list_done = True
        for leg in listed.orders:
            progress = bracket.progress.get(leg.client_order_id)
            if progress is not None:
                progress.exchange_id = leg.order_id
                progress.seen = True

    def _apply_leg(self, bracket: _Bracket, order: ExchangeOrder) -> None:
        progress = bracket.progress.get(order.client_order_id)
        if progress is not None:
            progress.apply(order)

    async def _refresh_legs(self, bracket: _Bracket) -> None:
        for leg in bracket.legs:
            order = await self._client.get_order(bracket.symbol, leg)
            self._apply_leg(bracket, order)

    async def _refresh_bracket(self, bracket: _Bracket) -> None:
        """Bring a bracket up to date over REST (raises `UnknownOrder` if it does not exist)."""
        if bracket.kind == "oco":
            try:
                listed = await self._client.get_order_list(bracket.id)
            except UnknownOrder:
                if bracket.verified:
                    raise
                single = await self._client.get_order(bracket.symbol, bracket.id)
                holding = self._holdings.get(bracket.symbol)
                cancel = bracket.cancel_requested
                fresh = self._retype_single(bracket, holding, single)
                fresh.cancel_requested = cancel
                await self._fetch_trades(fresh)
                return
            self._apply_list(bracket, listed)
            await self._refresh_legs(bracket)
        else:
            order = await self._client.get_order(bracket.symbol, bracket.id)
            if not bracket.verified:
                holding = self._holdings.get(bracket.symbol)
                cancel = bracket.cancel_requested
                bracket = self._retype_single(bracket, holding, order)
                bracket.cancel_requested = cancel
            else:
                bracket.progress[bracket.id].apply(order)
        await self._fetch_trades(bracket)

    async def _query_bracket(self, order: Order) -> OrderState:
        bracket = self._brackets.get(order.id)
        if bracket is None:
            holding = self._holdings.get(order.symbol)
            position = (
                holding.position
                if holding is not None
                else Position(
                    order.symbol,
                    self._default_interval,
                    order.quantity or 0.0,
                    order.time,
                    0.0,
                    0.0,
                    0.0,
                    "",
                    order.exit_plan,
                    order.id,
                )
            )
            bracket = self._track_bracket(position, order.id, verified=False)
        await self._refresh_bracket(bracket)
        bracket = self._brackets[order.id]
        progresses = list(bracket.progress.values())
        if any(p.status == "FILLED" for p in progresses):
            return OrderState.FILLED
        if bracket.kind == "oco" and not bracket.list_done:
            return OrderState.OPEN
        if any(p.is_open for p in progresses):
            return OrderState.OPEN
        if any(p.executed > 0 for p in progresses):
            return OrderState.FILLED
        return OrderState.CANCELLED

    # -- sells --------------------------------------------------------------------------------

    async def _submit_sell(self, order: Order) -> SubmitResult:
        holding = self._holdings.get(order.symbol)
        if holding is None:
            return SubmitResult.rejected("not_holding")
        bid = holding.position.bracket_id
        if bid is not None and bid in self._brackets and not self._brackets[bid].done:
            return SubmitResult.rejected("bracket_open")
        if order.id in self._sells:
            return SubmitResult.rejected("duplicate_id")
        rules = await self._rules_for(order.symbol)
        qty = holding.remaining
        if order.quantity is not None:
            qty = min(qty, _d(order.quantity))
        if rules is not None:
            qty = rules.round_quantity(qty, market=True)
        if qty <= 0:
            return SubmitResult.rejected("quantity")
        holding.protect = False  # a sell is under way: no bracket retries
        sell = _Sell(order, _Progress(order.id, order.symbol))
        self._sells[order.id] = sell
        try:
            response = await self._client.new_order(
                order.symbol, "SELL", "MARKET", client_order_id=order.id, quantity=qty
            )
        except OrderStatusUnknown:
            return SubmitResult.unknown()
        except DuplicateOrder:
            return await self._submit_from_query(order)
        except ExchangeError as exc:
            del self._sells[order.id]
            return SubmitResult.rejected(self._rejection(exc))
        sell.progress.apply(response)
        await self._settle()
        return SubmitResult.accepted(str(response.order_id), self._take(order.id))

    def _close(
        self,
        holding: _Holding,
        order_id: str,
        progresses: Sequence[_Progress],
        reason: ExitReason | None,
    ) -> None:
        """The position is sold: one `ExitFilled` with everything it sold."""
        position = holding.position
        symbol = position.symbol
        qty = holding.sold_qty + sum((p.executed for p in progresses), ZERO)
        quote = holding.sold_quote + sum((p.quote for p in progresses), ZERO)
        base = base_asset(symbol)
        fee = holding.sold_fee
        times = [p.time for p in progresses if p.time is not None]
        for p in progresses:
            avg = p.quote / p.executed if p.executed > 0 else ZERO
            fee += p.commission(QUOTE_ASSET) + p.commission(base) * avg
            for asset, amount in p.other_commissions(base, QUOTE_ASSET).items():
                self.other_commissions.append((p.client_id, asset, amount))
                logger.info("%s: commission %s %s (not converted)", p.client_id, amount, asset)
        price = quote / qty if qty > 0 else ZERO
        fill = Fill(
            order_id,
            symbol,
            Side.SELL,
            max(times) if times else self._clock(),
            float(price),
            float(qty),
            float(fee),
            reason,
        )
        left = holding.quantity - qty
        if left > 0:
            self.dust[symbol] = self.dust.get(symbol, ZERO) + left
            logger.warning("%s: %s %s left unsold (kept as dust)", symbol, left, base)
        del self._holdings[symbol]
        tag = position.entry_order_id if order_id == position.bracket_id else order_id
        self._queue.append((tag or order_id, ExitFilled(fill, position)))

    async def _finish_sell(self, sell: _Sell) -> None:
        progress, order = sell.progress, sell.order
        sell.done = True
        time = progress.time or self._clock()
        holding = self._holdings.get(order.symbol)
        if progress.executed > 0:
            if holding is None:  # a sell the broker never saw the position of
                holding = _Holding(
                    Position(
                        order.symbol,
                        self._default_interval,
                        float(progress.executed),
                        order.time,
                        0.0,
                        0.0,
                        0.0,
                        "",
                    ),
                    ZERO,
                    progress.executed,
                    time,
                )
                self._holdings[order.symbol] = holding
            self._close(holding, order.id, [progress], None)
            return
        if progress.status == "REJECTED":
            self._queue.append(
                (order.id, OrderRejection(order.id, order.symbol, Side.SELL, time, "rejected"))
            )
        elif not sell.cancel_requested:
            reason = (
                CancelReason.EXPIRED if progress.status == "EXPIRED" else CancelReason.CANCELLED
            )
            self._queue.append((order.id, OrderCancelled(order, time, reason)))

    async def _cancel_order(self, order: Order) -> CancelResult:
        """Cancel a market order (one the exchange holds open, e.g. without liquidity)."""
        tracked = self._entries.get(order.id) or self._sells.get(order.id)
        if tracked is None:
            return CancelResult(CancelStatus.NOT_OPEN)
        if tracked.done:
            return CancelResult(CancelStatus.NOT_OPEN, tuple(self._take(order.id)))
        tracked.cancel_requested = True
        try:
            response = await self._client.cancel_order(order.symbol, order.id)
        except UnknownOrder:
            try:
                await self._query_order(order)
            except UnknownOrder:
                tracked.done = True  # never placed
                return CancelResult(CancelStatus.NOT_OPEN)
            except ExchangeError:
                tracked.cancel_requested = False
                return CancelResult(CancelStatus.UNKNOWN)
            await self._settle()
            return CancelResult(CancelStatus.NOT_OPEN, tuple(self._take(order.id)))
        except (AuthError, Banned) as exc:
            self._halt(exc)
            tracked.cancel_requested = False
            return CancelResult(CancelStatus.UNKNOWN)
        except ExchangeError:
            tracked.cancel_requested = False
            return CancelResult(CancelStatus.UNKNOWN)
        tracked.progress.apply(response)
        await self._fetch_order_trades(tracked.progress)
        await self._settle()
        events = self._take(order.id)
        if tracked.progress.executed > 0:  # it partly filled first: that is its outcome
            return CancelResult(CancelStatus.NOT_OPEN, tuple(events))
        return CancelResult(CancelStatus.CANCELLED, tuple(events))

    async def _query_order(self, order: Order) -> OrderState:
        """Bring a single order up to date by client ID (raises `UnknownOrder`)."""
        response = await self._client.get_order(order.symbol, order.id)
        tracked: _Entry | _Sell | None = self._entries.get(order.id) or self._sells.get(order.id)
        if tracked is None:
            if order.side is Side.BUY:
                tracked = _Entry(
                    order, self._default_interval, _Progress(order.id, order.symbol), recovered=True
                )
                self._entries[order.id] = tracked
            else:
                tracked = _Sell(order, _Progress(order.id, order.symbol))
                self._sells[order.id] = tracked
        tracked.progress.apply(response)
        await self._fetch_order_trades(tracked.progress)
        status = tracked.progress.status
        if status == "FILLED" or (tracked.progress.final and tracked.progress.executed > 0):
            return OrderState.FILLED
        if status in ("CANCELED", "EXPIRED", "EXPIRED_IN_MATCH"):
            return OrderState.CANCELLED
        if status == "REJECTED":
            return OrderState.REJECTED
        return OrderState.OPEN

    # -- stream and polling -------------------------------------------------------------------

    def _process_inbox(self) -> None:
        inbox, self._inbox = self._inbox, []
        for message in inbox:
            if isinstance(message, ExecutionReport):
                self._apply_report(message)
            elif isinstance(message, ListStatus):
                bracket = self._brackets.get(message.list_client_order_id)
                if bracket is not None:
                    bracket.verified = True
                    if message.list_order_status == "ALL_DONE":
                        bracket.list_done = True
            elif isinstance(message, AccountPosition):
                self.balances.update(message.balances)

    def _progress_of(self, client_id: str) -> _Progress | None:
        entry = self._entries.get(client_id)
        if entry is not None:
            return entry.progress
        sell = self._sells.get(client_id)
        if sell is not None:
            return sell.progress
        bid = self._legs.get(client_id)
        if bid is not None:
            return self._brackets[bid].progress.get(client_id)
        return None

    def _apply_report(self, report: ExecutionReport) -> None:
        progress = self._progress_of(report.order_client_id)
        if progress is None:
            return
        progress.update(
            report.status,
            report.cumulative_quantity,
            report.cumulative_quote,
            report.time,
            report.order_id,
            report.order_type,
        )
        if report.execution_type == "TRADE" and report.trade_id is not None:
            progress.trades.setdefault(
                report.trade_id,
                _Trade(
                    report.last_quantity,
                    report.last_quantity * report.last_price,
                    report.commission,
                    report.commission_asset,
                ),
            )

    def _poll_due(self, now: datetime) -> bool:
        if self._force_poll or self._last_poll is None:
            return True
        gap = self._poll_interval if self.stream_connected else self._poll_interval_down
        return now - self._last_poll >= gap

    async def _poll(self) -> None:
        """Bring every working order up to date over REST."""
        self._force_poll = False
        self._last_poll = self._clock()
        for entry in list(self._entries.values()):
            if not entry.done:
                await self._poll_order(entry.order)
        for sell in list(self._sells.values()):
            if not sell.done:
                await self._poll_order(sell.order)
        for bracket in list(self._brackets.values()):
            if bracket.done:
                continue
            try:
                await self._refresh_bracket(bracket)
            except UnknownOrder:
                logger.warning("bracket %s is not on the exchange", bracket.id)
                self._lost_bracket(bracket)

    async def _poll_order(self, order: Order) -> None:
        try:
            await self._query_order(order)
        except UnknownOrder:
            # Not on the exchange (yet): an unknown submit the engine resolves by query, or
            # the testnet was reset. Nothing to report from here.
            logger.info("order %s is not on the exchange", order.id)

    def _lost_bracket(self, bracket: _Bracket) -> None:
        """A bracket the exchange does not know (a testnet reset): the position is
        unprotected; reported as cancelled."""
        bracket.done = True
        holding = self._holdings.get(bracket.symbol)
        self._untrack_bracket(bracket)
        if holding is None or holding.position.bracket_id != bracket.id:
            return
        order = holding.position.bracket()
        holding.position = replace(holding.position, bracket_id=None)
        holding.protect = False
        if order is not None and not bracket.cancel_requested:
            self._queue.append(
                (bracket.entry_id, OrderCancelled(order, self._clock(), CancelReason.CANCELLED))
            )

    async def _fetch_trades(self, bracket: _Bracket) -> None:
        for progress in bracket.progress.values():
            await self._fetch_order_trades(progress)

    async def _fetch_order_trades(self, progress: _Progress) -> None:
        """Fetch an order's trades when the reports heard do not cover its executed
        quantity (their commissions are needed)."""
        if progress.executed <= 0 or progress.trades_complete:
            return
        if progress.exchange_id is None:
            order = await self._client.get_order(progress.symbol, progress.client_id)
            progress.apply(order)
        assert progress.exchange_id is not None
        for trade in await self._client.my_trades(progress.symbol, progress.exchange_id):
            progress.trades.setdefault(
                trade.trade_id,
                _Trade(
                    trade.quantity, trade.quote_quantity, trade.commission, trade.commission_asset
                ),
            )

    # -- settling: turning progress into events ----------------------------------------------

    async def _settle(self) -> None:
        """Report every order that reached its outcome (fetching missing trades first)."""
        for entry in list(self._entries.values()):
            if not entry.done and entry.progress.final:
                await self._fetch_order_trades(entry.progress)
                await self._finish_entry(entry)
        for sell in list(self._sells.values()):
            if not sell.done and sell.progress.final:
                await self._fetch_order_trades(sell.progress)
                await self._finish_sell(sell)
        for bracket in list(self._brackets.values()):
            if not bracket.done:
                await self._settle_bracket(bracket)

    async def _settle_bracket(self, bracket: _Bracket) -> None:
        progresses = list(bracket.progress.values())
        filled = [p for p in progresses if p.status == "FILLED"]
        if not filled and not all(p.final for p in progresses):
            return
        await self._fetch_trades(bracket)
        holding = self._holdings.get(bracket.symbol)
        bracket.done = True
        self._untrack_bracket(bracket)
        if holding is None or holding.position.bracket_id != bracket.id:
            return
        sold = [p for p in progresses if p.executed > 0]
        if filled:
            leg = filled[0]
            self._close(holding, bracket.id, sold, bracket.legs[leg.client_id])
            return
        if bracket.cancel_requested:
            base = base_asset(bracket.symbol)
            for p in sold:  # carried into the sell that follows
                avg = p.quote / p.executed
                holding.sold_qty += p.executed
                holding.sold_quote += p.quote
                holding.sold_fee += p.commission(QUOTE_ASSET) + p.commission(base) * avg
            holding.position = replace(holding.position, bracket_id=None)
            holding.protect = False
            return
        if sold:  # partly filled, then its list ended without the engine asking
            leg = max(sold, key=lambda p: p.executed)
            self._close(holding, bracket.id, sold, bracket.legs[leg.client_id])
            return
        order = holding.position.bracket()
        holding.position = replace(holding.position, bracket_id=None)
        holding.protect = False
        statuses = {p.status for p in progresses}
        reason = CancelReason.EXPIRED if statuses <= {"EXPIRED"} else CancelReason.CANCELLED
        if order is not None:
            times = [p.time for p in progresses if p.time is not None]
            time = max(times) if times else self._clock()
            self._queue.append((bracket.entry_id, OrderCancelled(order, time, reason)))

    def _take(self, tag: str) -> list[BrokerEvent]:
        """The queued events of order `tag`, taken out of the queue."""
        if not tag:
            return []
        mine = [event for t, event in self._queue if t == tag]
        if mine:
            self._queue = [(t, e) for t, e in self._queue if t != tag]
        return mine

    # -- reconciliation -----------------------------------------------------------------------

    def _known_ids(self) -> set[str]:
        known = set(self._entries) | set(self._sells) | set(self._legs) | set(self._brackets)
        for holding in self._holdings.values():
            if holding.position.bracket_id is not None:
                known.add(holding.position.bracket_id)
                known.update(leg_ids(holding.position.bracket_id))
        return known

    def _candidates(self) -> dict[str, _Holding]:
        """Bracket IDs an unprotected holding's entry may have used -> the holding."""
        out: dict[str, _Holding] = {}
        for holding in self._holdings.values():
            if holding.position.bracket_id is None and needs_bracket(holding.position.exit_plan):
                for attempt in range(MAX_ATTEMPT_SCAN):
                    out[bracket_id(holding.position.entry_order_id, attempt)] = holding
        return out

    async def _reconciliation(self, now: datetime) -> Reconciliation:
        """Refresh balances and open orders, bring known orders up to date (their events are
        queued), adopt lost brackets, and report orphans and balance mismatches."""
        account = await self._client.account()
        open_orders = await self._client.open_orders()
        open_lists = await self._client.open_order_lists()
        before = set(self._holdings)
        candidates = self._candidates()
        for listed in open_lists:
            holding = candidates.get(listed.list_client_order_id)
            if holding is not None and holding.position.bracket_id is None:
                await self._adopt_found(holding, listed.list_client_order_id)
        for order in open_orders:
            holding = candidates.get(order.client_order_id)
            if holding is not None and holding.position.bracket_id is None:
                await self._adopt_found(holding, order.client_order_id)
        await self._poll()
        await self._settle()
        known = self._known_ids()
        orphans: list[OrphanOrder] = []
        reported_lists: set[int] = set()
        list_ids = {ls.order_list_id: ls for ls in open_lists}
        for order in open_orders:
            cid = order.client_order_id
            if cid in known or not cid.startswith(ID_PREFIXES):
                continue
            parent = list_ids.get(order.order_list_id) if order.order_list_id else None
            if parent is not None:
                if parent.list_client_order_id in known:
                    continue
                if parent.order_list_id in reported_lists:
                    continue
                reported_lists.add(parent.order_list_id)
                orphans.append(
                    OrphanOrder(
                        parent.list_client_order_id,
                        order.symbol,
                        Side.SELL if order.side == "SELL" else Side.BUY,
                        "OCO order list unknown locally",
                    )
                )
                continue
            orphans.append(
                OrphanOrder(
                    cid,
                    order.symbol,
                    Side.SELL if order.side == "SELL" else Side.BUY,
                    f"{order.order_type} order unknown locally",
                )
            )
        mismatches = await self._mismatches(account, before)
        return Reconciliation(now, tuple(orphans), tuple(mismatches))

    async def _adopt_found(self, holding: _Holding, bid: str) -> None:
        if await self._adopt_bracket(holding, bid):
            position = holding.position
            logger.warning("%s: adopted bracket %s found on the exchange", position.symbol, bid)
            self._queue.append((position.entry_order_id, BracketPlaced(position, self._clock())))

    async def _mismatches(self, account: Account, before: set[str]) -> list[BalanceMismatch]:
        need: dict[str, Decimal] = {}
        slack: dict[str, Decimal] = {}
        for symbol, holding in self._holdings.items():
            if symbol not in before:
                continue  # opened after the balances were read
            asset = base_asset(symbol)
            need[asset] = need.get(asset, ZERO) + holding.remaining
            rules = await self._rules_for(symbol)
            step = rules.lot_step if rules is not None else ZERO
            slack[asset] = slack.get(asset, ZERO) + step
        out: list[BalanceMismatch] = []
        for asset, quantity in sorted(need.items()):
            actual = account.balance(asset).total
            if actual + slack[asset] < quantity:
                out.append(BalanceMismatch(asset, float(quantity), float(actual)))
        return out
