"""The engine on its live path: a hand-driven feed and a fake clock."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from engine_fakes import (
    Bar,
    FakeClock,
    FakeFeed,
    Row,
    Scheduled,
    flat_bars,
    until,
    write_bars,
)

from crypto_moonshot.broker.base import ExitPlan, Order, Side, client_order_id
from crypto_moonshot.live.config import ConfigError, RunConfig, config_from_dict
from crypto_moonshot.live.contract import CONTRACT_VERSION, SignalBatch, SignalIn
from crypto_moonshot.live.engine import (
    Engine,
    Step,
    build_specs,
    feed_subscriptions,
    feed_warmups,
)
from crypto_moonshot.live.feed import FeedDisconnected, Seeded, StaleSymbolsChanged
from crypto_moonshot.live.replay import prepare_replay
from crypto_moonshot.live.state import (
    BatchStatus,
    EventLevel,
    OrderRecord,
    OrderStatus,
    PositionStatus,
    SignalOutcome,
    StateStore,
)
from crypto_moonshot.models import KLINE_SCHEMA, Interval

pytestmark = pytest.mark.anyio

H1 = Interval.H1
HOUR = timedelta(hours=1)
T0 = datetime(2024, 3, 1, tzinfo=UTC)
SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT")
PLAN = ExitPlan(max_hold=10 * HOUR)


def at(k: int) -> datetime:
    return T0 + k * HOUR


def strategy(name: str, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "source": "tests:not_loaded",
        "symbols": list(SYMBOLS),
        "capital": 1000.0,
        "sizing": {"stake": 100.0},
    }
    entry.update(overrides)
    return entry


def external(name: str = "ext", **overrides: Any) -> dict[str, Any]:
    entry = {"source": "external", "interval": "1h", "warmup": 3, "exit": {"max_hold": "10h"}}
    return strategy(name, **(entry | overrides))


class Live:
    """An engine on a `FakeFeed` with a `FakeClock`, run as a task."""

    def __init__(
        self,
        path: Path,
        strategies: Sequence[dict[str, Any]],
        plugins: dict[str, Any] | None = None,
        *,
        bars: dict[str, list[Bar | None]] | None = None,
        engine: dict[str, Any] | None = None,
        account: dict[str, Any] | None = None,
        history: Any = None,
    ) -> None:
        self.config: RunConfig = config_from_dict(
            {
                "state": str(path),
                "account": {"initial_cash": 10_000.0, **(account or {})},
                "engine": {"signal_deadline": "1.2s", "entry_grace": "30m", **(engine or {})},
                "strategies": list(strategies),
            }
        )
        self.clock = FakeClock(at(0))
        self.feed = FakeFeed(H1, T0, bars or {s: flat_bars(40) for s in SYMBOLS})
        self.store = StateStore.open(path)
        specs = build_specs(self.config, None, plugins=plugins or {})
        self.done: list[datetime] = []
        self.engine = Engine(
            self.config, specs, self.store, clock=self.clock, history=history, step_hook=self._hook
        )
        self.task: asyncio.Task[None] | None = None

    def _hook(self, step: str, tick: Any) -> None:
        if step == Step.TICK_DONE:
            self.done.append(tick.open_time)

    async def start(self) -> None:
        self.task = asyncio.create_task(self.engine.run(self.feed))
        await until(lambda: self.engine._started)

    async def tick(self, k: int, symbols: Sequence[str] | None = None) -> None:
        """Candle k closes (wall clock one second after) and is fully processed."""
        self.clock.now = at(k + 1) + timedelta(seconds=1)
        await self.feed.emit(k, symbols)
        await until(lambda: at(k) in self.done)

    async def close(self) -> None:
        await self.engine.stop()
        assert self.task is not None
        await self.task
        self.store.close()

    def batch(self, name: str, k: int) -> Any:
        return self.store.batch_for(name, H1, at(k))

    def kinds(self) -> list[str]:
        return [e.kind for e in self.store.events_since()]

    def position(self, name: str, symbol: str, k: int) -> Any:
        return self.store.position_for_order(client_order_id(name, symbol, at(k), Side.BUY))


def batch(name: str, k: int, *signals: SignalIn, interval: Interval = H1) -> SignalBatch:
    return SignalBatch(
        contract=CONTRACT_VERSION,
        strategy=name,
        interval=interval,
        candle_time=at(k),
        signals=list(signals),
    )


def enter(symbol: str, priority: float = 0.0) -> SignalIn:
    return SignalIn(symbol=symbol, priority=priority)


def leave(symbol: str) -> SignalIn:
    return SignalIn(symbol=symbol, action="exit")


# -- deadline -------------------------------------------------------------------------------


async def test_the_deadline_misses_a_slow_plugin_and_a_silent_provider(tmp_path: Path) -> None:
    slow = Scheduled({at(0): [("AAAUSDT", 0.0)]}, delay=1.0)
    live = Live(tmp_path / "s.sqlite", [strategy("slow"), external()], {"slow": slow})
    await live.start()
    loop = asyncio.get_running_loop()
    started = loop.time()
    await live.tick(0)
    took = loop.time() - started
    assert 0.15 < took < 0.9  # processed at the deadline (0.2 s left), not after the plugin
    assert live.batch("slow", 0).status is BatchStatus.MISSED
    assert live.batch("ext", 0).status is BatchStatus.MISSED
    assert {"plugin_timeout", "batches_missed"} <= set(live.kinds())
    # The plugin is still computing candle 0 when candle 1 closes: it misses that one too.
    await live.tick(1)
    assert live.batch("slow", 1).status is BatchStatus.MISSED
    assert "plugin_busy" in live.kinds()
    # Its late answer for candle 0 is dropped, never traded.
    await until(lambda: "plugin_late" in live.kinds())
    assert live.store.live_positions() == []
    await live.close()


async def test_a_candle_is_decided_as_soon_as_every_strategy_reported(tmp_path: Path) -> None:
    fast = Scheduled({at(0): [("AAAUSDT", 0.0)]}, exit=PLAN)
    live = Live(
        tmp_path / "s.sqlite",
        [strategy("fast"), external()],
        {"fast": fast},
        engine={"signal_deadline": "30s"},
    )

    async def provider(tick: Any) -> None:
        outcome = await live.engine.submit_batch(batch("ext", 0, enter("BBBUSDT")))
        assert outcome.status == "on_time"
        assert outcome.lateness == timedelta(seconds=1)

    live.engine.add_tick_listener(provider)
    await live.start()
    started = asyncio.get_running_loop().time()
    await live.tick(0)
    assert asyncio.get_running_loop().time() - started < 2
    assert {p.symbol for p in live.store.live_positions()} == {"AAAUSDT", "BBBUSDT"}
    provider_rows = live.store.providers()
    assert [(p.name, p.last_batch_time) for p in provider_rows] == [("ext", at(0))]
    await live.close()


# -- external batches -----------------------------------------------------------------------


async def test_a_late_batch_expires_its_entries_and_still_executes_its_exits(
    tmp_path: Path,
) -> None:
    live = Live(tmp_path / "s.sqlite", [external()])
    posts = {0: [enter("AAAUSDT")], 1: [], 3: [], 4: []}

    async def provider(tick: Any) -> None:
        k = int((tick.open_time - T0) / HOUR)
        if k in posts:
            await live.engine.submit_batch(batch("ext", k, *posts[k]))

    live.engine.add_tick_listener(provider)
    await live.start()
    await live.tick(0)
    await live.tick(1)
    assert live.position("ext", "AAAUSDT", 0).status is PositionStatus.OPEN
    await live.tick(2)  # the provider is silent: missed at the deadline
    assert live.batch("ext", 2).status is BatchStatus.MISSED
    live.clock.now = at(3) + timedelta(seconds=40)
    late = await live.engine.submit_batch(batch("ext", 2, enter("BBBUSDT"), leave("AAAUSDT")))
    assert late.status == "late"
    assert late.lateness == timedelta(seconds=40)
    record = live.batch("ext", 2)
    assert (record.status, record.lateness) == (BatchStatus.LATE, timedelta(seconds=40))
    outcomes = {(s.symbol, s.action): (s.outcome, s.reason) for s in record.signals}
    assert outcomes[("BBBUSDT", "enter")] == (SignalOutcome.EXPIRED, "late")
    assert outcomes[("AAAUSDT", "exit")] == (SignalOutcome.PLACED, None)
    sell = live.store.order(client_order_id("ext", "AAAUSDT", at(2), Side.SELL))
    assert sell is not None
    assert sell.time == at(4)  # the next candle open after it arrived
    duplicate = await live.engine.submit_batch(batch("ext", 2))
    assert duplicate.status == "duplicate"
    await live.tick(3)
    await live.tick(4)
    closed = live.store.closed_positions()
    assert [(p.symbol, p.exit_time, p.exit_reason) for p in closed] == [
        ("AAAUSDT", at(4), "signal")
    ]
    assert "BBBUSDT" not in {p.symbol for p in live.store.live_positions()}
    assert "batch_late" in live.kinds()
    await live.close()


async def test_invalid_and_duplicate_batches_are_refused(tmp_path: Path) -> None:
    plugin = Scheduled({})
    live = Live(tmp_path / "s.sqlite", [external(), strategy("inproc")], {"inproc": plugin})
    refused = await live.engine.submit_batch(batch("ext", 0))
    assert (refused.status, refused.reason) == ("refused", "the engine is not running")
    await live.start()
    live.clock.now = at(1)
    first = await live.engine.submit_batch(batch("ext", 0))
    assert first.status == "on_time"  # before the engine saw tick 0: it joins that tick
    assert first.accepted
    again = await live.engine.submit_batch(batch("ext", 0, enter("AAAUSDT")))
    assert (again.status, again.accepted) == ("duplicate", False)
    cases = [
        (batch("nobody", 0), "unknown strategy"),
        (batch("inproc", 0), "runs in process"),
        (batch("ext", 0, interval=Interval.M5), "trades 1h, not 5m"),
        (batch("ext", 1), "has not closed yet"),
    ]
    for item, reason in cases:
        outcome = await live.engine.submit_batch(item)
        assert outcome.status == "refused"
        assert outcome.reason is not None
        assert reason in outcome.reason
    await live.tick(0)
    assert live.batch("ext", 0).status is BatchStatus.ON_TIME
    await live.close()


# -- plugins --------------------------------------------------------------------------------


class BadFrame:
    name = "bad"
    interval = H1
    warmup = 2
    context_symbols: tuple[str, ...] = ()
    exit = None

    def on_candles(self, windows: Any, time: datetime, context: Any) -> pl.DataFrame:
        return pl.DataFrame({"symbol": ["AAAUSDT"]})  # no priority


async def test_a_plugin_that_raises_or_breaks_the_contract_misses_its_batch(
    tmp_path: Path,
) -> None:
    flaky = Scheduled({at(1): [("AAAUSDT", 0.0)]}, fail_at=[at(0)], exit=PLAN)
    live = Live(
        tmp_path / "s.sqlite",
        [strategy("flaky"), strategy("bad")],
        {"flaky": flaky, "bad": BadFrame()},
    )
    await live.start()
    await live.tick(0)
    assert live.batch("flaky", 0).status is BatchStatus.MISSED
    assert live.batch("bad", 0).status is BatchStatus.MISSED
    errors = [e for e in live.store.events_since() if e.kind == "plugin_error"]
    assert len(errors) == 2
    assert any("RuntimeError: boom" in e.message for e in errors)
    assert any("priority" in e.message for e in errors)
    await live.tick(1)  # the engine carries on
    assert live.position("flaky", "AAAUSDT", 1).status is PositionStatus.PENDING
    await live.close()


async def test_plugin_windows_skip_symbols_without_a_candle_in_the_tick(tmp_path: Path) -> None:
    plugin = Scheduled({}, warmup=2)
    live = Live(tmp_path / "s.sqlite", [strategy("p", symbols=["AAAUSDT"])], {"p": plugin})
    await live.start()
    await live.tick(0, ["BBBUSDT"])  # none of its symbols closed candle 0: not called
    await live.tick(1)
    assert plugin.calls == [at(1)]
    assert live.batch("p", 0).status is BatchStatus.ON_TIME
    await live.close()


# -- controls -------------------------------------------------------------------------------


async def test_pause_resume_and_flatten(tmp_path: Path) -> None:
    schedule: dict[datetime, list[Row]] = {
        at(0): [("AAAUSDT", 0.0)],
        at(2): [("BBBUSDT", 0.0)],
        at(3): [("AAAUSDT", 0.0, "exit")],
        at(4): [("CCCUSDT", 0.0), ("DDDUSDT", 0.0), ("FFFUSDT", 0.0)],
        at(5): [("EEEUSDT", 0.0)],
    }
    plugin = Scheduled(schedule, exit=PLAN)
    live = Live(tmp_path / "s.sqlite", [strategy("s")], {"s": plugin})
    await live.start()
    await live.tick(0)
    await live.tick(1)
    live.engine.pause()
    await live.tick(2)
    assert live.batch("s", 2).signals[0].reason == "paused"
    await live.tick(3)  # exits still work while paused
    assert live.batch("s", 3).signals[0].outcome is SignalOutcome.PLACED
    live.engine.resume()
    await live.tick(4)
    assert live.store.closed_positions()[0].exit_reason == "signal"
    assert live.store.controls().paused is False
    live.engine.flatten()
    await live.tick(5, [s for s in SYMBOLS if s != "FFFUSDT"])  # FFF's entry candle is late
    controls = live.store.controls()
    assert (controls.paused, controls.flatten_requested) == (True, False)
    assert live.batch("s", 5).signals[0].reason == "paused"
    fff = live.position("s", "FFFUSDT", 4)
    assert (fff.status, fff.exit_reason) == (PositionStatus.CANCELLED, "flatten")
    await live.tick(6)
    closed = {p.symbol: (p.exit_time, p.exit_reason) for p in live.store.closed_positions()}
    assert closed["CCCUSDT"] == (at(6), "flatten")
    assert closed["DDDUSDT"] == (at(6), "flatten")
    assert live.store.live_positions() == []
    assert {"paused", "resumed", "flatten_requested", "flatten"} <= set(live.kinds())
    await live.close()


# -- account checks -------------------------------------------------------------------------


async def test_a_stale_symbol_is_refused(tmp_path: Path) -> None:
    live = Live(tmp_path / "s.sqlite", [external()])
    await live.start()
    live.clock.now = at(1)
    await live.engine.submit_batch(batch("ext", 0))
    await live.tick(0)
    others = [s for s in SYMBOLS if s != "SSSUSDT" and s != "FFFUSDT"]
    live.clock.now = at(2)
    await live.engine.submit_batch(batch("ext", 1))
    await live.tick(1, others)
    live.clock.now = at(3)
    await live.engine.submit_batch(batch("ext", 2, enter("FFFUSDT"), enter("AAAUSDT")))
    await live.tick(2, others)  # FFF's latest candle opened 3 hours before the forming one
    outcomes = {s.symbol: (s.outcome, s.reason) for s in live.batch("ext", 2).signals}
    assert outcomes["FFFUSDT"] == (SignalOutcome.REJECTED, "stale")
    assert outcomes["AAAUSDT"] == (SignalOutcome.PLACED, None)
    await live.close()


async def test_a_pending_buy_without_its_candle_is_cancelled_after_the_grace(
    tmp_path: Path,
) -> None:
    live = Live(tmp_path / "s.sqlite", [external()])
    await live.start()
    live.clock.now = at(1)
    await live.engine.submit_batch(batch("ext", 0, enter("EEEUSDT"), enter("FFFUSDT")))
    await live.tick(0)
    book = live.engine.books["ext"]
    assert book.cash == pytest.approx(800.0)
    without = [s for s in SYMBOLS if s not in ("EEEUSDT", "FFFUSDT")]
    await live.engine.submit_batch(batch("ext", 1))
    await live.tick(1, without)  # neither entry candle has arrived
    assert live.position("ext", "EEEUSDT", 0).status is PositionStatus.PENDING
    live.feed.late("FFFUSDT", 1)  # FFF's candle 1 turns up late, inside the grace
    live.clock.now = at(3)
    await live.engine.submit_batch(batch("ext", 2))
    await live.tick(2, without)
    eee = live.position("ext", "EEEUSDT", 0)
    assert (eee.status, eee.exit_reason, eee.exit_time) == (
        PositionStatus.CANCELLED,
        "no_candle",
        at(3) + timedelta(seconds=1),
    )
    order = live.store.order(eee.entry_order_id)
    assert order is not None
    assert (order.status, order.reason) == (OrderStatus.CANCELLED, "no_candle")
    fff = live.position("ext", "FFFUSDT", 0)
    assert (fff.status, fff.entry_time) == (PositionStatus.OPEN, at(1))
    assert live.engine.books["ext"].cash == pytest.approx(900.0)
    assert live.engine.books["ext"] == live.store.derive_book("ext", 1000.0, live.clock())
    assert "entries_cancelled" in live.kinds()
    await live.close()


async def test_status_snapshot(tmp_path: Path) -> None:
    live = Live(tmp_path / "s.sqlite", [external()])
    await live.start()
    live.clock.now = at(1)
    await live.engine.submit_batch(batch("ext", 0, enter("AAAUSDT")))
    await live.tick(0)
    status = live.engine.status()
    assert status.mode == "paper"
    assert status.running
    assert status.books["ext"].open_positions == 1
    assert [p.symbol for p in status.positions] == ["AAAUSDT"]
    assert [o.side for o in status.open_orders] == [Side.BUY]
    assert status.last_ticks == {"1h": at(0)}
    assert status.stale_symbols == {"1h": []}
    document = status.to_json()
    assert document["books"]["ext"]["cash"] == pytest.approx(900.0)
    assert document["positions"][0]["placed_at"] == at(1).isoformat()
    assert document["health"]["feed"]["connected"] is True
    await live.close()


# -- restarts -------------------------------------------------------------------------------


class Rest:
    """A fake `KlineFetcher` over the feed's bars."""

    def __init__(self, feed: FakeFeed) -> None:
        self.feed = feed
        self.calls: list[tuple[str, datetime, datetime]] = []

    async def fetch_klines(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> pl.DataFrame:
        self.calls.append((symbol, start, end))
        rows = []
        k = 0
        while self.feed.time(k) < end:
            candle = self.feed.candle(symbol, k)
            if candle is not None and candle.open_time >= start:
                rows.append(candle)
            k += 1
        return pl.DataFrame(rows, schema=KLINE_SCHEMA, orient="row")


@pytest.mark.parametrize("windows_reach_back", [True, False])
async def test_a_restart_applies_exits_that_happened_while_down(
    tmp_path: Path, windows_reach_back: bool
) -> None:
    bars = {s: flat_bars(40) for s in SYMBOLS}
    bars["AAAUSDT"][4] = (100.0, 106.0, 100.0, 104.0)  # the take profit is hit in candle 4
    plan = ExitPlan(take_profit=0.05, max_hold=20 * HOUR)
    plugin = Scheduled({at(0): [("AAAUSDT", 0.0)]}, exit=plan)
    path = tmp_path / "s.sqlite"
    live = Live(path, [strategy("s")], {"s": plugin}, bars=bars)
    await live.start()
    await live.tick(0)
    await live.tick(1)
    assert live.position("s", "AAAUSDT", 0).status is PositionStatus.OPEN
    await live.close()

    # Down from candle 2 to 9; back for candle 10.
    again = Live(path, [strategy("s")], {"s": Scheduled({})}, bars=bars)
    rest = Rest(again.feed)
    again.engine.history = rest
    again.feed.seed(10 if windows_reach_back else 0)
    if not windows_reach_back:
        for k in range(8, 10):
            for symbol in SYMBOLS:
                again.feed.late(symbol, k)
    await again.start()
    await again.tick(10)
    position = again.position("s", "AAAUSDT", 0)
    assert (position.status, position.exit_reason, position.exit_time) == (
        PositionStatus.CLOSED,
        "take_profit",
        at(5),
    )
    assert bool(rest.calls) is not windows_reach_back
    missed = [e for e in again.store.events_since() if e.kind == "ticks_missed"]
    assert len(missed) == 1
    assert missed[0].message.startswith("8 1h tick(s)")
    assert "restart" in again.kinds()
    await again.close()


async def test_a_restart_refuses_a_contradicting_config(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    live = Live(path, [external(), external("other")])
    await live.start()
    live.clock.now = at(1)
    await live.engine.submit_batch(batch("ext", 0, enter("AAAUSDT")))
    await live.tick(0)
    await live.close()

    cases = [
        ([external(capital=2000.0), external("other")], {}, "capital changed from 1000 to 2000"),
        ([external(), external("other")], {"initial_cash": 5000.0}, "initial_cash is 5000"),
        ([external("other")], {}, "strategy ext has 1 open or pending position"),
    ]
    for strategies, account, message in cases:
        config = config_from_dict(
            {
                "state": str(path),
                "account": {"initial_cash": 10_000.0, **account},
                "strategies": strategies,
            }
        )
        with StateStore.open(path) as store:
            engine = Engine(config, build_specs(config, None), store)
            with pytest.raises(ConfigError, match=message):
                engine.start()
    # Removing a strategy without positions is fine.
    config = config_from_dict(
        {"state": str(path), "account": {"initial_cash": 10_000.0}, "strategies": [external()]}
    )
    with StateStore.open(path) as store:
        Engine(config, build_specs(config, None), store).start()


def test_the_engine_refuses_testnet_and_live(tmp_path: Path) -> None:
    for mode in ("testnet", "live"):
        config = config_from_dict(
            {
                "mode": mode,
                "state": str(tmp_path / "s.sqlite"),
                "account": {"initial_cash": 1000.0},
                "strategies": [external()],
            }
        )
        with StateStore.open(tmp_path / "s.sqlite") as store, pytest.raises(ConfigError):
            Engine(config, build_specs(config, None), store)


# -- several strategies ---------------------------------------------------------------------


def _multi_run(
    root: Path, path: Path, strategies: list[dict[str, Any]], plugins: dict[str, Any]
) -> tuple[pl.DataFrame, StateStore, int]:
    """Replay the strategies over the flat store; checks every book against `derive_book`
    after every tick. Returns the trades, the open store and the number of ticks checked."""
    config = config_from_dict(
        {
            "state": str(path),
            "account": {
                "initial_cash": 2000.0,
                "max_order_usdt": 150.0,
                "max_total_exposure": 0.2,
            },
            "strategies": strategies,
        }
    )
    checked = [0]
    box: dict[str, Engine] = {}

    def hook(step: str, tick: Any) -> None:
        if step != Step.TICK_DONE:
            return
        engine = box["engine"]
        for name, book in engine.books.items():
            capital = engine.specs[name].capital
            assert book == engine.store.derive_book(name, capital, engine.clock())
        checked[0] += 1

    run = prepare_replay(config, at(1), at(14), root, plugins=plugins, step_hook=hook)
    box["engine"] = run.engine
    result = asyncio.run(run.run())
    return result.trades, run.store, checked[0]


def test_strategies_trade_independently_under_the_account_rules(tmp_path: Path) -> None:
    for symbol in SYMBOLS:
        write_bars(tmp_path, symbol, flat_bars(20), H1, at(0))
    hold = ExitPlan(max_hold=3 * HOUR)
    alpha = {
        at(1): [("AAAUSDT", 0.0)],
        at(2): [("CCCUSDT", -1.0)],  # beats beta's CCC on priority
        at(3): [("AAAUSDT", 0.0)],  # still holding AAA: one position per symbol
        at(5): [("FFFUSDT", 0.0)],  # ties with beta's FFF: strategy name decides
    }
    beta = {
        at(1): [("BBBUSDT", 0.0)],
        at(2): [("CCCUSDT", 0.0)],
        at(3): [("DDDUSDT", 0.0)],  # 100 + 120 + 100 open, + 120 > 0.2 x 2000
        at(5): [("FFFUSDT", 0.0)],
    }
    gamma = {at(3): [("EEEUSDT", 0.0)]}  # a 200 stake > max_order_usdt
    strategies = [
        strategy("alpha", capital=1000.0),
        strategy("beta", capital=500.0, sizing={"stake": 120.0}),
        strategy("gamma", capital=500.0, sizing={"stake": 200.0}),
    ]
    plugins = {
        "alpha": Scheduled(alpha, name="alpha", exit=hold),
        "beta": Scheduled(beta, name="beta", exit=hold),
        "gamma": Scheduled(gamma, name="gamma", exit=hold),
    }
    trades, store, checked = _multi_run(tmp_path, tmp_path / "multi.sqlite", strategies, plugins)
    assert checked == 13
    held = {(r["strategy"], r["symbol"], r["entry_time"]) for r in trades.iter_rows(named=True)}
    assert held == {
        ("alpha", "AAAUSDT", at(2)),
        ("beta", "BBBUSDT", at(2)),
        ("alpha", "CCCUSDT", at(3)),
        ("alpha", "FFFUSDT", at(6)),
    }
    reasons = {
        (b.strategy, b.candle_time, s.symbol): s.reason
        for k in range(1, 14)
        for b in store.batches_for(H1, at(k))
        for s in b.signals
        if s.outcome is SignalOutcome.REJECTED
    }
    assert reasons == {
        ("beta", at(2), "CCCUSDT"): "symbol_held",
        ("alpha", at(3), "AAAUSDT"): "symbol_held",
        ("beta", at(3), "DDDUSDT"): "max_total_exposure",
        ("gamma", at(3), "EEEUSDT"): "max_order_usdt",
        ("beta", at(5), "FFFUSDT"): "symbol_held",
    }
    # Separate books: each strategy's cash moved only with its own trades.
    for name, capital in (("alpha", 1000.0), ("beta", 500.0), ("gamma", 500.0)):
        own = trades.filter(pl.col("strategy") == name)
        book = store.derive_book(name, capital, at(14))
        assert book.cash == pytest.approx(capital + float(own["pnl"].sum()))
    store.close()

    # Alpha alone (no account limit ever refused it) trades exactly the same.
    alone, store, _ = _multi_run(
        tmp_path,
        tmp_path / "alone.sqlite",
        [strategy("alpha", capital=1000.0)],
        {"alpha": Scheduled(alpha, name="alpha", exit=hold)},
    )
    store.close()
    mine = trades.filter(pl.col("strategy") == "alpha")
    assert alone.select(mine.columns).equals(mine)


def test_a_replay_does_not_wait_for_a_plugin_that_failed(tmp_path: Path) -> None:
    for symbol in ("AAAUSDT", "BBBUSDT"):
        write_bars(tmp_path, symbol, flat_bars(10), H1, at(0))
    plugin = Scheduled({at(3): [("AAAUSDT", 0.0)]}, name="p", fail_at=[at(2)], exit=PLAN)
    config = config_from_dict(
        {
            "state": str(tmp_path / "s.sqlite"),
            "account": {"initial_cash": 1000.0},
            "engine": {"replay_timeout": "30s"},
            "strategies": [strategy("p", symbols=["AAAUSDT", "BBBUSDT"])],
        }
    )
    run = prepare_replay(config, at(1), at(8), tmp_path, plugins={"p": plugin})
    started = asyncio.run(_timed(run.run()))
    assert started < 5  # not the 30 s replay timeout
    assert run.store.batch_for("p", H1, at(2)).status is BatchStatus.MISSED  # type: ignore[union-attr]
    assert [e.kind for e in run.store.events_since()].count("plugin_error") == 1
    assert len(run.store.closed_positions()) + len(run.store.live_positions()) == 1
    run.close()


async def _timed(coro: Any) -> float:
    loop = asyncio.get_running_loop()
    started = loop.time()
    await coro
    return loop.time() - started


async def test_stop_abandons_the_tick_being_collected(tmp_path: Path) -> None:
    live = Live(tmp_path / "s.sqlite", [external()], engine={"signal_deadline": "60s"})
    await live.start()
    live.clock.now = at(1)
    await live.feed.emit(0)
    await until(lambda: H1 in live.engine._collecting)
    await live.engine.stop()
    assert live.task is not None
    await live.task
    assert live.store.last_tick(H1) is None
    assert live.store.batches_for(H1, at(0)) == []  # nothing recorded as missed
    live.store.close()


def test_feed_events_go_to_the_event_feed(tmp_path: Path) -> None:
    config = config_from_dict(
        {
            "state": str(tmp_path / "s.sqlite"),
            "account": {"initial_cash": 1000.0},
            "strategies": [external()],
        }
    )
    with StateStore.open(config.state) as store:
        engine = Engine(config, build_specs(config, None), store, clock=FakeClock(at(3)))
        engine.on_feed_event(FeedDisconnected(0, "closed by peer"))
        engine.on_feed_event(Seeded(H1, 6, 120, 100))
        engine.on_feed_event(StaleSymbolsChanged(H1, ("AAAUSDT",)))
        engine.on_feed_event(StaleSymbolsChanged(H1, ()))
        events = store.events_since()
    assert [(e.kind, e.level) for e in events] == [
        ("feed_disconnected", EventLevel.WARNING),
        ("seeded", EventLevel.INFO),
        ("stale_symbols_changed", EventLevel.WARNING),
        ("stale_symbols_changed", EventLevel.INFO),
    ]
    assert events[0].data == {"connection": 0, "reason": "closed by peer"}
    assert events[2].data == {"interval": "1h", "symbols": ["AAAUSDT"]}
    assert events[0].time == at(3)


def test_feed_subscriptions_cover_traded_context_and_held_symbols(tmp_path: Path) -> None:
    config = config_from_dict(
        {
            "state": str(tmp_path / "s.sqlite"),
            "account": {"initial_cash": 2000.0},
            "strategies": [
                external(symbols=["AAAUSDT", "BBBUSDT"], warmup=50),
                external("five", symbols=["BBBUSDT"], interval="5m", warmup=7),
            ],
        }
    )
    specs = build_specs(config, None)
    with StateStore.open(config.state) as store:
        store.upsert_strategy("ext", {}, 1000.0)
        store.insert_order(
            OrderRecord.from_order(
                Order("ZZZUSDT", Side.BUY, at(1), quote_amount=10.0, id="z"), "ext", H1
            )
        )
        store.insert_position("ext", "ZZZUSDT", H1, at(1), "z", 10.0)
        held = store.live_positions()
    assert feed_subscriptions(specs) == {H1: ["AAAUSDT", "BBBUSDT"], Interval.M5: ["BBBUSDT"]}
    assert feed_subscriptions(specs, held)[H1] == ["AAAUSDT", "BBBUSDT", "ZZZUSDT"]
    assert feed_warmups(specs) == {H1: 50, Interval.M5: 7}
