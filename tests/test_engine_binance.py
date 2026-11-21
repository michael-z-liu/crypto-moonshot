"""The real engine trading through `BinanceBroker` on the fake exchange: entries, brackets, a take
profit, an exit signal, an engine time exit, flatten, the between-ticks drain, unprotected
positions, reconciliation at start (orphans, balance mismatches) and crash recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from binance_fakes import BrokerRig, Clock, make_rig
from engine_fakes import FakeFeed, Row, Scheduled, dump_state, flat_bars
from fake_exchange import FakeExchange

from crypto_moonshot.broker.base import ExitPlan, Side, client_order_id
from crypto_moonshot.broker.binance import bracket_id
from crypto_moonshot.live.config import RunConfig, config_from_dict
from crypto_moonshot.live.engine import Engine, Step, build_specs, time_exit_id
from crypto_moonshot.live.health import HealthMonitor
from crypto_moonshot.live.state import (
    EventLevel,
    OrderStatus,
    PositionRecord,
    PositionStatus,
    SignalOutcome,
    StateStore,
)
from crypto_moonshot.models import Interval

pytestmark = pytest.mark.anyio

D = Decimal
H1 = Interval.H1
HOUR = timedelta(hours=1)
SECOND = timedelta(seconds=1)
T0 = datetime(2024, 3, 1, tzinfo=UTC)
SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")
PLAN = ExitPlan(take_profit=0.1, stop=0.1, max_hold=3 * HOUR)


def at(k: int) -> datetime:
    return T0 + k * HOUR


def buy_id(symbol: str, k: int) -> str:
    return client_order_id("s", symbol, at(k), Side.BUY)


class Crash(Exception):
    """An injected failure."""


def crash_at(point: str, n: int) -> Callable[[str, Any], None]:
    seen = [0]

    def hook(step: str, tick: Any) -> None:
        if step == point:
            seen[0] += 1
            if seen[0] == n:
                raise Crash(f"{point} #{n}")

    return hook


def _plenty(path: Path) -> int:
    return 10**12


class Rig:
    """The engine on a `FakeFeed` and a `BinanceBroker` over a `FakeExchange`, sharing a clock."""

    def __init__(
        self,
        path: Path,
        schedule: Mapping[datetime, Sequence[Row]],
        exchange: BrokerRig,
        *,
        hook: Callable[[str, Any], None] | None = None,
        drain: str = "none",
        plan: ExitPlan | None = PLAN,
    ) -> None:
        self.exchange = exchange
        self.clock = exchange.clock
        self.fake = exchange.fake
        self.config: RunConfig = config_from_dict(
            {
                "state": str(path),
                "account": {"initial_cash": 10_000.0},
                "engine": {
                    "signal_deadline": "1.2s",
                    "entry_grace": "30m",
                    "drain_interval": drain,
                },
                "strategies": [
                    {
                        "name": "s",
                        "source": "tests:not_loaded",
                        "symbols": list(SYMBOLS),
                        "capital": 1000.0,
                        "sizing": {"stake": 100.0},
                    }
                ],
            }
        )
        self.feed = FakeFeed(H1, T0, {s: flat_bars(40) for s in SYMBOLS})
        self.store = StateStore.open(path)
        self.plugin = Scheduled(schedule, exit=plan)
        self.done: list[datetime] = []
        self.crash = hook
        specs = build_specs(self.config, None, plugins={"s": self.plugin})
        self.engine = Engine(
            self.config,
            specs,
            self.store,
            clock=self.clock,
            step_hook=self._hook,
            broker=exchange.broker,
        )
        self.task: asyncio.Task[None] | None = None

    @classmethod
    async def create(
        cls,
        path: Path,
        schedule: Mapping[datetime, Sequence[Row]],
        *,
        clock: Clock | None = None,
        fake: FakeExchange | None = None,
        **kwargs: Any,
    ) -> Rig:
        clock = clock or Clock(at(0))
        exchange = await make_rig(clock=clock, fake=fake)
        return cls(path, schedule, exchange, **kwargs)

    def _hook(self, step: str, tick: Any) -> None:
        if self.crash is not None:
            self.crash(step, tick)
        if step == Step.TICK_DONE:
            self.done.append(tick.open_time)

    async def start(self) -> None:
        self.task = asyncio.create_task(self.engine.run(self.feed))
        await self.until(lambda: self.engine.reconciliation is not None)

    async def until(self, check: Callable[[], bool], timeout: float = 5.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not check():
            task = self.task
            if task is not None and task.done():
                task.result()
                msg = "the engine stopped"
                raise AssertionError(msg)
            if loop.time() > deadline:
                msg = "condition not reached"
                raise TimeoutError(msg)
            await asyncio.sleep(0.003)

    async def tick(self, k: int) -> None:
        """Candle k closes (a second before the clock) and is processed."""
        await self.exchange.delivered()
        self.clock.now = at(k + 1) + SECOND
        await self.feed.emit(k)
        await self.until(lambda: at(k) in self.done)

    async def close(self) -> None:
        task = self.task
        if task is not None and not task.done():
            await self.engine.stop()
            await task
        self.store.close()
        await self.exchange.close()

    def position(self, symbol: str, k: int) -> PositionRecord:
        position = self.store.position_for_order(buy_id(symbol, k))
        assert position is not None
        return position

    def kinds(self) -> list[str]:
        return [e.kind for e in self.store.events_since()]

    def book_matches(self) -> bool:
        return self.engine.books["s"] == self.store.derive_book("s", 1000.0, self.clock())


# -- a trading session ------------------------------------------------------------------------


async def test_a_session_of_entries_brackets_signals_time_exits_and_flatten(
    tmp_path: Path,
) -> None:
    schedule: dict[datetime, list[Row]] = {
        at(0): [("AAAUSDT", 0.0), ("BBBUSDT", 1.0), ("CCCUSDT", 2.0)],
        at(1): [("BBBUSDT", 0.0, "exit")],
        at(3): [("DDDUSDT", 0.0)],
    }
    rig = await Rig.create(tmp_path / "s.sqlite", schedule)
    await rig.start()
    try:
        await rig.tick(0)
        for symbol in ("AAAUSDT", "BBBUSDT", "CCCUSDT"):
            position = rig.position(symbol, 0)
            assert position.status is PositionStatus.OPEN  # filled with the acknowledgement
            assert position.entry_price == pytest.approx(100.0)
            assert position.quantity == pytest.approx(0.999)  # 1.000 less 0.001 commission
            assert position.bracket_id == bracket_id(buy_id(symbol, 0))
            assert rig.fake.lists[position.bracket_id].status == "EXECUTING"
        assert rig.book_matches()

        rig.fake.set_price("AAAUSDT", D("111"))  # AAA's take profit fills between ticks
        await rig.tick(1)
        aaa = rig.position("AAAUSDT", 0)
        assert (aaa.status, aaa.exit_reason) == (PositionStatus.CLOSED, "take_profit")
        assert aaa.exit_price == pytest.approx(110.0)
        bbb = rig.position("BBBUSDT", 0)  # the exit signal: bracket cancelled, then sold
        assert (bbb.status, bbb.exit_reason) == (PositionStatus.CLOSED, "signal")
        assert rig.fake.lists[bracket_id(buy_id("BBBUSDT", 0))].legs[0].status == "CANCELED"

        await rig.tick(2)
        await rig.tick(3)  # CCC's holding time is up at the entry time of tick 3
        ccc = rig.position("CCCUSDT", 0)
        assert (ccc.status, ccc.exit_reason) == (PositionStatus.CLOSED, "time")
        sell = rig.store.order(time_exit_id(buy_id("CCCUSDT", 0), at(4)))
        assert sell is not None
        assert sell.status is OrderStatus.FILLED
        assert rig.position("DDDUSDT", 3).status is PositionStatus.OPEN

        rig.engine.flatten()
        await rig.tick(4)
        ddd = rig.position("DDDUSDT", 3)
        assert (ddd.status, ddd.exit_reason) == (PositionStatus.CLOSED, "flatten")
        assert rig.store.controls().paused
        assert rig.store.pending_orders() == []
        assert not any(o.open for o in rig.fake.orders.values())
        assert len(rig.store.fills()) == 8
        assert rig.book_matches()
        pnl = sum(p.pnl or 0.0 for p in rig.store.closed_positions())
        assert rig.engine.books["s"].cash == pytest.approx(1000.0 + pnl)
        assert rig.exchange.hosts() <= {"testnet.invalid", "127.0.0.1"}
    finally:
        await rig.close()


async def test_the_drain_records_a_fill_before_the_next_tick(tmp_path: Path) -> None:
    rig = await Rig.create(tmp_path / "s.sqlite", {at(0): [("AAAUSDT", 0.0)]}, drain="0.01s")
    await rig.start()
    try:
        await rig.tick(0)
        assert rig.position("AAAUSDT", 0).status is PositionStatus.OPEN
        rig.fake.set_price("AAAUSDT", D("89"))  # the stop
        await rig.until(lambda: rig.position("AAAUSDT", 0).status is PositionStatus.CLOSED)
        assert rig.position("AAAUSDT", 0).exit_reason == "stop"
        assert at(1) not in rig.done  # no tick needed
        assert rig.book_matches()
        assert rig.engine.books["s"].open_positions == 0
    finally:
        await rig.close()


async def test_an_unprotected_position_alerts_until_its_bracket_is_placed(tmp_path: Path) -> None:
    rig = await Rig.create(tmp_path / "s.sqlite", {at(0): [("AAAUSDT", 0.0)]})
    rig.fake.fail("/api/v3/orderList/oco", code=-1013, msg="Filter failure: PERCENT_PRICE_BY_SIDE")
    await rig.start()
    try:
        await rig.tick(0)
        position = rig.position("AAAUSDT", 0)
        assert (position.status, position.bracket_id) == (PositionStatus.OPEN, None)
        assert "bracket_failed" in rig.kinds()
        monitor = HealthMonitor(rig.engine, rig.feed, rig.config.health, disk_free=_plenty)
        found = {f.key: f.level for f in monitor.check()}
        assert found["unprotected_position:AAAUSDT"] is EventLevel.ERROR
        await rig.tick(1)  # the broker retries (its backoff is long past)
        position = rig.position("AAAUSDT", 0)
        assert position.bracket_id == bracket_id(buy_id("AAAUSDT", 0))
        assert "bracket_placed" in rig.kinds()
        assert rig.engine.unprotected_positions() == []
        monitor.check()
        assert "unprotected_position:AAAUSDT" not in {a.key for a in rig.store.alerts()}
    finally:
        await rig.close()


async def test_an_auth_failure_halts_and_alerts(tmp_path: Path) -> None:
    rig = await Rig.create(tmp_path / "s.sqlite", {at(0): [("AAAUSDT", 0.0)]})
    rig.fake.fail("/api/v3/order", method="POST", status=401, code=-2015, msg="Invalid API-key")
    await rig.start()
    try:
        await rig.tick(0)
        position = rig.position("AAAUSDT", 0)
        assert (position.status, position.exit_reason) == (PositionStatus.CANCELLED, "broker_auth")
        await rig.tick(1)
        assert "broker_halted" in rig.kinds()
        monitor = HealthMonitor(rig.engine, rig.feed, rig.config.health, disk_free=_plenty)
        assert "broker_halted" in {f.key for f in monitor.check()}
    finally:
        await rig.close()


# -- reconciliation at start ------------------------------------------------------------------


async def test_reconciliation_at_start_reports_orphans_and_mismatches(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    first = await Rig.create(path, {at(0): [("AAAUSDT", 0.0)]})
    await first.start()
    await first.tick(0)
    await first.close()
    fake, clock = first.fake, first.clock
    fake.balances["AAA"] = [D(0), D(0)]  # the coins are gone (and the bracket's lock with them)
    fake.deposit("DDD", D(5))
    orphan = "cmx-" + "0" * 32
    fake._new_order(
        {
            "symbol": "DDDUSDT",
            "side": "SELL",
            "type": "LIMIT_MAKER",
            "quantity": "1",
            "price": "150",
            "newClientOrderId": orphan,
        }
    )
    clock.now = at(2) + SECOND
    again = await Rig.create(path, {at(2): [("BBBUSDT", 0.0)]}, clock=clock, fake=fake)
    await again.start()
    try:
        events = {e.kind: e for e in again.store.events_since()}
        assert events["orphan_order"].data is not None
        assert events["orphan_order"].data["order_id"] == orphan
        assert events["balance_mismatch"].level is EventLevel.ERROR
        assert again.store.controls().paused
        monitor = HealthMonitor(again.engine, again.feed, again.config.health, disk_free=_plenty)
        found = {f.key: f.level for f in monitor.check()}
        assert found["orphan_order"] is EventLevel.WARNING
        assert found["balance_mismatch"] is EventLevel.ERROR
        await again.tick(2)
        batch = again.store.batch_for("s", H1, at(2))
        assert batch is not None
        (signal,) = batch.signals
        assert (signal.outcome, signal.reason) == (SignalOutcome.REJECTED, "paused")
    finally:
        await again.close()


# -- crash recovery ---------------------------------------------------------------------------

SCENARIO: dict[datetime, list[Row]] = {
    at(0): [("AAAUSDT", 0.0), ("BBBUSDT", 1.0)],
    at(2): [("CCCUSDT", 0.0)],
}
TICKS = 8


async def run_scenario(
    path: Path, crash: tuple[str, int] | None = None
) -> tuple[dict[str, list[tuple[object, ...]]], FakeExchange]:
    """Ticks 0..7 of `SCENARIO`: entries at 0 and 2, AAA's take profit while the engine may be
    down after tick 1, BBB's time exit at 3 and CCC's at 5. With `crash`, the engine fails there
    and a new engine with a new broker process (over the same exchange) resumes."""
    hook = None if crash is None else crash_at(*crash)
    rig = await Rig.create(path, SCENARIO, hook=hook)
    await rig.start()
    k = 0
    while k < TICKS:
        try:
            await rig.tick(k)
        except Crash:
            assert crash is not None
            fake, clock = rig.fake, rig.clock
            rig.store.close()
            await rig.exchange.close()
            if k >= 1 and fake.prices["AAAUSDT"] != D("111"):
                fake.set_price("AAAUSDT", D("111"))  # fills while nothing runs
            rig = await Rig.create(path, SCENARIO, clock=clock, fake=fake)
            await rig.start()
            continue  # the tick is processed again
        if k == 1:
            rig.fake.set_price("AAAUSDT", D("111"))
        k += 1
    tables = dump_state(rig.store)
    await rig.close()
    return tables, rig.fake


def same_content(
    got: dict[str, list[tuple[object, ...]]], want: dict[str, list[tuple[object, ...]]]
) -> None:
    """Every row equal, in any order (a resumed engine learns an outcome at another step), the
    fills' surrogate ids left out."""
    assert got.keys() == want.keys()
    for table in want:
        skip = 1 if table == "fills" else 0
        mine = sorted((r[skip:] for r in got[table]), key=repr)
        theirs = sorted((r[skip:] for r in want[table]), key=repr)
        assert mine == theirs, table


def placed_once(fake: FakeExchange) -> None:
    ids = [
        dict(r.url.params)["newClientOrderId"]
        for r in fake.requests_to("/api/v3/order", "POST")
        if "newClientOrderId" in r.url.params
    ]
    assert len(ids) == len(set(ids)), ids


@pytest.fixture
async def baseline(tmp_path: Path) -> dict[str, list[tuple[object, ...]]]:
    tables, fake = await run_scenario(tmp_path / "baseline.sqlite")
    placed_once(fake)
    reasons = sorted(str(r[16]) for r in tables["positions"])  # exit_reason
    assert reasons == ["take_profit", "time", "time"]
    return tables


BBB_TIME = time_exit_id(buy_id("BBBUSDT", 0), at(4))


@pytest.mark.parametrize(
    ("point", "n", "queried"),
    [
        # AAA's entry filled (and bracketed) on the exchange, the outcome not recorded
        (Step.AFTER_BROKER_CALL, 1, buy_id("AAAUSDT", 0)),
        # CCC's entry, with AAA's take profit filled while nothing ran
        (Step.AFTER_BROKER_CALL, 3, buy_id("CCCUSDT", 2)),
        # BBB's time exit: the intent written, the bracket still live: the sell is finished
        (Step.BEFORE_BROKER_CALL, 4, BBB_TIME),
        (Step.AFTER_BROKER_CALL, 4, BBB_TIME),  # BBB's time exit sold, not recorded
        (Step.AFTER_BROKER, 2, None),
        (Step.BEFORE_TICK_MARKER, 1, None),
    ],
)
async def test_a_crash_and_restart_ends_like_an_uninterrupted_run(
    tmp_path: Path,
    baseline: dict[str, list[tuple[object, ...]]],
    point: str,
    n: int,
    queried: str | None,
) -> None:
    tables, fake = await run_scenario(tmp_path / "s.sqlite", (point, n))
    placed_once(fake)
    subscribes = [m for m in fake.ws_requests if m.get("method", "").startswith("userData")]
    assert len(subscribes) == 2  # a second broker process: the crash happened
    if queried is not None:
        queries = [
            dict(r.url.params).get("origClientOrderId")
            for r in fake.requests_to("/api/v3/order", "GET")
        ]
        assert queried in queries
    same_content(tables, baseline)
