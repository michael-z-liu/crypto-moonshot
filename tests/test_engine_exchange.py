"""The engine on an exchange-style broker: asynchronous fills, rejections, unknown outcomes,
engine-run time exits, flatten and exits through the bracket cancel, reconciliation, and crash
recovery between an order's intent and its outcome."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from engine_fakes import FakeClock, FakeFeed, Row, Scheduled, dump_state, flat_bars
from execution_fakes import FakeExchangeBroker

from crypto_moonshot.broker.base import ExitPlan, Order, Side, child_order_id, client_order_id
from crypto_moonshot.live.config import RunConfig, config_from_dict
from crypto_moonshot.live.engine import Engine, Step, build_specs, time_exit_id
from crypto_moonshot.live.health import HealthMonitor
from crypto_moonshot.live.state import (
    EventLevel,
    OrderRecord,
    OrderStatus,
    PositionRecord,
    PositionStatus,
    SignalOutcome,
    StateStore,
)
from crypto_moonshot.models import Interval

pytestmark = pytest.mark.anyio

H1 = Interval.H1
HOUR = timedelta(hours=1)
SECOND = timedelta(seconds=1)
T0 = datetime(2024, 3, 1, tzinfo=UTC)
SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")
PLAN = ExitPlan(take_profit=0.1, stop=0.1, max_hold=3 * HOUR)


def at(k: int) -> datetime:
    return T0 + k * HOUR


def buy_id(symbol: str, k: int, name: str = "s") -> str:
    return client_order_id(name, symbol, at(k), Side.BUY)


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


class Rig:
    """An engine on a `FakeFeed`, a `FakeClock` and a `FakeExchangeBroker`, run as a task."""

    def __init__(
        self,
        path: Path,
        schedule: Mapping[datetime, Sequence[Row]],
        *,
        plan: ExitPlan | None = PLAN,
        fake: FakeExchangeBroker | None = None,
        now: datetime | None = None,
        hook: Callable[[str, Any], None] | None = None,
        **overrides: Any,
    ) -> None:
        strategy = {
            "name": "s",
            "source": "tests:not_loaded",
            "symbols": list(SYMBOLS),
            "capital": 1000.0,
            "sizing": {"stake": 100.0},
            **overrides,
        }
        self.config: RunConfig = config_from_dict(
            {
                "state": str(path),
                "account": {"initial_cash": 10_000.0},
                "engine": {"signal_deadline": "1.2s", "entry_grace": "30m"},
                "strategies": [strategy],
            }
        )
        self.clock = FakeClock(now or at(0))
        self.fake = fake or FakeExchangeBroker(self.clock, dict.fromkeys(SYMBOLS, 100.0))
        self.fake.clock = self.clock
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
            broker=self.fake,
        )
        self.task: asyncio.Task[None] | None = None

    def _hook(self, step: str, tick: Any) -> None:
        if self.crash is not None:
            self.crash(step, tick)
        if step == Step.TICK_DONE:
            self.done.append(tick.open_time)

    async def start(self) -> None:
        self.task = asyncio.create_task(self.engine.run(self.feed))
        await self._until(lambda: self.engine._started)

    async def _until(self, check: Callable[[], bool]) -> None:
        deadline = asyncio.get_running_loop().time() + 5.0
        while not check():
            task = self.task
            if task is not None and task.done():
                task.result()  # raises the engine's exception
                msg = "the engine stopped"
                raise AssertionError(msg)
            if asyncio.get_running_loop().time() > deadline:
                msg = "condition not reached"
                raise TimeoutError(msg)
            await asyncio.sleep(0.005)

    async def tick(self, k: int) -> None:
        """Candle k closes (one second before the clock) and is fully processed."""
        self.clock.now = at(k + 1) + SECOND
        await self.feed.emit(k)
        await self._until(lambda: at(k) in self.done)

    async def recovered(self) -> None:
        """Wait until the start-up recovery (queries, reconcile) is done."""
        await self._until(lambda: ("reconcile", self.clock().isoformat()) in self.fake.calls)
        await asyncio.sleep(0.01)

    async def close(self) -> None:
        task = self.task
        if task is not None and not task.done():
            await self.engine.stop()
            await task
        self.store.close()

    def order(self, order_id: str) -> OrderRecord:
        order = self.store.order(order_id)
        assert order is not None, order_id
        return order

    def position(self, symbol: str, k: int) -> PositionRecord:
        position = self.store.position_for_order(buy_id(symbol, k))
        assert position is not None
        return position

    def signal(self, k: int, symbol: str, action: str = "enter") -> tuple[Any, Any]:
        batch = self.store.batch_for("s", H1, at(k))
        assert batch is not None
        (signal,) = [s for s in batch.signals if s.symbol == symbol and s.action == action]
        return signal.outcome, signal.reason

    def kinds(self) -> list[str]:
        return [e.kind for e in self.store.events_since()]

    def book_matches(self) -> bool:
        book = self.engine.books["s"]
        return book == self.store.derive_book("s", 1000.0, self.clock())


# -- entries --------------------------------------------------------------------------------


@pytest.mark.parametrize("immediate", [False, True])
async def test_an_accepted_entry_fills_later_and_its_bracket_exits(
    tmp_path: Path, immediate: bool
) -> None:
    rig = Rig(tmp_path / "s.sqlite", {at(0): [("AAAUSDT", 0.0)]})
    rig.fake.immediate = immediate
    await rig.start()
    await rig.tick(0)
    assert rig.signal(0, "AAAUSDT") == (SignalOutcome.PLACED, None)
    assert "AAAUSDT" in rig.fake.positions  # filled at the exchange at once
    position = rig.position("AAAUSDT", 0)
    if immediate:  # the fill came with the acknowledgement
        assert position.status is PositionStatus.OPEN
    else:  # reported by the next `events`: pending, stake reserved
        assert position.status is PositionStatus.PENDING
        assert rig.order(buy_id("AAAUSDT", 0)).status is OrderStatus.PENDING
        assert rig.store.fills() == []
    assert rig.engine.books["s"].cash == pytest.approx(900.0)
    assert rig.book_matches()
    await rig.tick(1)
    position = rig.position("AAAUSDT", 0)
    assert position.status is PositionStatus.OPEN
    assert (position.entry_time, position.entry_price) == (at(1) + SECOND, 100.0)
    bracket_id = child_order_id(buy_id("AAAUSDT", 0), "bracket")
    assert position.bracket_id == bracket_id
    assert rig.order(bracket_id).status is OrderStatus.PENDING
    assert rig.order(buy_id("AAAUSDT", 0)).status is OrderStatus.FILLED
    assert len(rig.store.fills()) == 1
    assert rig.engine.books["s"].open_cost == pytest.approx(100.0)
    assert rig.book_matches()
    rig.clock.now = at(2) + 30 * timedelta(minutes=1)
    rig.fake.move("AAAUSDT", 111.0)  # through the take profit at 110
    await rig.tick(2)
    closed = rig.position("AAAUSDT", 0)
    assert (closed.status, closed.exit_reason) == (PositionStatus.CLOSED, "take_profit")
    assert closed.exit_price == pytest.approx(110.0)
    assert closed.exit_time == at(2) + 30 * timedelta(minutes=1)
    assert rig.order(bracket_id).status is OrderStatus.FILLED
    assert rig.engine.books["s"].cash > 1000.0
    assert rig.book_matches()
    await rig.close()


async def test_a_rejection_at_submit_frees_the_slot_for_the_next_entry(tmp_path: Path) -> None:
    schedule: dict[datetime, list[Row]] = {at(0): [("AAAUSDT", 0.0), ("BBBUSDT", 1.0)]}
    # Without the rejection, the second entry finds the one slot taken.
    control = Rig(tmp_path / "c.sqlite", schedule, risk={"max_positions": 1})
    await control.start()
    await control.tick(0)
    assert control.signal(0, "BBBUSDT") == (SignalOutcome.REJECTED, "slots_full")
    await control.close()

    rig = Rig(tmp_path / "s.sqlite", schedule, risk={"max_positions": 1})
    rig.fake.reject_next["AAAUSDT"] = "min_notional"
    await rig.start()
    await rig.tick(0)
    assert rig.signal(0, "AAAUSDT") == (SignalOutcome.REJECTED, "broker_min_notional")
    order = rig.order(buy_id("AAAUSDT", 0))
    assert (order.status, order.reason) == (OrderStatus.REJECTED, "min_notional")
    aaa = rig.position("AAAUSDT", 0)
    assert (aaa.status, aaa.exit_reason) == (PositionStatus.CANCELLED, "broker_min_notional")
    assert rig.signal(0, "BBBUSDT") == (SignalOutcome.PLACED, None)
    assert rig.position("BBBUSDT", 0).status is PositionStatus.PENDING
    refused = [e for e in rig.store.events_since() if e.kind == "entries_refused"]
    assert refused[-1].data is not None
    assert refused[-1].data["reasons"] == {"broker_min_notional": 1}
    assert rig.engine.books["s"].open_positions == 1
    assert rig.book_matches()
    await rig.close()


async def test_a_rejection_after_acceptance_cancels_the_position(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "s.sqlite", {at(0): [("AAAUSDT", 0.0)]})
    rig.fake.reject_later["AAAUSDT"] = "insufficient_balance"
    await rig.start()
    await rig.tick(0)
    assert rig.order(buy_id("AAAUSDT", 0)).status is OrderStatus.PENDING
    assert rig.engine.books["s"].cash == pytest.approx(900.0)
    await rig.tick(1)
    order = rig.order(buy_id("AAAUSDT", 0))
    assert (order.status, order.reason) == (OrderStatus.REJECTED, "insufficient_balance")
    position = rig.position("AAAUSDT", 0)
    assert (position.status, position.exit_reason) == (
        PositionStatus.CANCELLED,
        "broker_insufficient_balance",
    )
    assert rig.signal(0, "AAAUSDT") == (SignalOutcome.REJECTED, "broker_insufficient_balance")
    assert rig.engine.books["s"].cash == pytest.approx(1000.0)
    assert rig.book_matches()
    assert "order_rejected" in rig.kinds()
    await rig.close()


# -- unknown outcomes -----------------------------------------------------------------------


@pytest.mark.parametrize("truth", ["filled", "open", "never_placed"])
async def test_an_unknown_outcome_is_resolved_by_query_at_the_next_tick(
    tmp_path: Path, truth: str
) -> None:
    schedule: dict[datetime, list[Row]] = {at(0): [("AAAUSDT", 0.0)], at(1): [("AAAUSDT", 0.0)]}
    rig = Rig(tmp_path / "s.sqlite", schedule)
    rig.fake.unknown_next["AAAUSDT"] = truth != "never_placed"
    if truth == "open":
        rig.fake.hold.add("AAAUSDT")
    await rig.start()
    await rig.tick(0)
    first = buy_id("AAAUSDT", 0)
    assert rig.order(first).status is OrderStatus.UNKNOWN
    assert rig.position("AAAUSDT", 0).status is PositionStatus.PENDING  # stake reserved
    assert rig.engine.books["s"].cash == pytest.approx(900.0)
    assert "order_unknown" in rig.kinds()
    await rig.tick(1)
    assert rig.fake.calls.count(("query", first)) == 1
    assert rig.fake.submits(first) == 1  # never placed again
    if truth == "filled":
        assert rig.order(first).status is OrderStatus.FILLED
        assert rig.position("AAAUSDT", 0).status is PositionStatus.OPEN
        assert rig.signal(1, "AAAUSDT") == (SignalOutcome.REJECTED, "symbol_held")
    elif truth == "open":
        assert rig.order(first).status is OrderStatus.PENDING
        assert rig.position("AAAUSDT", 0).status is PositionStatus.PENDING
        rig.fake.release("AAAUSDT")
        await rig.tick(2)
        assert rig.position("AAAUSDT", 0).status is PositionStatus.OPEN
    else:
        order = rig.order(first)
        assert (order.status, order.reason) == (OrderStatus.REJECTED, "not_placed")
        cancelled = rig.position("AAAUSDT", 0)
        assert (cancelled.status, cancelled.exit_reason) == (
            PositionStatus.CANCELLED,
            "broker_not_placed",
        )
        assert rig.signal(0, "AAAUSDT") == (SignalOutcome.REJECTED, "broker_not_placed")
        assert "order_not_placed" in rig.kinds()
        # Only a later tick's signal enters the symbol again, with its own order.
        assert rig.signal(1, "AAAUSDT") == (SignalOutcome.PLACED, None)
        assert rig.position("AAAUSDT", 1).status is PositionStatus.PENDING
        assert first not in rig.fake.orders
    assert rig.book_matches()
    await rig.close()


async def test_an_order_the_broker_still_cannot_find_keeps_its_symbol_held(
    tmp_path: Path,
) -> None:
    schedule: dict[datetime, list[Row]] = {at(0): [("AAAUSDT", 0.0)], at(1): [("AAAUSDT", 0.0)]}
    rig = Rig(tmp_path / "s.sqlite", schedule)
    rig.fake.unknown_next["AAAUSDT"] = True
    rig.fake.hold.add("AAAUSDT")  # placed, open, nothing reported
    rig.fake.query_unknown.add(buy_id("AAAUSDT", 0))
    await rig.start()
    await rig.tick(0)
    await rig.tick(1)
    assert rig.order(buy_id("AAAUSDT", 0)).status is OrderStatus.UNKNOWN
    assert rig.signal(1, "AAAUSDT") == (SignalOutcome.REJECTED, "symbol_held")
    rig.fake.query_unknown.clear()
    await rig.tick(2)
    assert rig.order(buy_id("AAAUSDT", 0)).status is OrderStatus.PENDING
    rig.fake.release("AAAUSDT")
    await rig.tick(3)
    assert rig.order(buy_id("AAAUSDT", 0)).status is OrderStatus.FILLED
    assert rig.position("AAAUSDT", 0).status is PositionStatus.OPEN
    await rig.close()


# -- time exits, exits and flatten ----------------------------------------------------------


async def test_the_engine_runs_the_time_exit_cancelling_the_bracket_first(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "s.sqlite", {at(0): [("AAAUSDT", 0.0)]})
    await rig.start()
    for k in range(4):  # filled at 01:00:01, held for 3h: not yet at the tick of 04:00
        await rig.tick(k)
    entry = buy_id("AAAUSDT", 0)
    bracket_id = child_order_id(entry, "bracket")
    assert rig.position("AAAUSDT", 0).status is PositionStatus.OPEN
    assert rig.order(bracket_id).status is OrderStatus.PENDING
    await rig.tick(4)  # the first tick at or after the deadline
    sell_id = time_exit_id(entry, at(5))
    bracket = rig.order(bracket_id)
    assert (bracket.status, bracket.reason) == (OrderStatus.CANCELLED, "position_sold")
    sell = rig.order(sell_id)
    assert (sell.status, sell.side, sell.time) == (OrderStatus.PENDING, Side.SELL, at(5))
    calls = rig.fake.calls
    assert calls.index(("cancel", bracket_id)) < calls.index(("submit", sell_id))
    await rig.tick(5)
    closed = rig.position("AAAUSDT", 0)
    assert (closed.status, closed.exit_reason, closed.exit_time) == (
        PositionStatus.CLOSED,
        "time",
        at(5) + SECOND,
    )
    assert rig.order(sell_id).status is OrderStatus.FILLED
    assert rig.fake.positions == {}
    assert rig.book_matches()
    await rig.tick(6)  # nothing more to sell
    assert rig.fake.calls.count(("submit", sell_id)) == 1
    await rig.close()


async def test_a_bracket_that_fills_during_the_time_exit_cancel_is_not_sold_twice(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "s.sqlite", {at(0): [("AAAUSDT", 0.0)]})
    await rig.start()
    for k in range(4):
        await rig.tick(k)
    rig.fake.race_cancel.add("AAAUSDT")
    rig.fake.prices["AAAUSDT"] = 95.0
    await rig.tick(4)
    entry = buy_id("AAAUSDT", 0)
    closed = rig.position("AAAUSDT", 0)
    assert (closed.status, closed.exit_reason, closed.exit_price) == (
        PositionStatus.CLOSED,
        "stop",
        95.0,
    )
    sell = rig.order(time_exit_id(entry, at(5)))
    assert (sell.status, sell.reason) == (OrderStatus.CANCELLED, "position_closed")
    assert rig.fake.submits(sell.id) == 0
    assert rig.order(child_order_id(entry, "bracket")).status is OrderStatus.FILLED
    assert len(rig.store.fills()) == 2  # the entry and the bracket's exit only
    assert rig.book_matches()
    await rig.tick(5)
    assert len(rig.store.fills()) == 2
    await rig.close()


async def test_an_exit_signal_cancels_the_bracket_then_sells(tmp_path: Path) -> None:
    schedule: dict[datetime, list[Row]] = {
        at(0): [("AAAUSDT", 0.0)],
        at(2): [("AAAUSDT", 0.0, "exit")],
    }
    rig = Rig(tmp_path / "s.sqlite", schedule)
    await rig.start()
    for k in range(3):
        await rig.tick(k)
    sell_id = client_order_id("s", "AAAUSDT", at(2), Side.SELL)
    bracket_id = child_order_id(buy_id("AAAUSDT", 0), "bracket")
    assert rig.signal(2, "AAAUSDT", "exit") == (SignalOutcome.PLACED, None)
    assert rig.order(bracket_id).status is OrderStatus.CANCELLED
    calls = rig.fake.calls
    assert calls.index(("cancel", bracket_id)) < calls.index(("submit", sell_id))
    await rig.tick(3)
    closed = rig.position("AAAUSDT", 0)
    assert (closed.status, closed.exit_reason) == (PositionStatus.CLOSED, "signal")
    await rig.close()


async def test_an_exit_whose_bracket_cancel_gets_no_answer_is_not_sold(tmp_path: Path) -> None:
    schedule: dict[datetime, list[Row]] = {
        at(0): [("AAAUSDT", 0.0)],
        at(2): [("AAAUSDT", 0.0, "exit")],
    }
    rig = Rig(tmp_path / "s.sqlite", schedule)
    await rig.start()
    for k in range(2):
        await rig.tick(k)
    rig.fake.cancel_unknown.add("AAAUSDT")
    await rig.tick(2)
    sell_id = client_order_id("s", "AAAUSDT", at(2), Side.SELL)
    sell = rig.order(sell_id)
    assert (sell.status, sell.reason) == (OrderStatus.REJECTED, "bracket_open")
    assert rig.signal(2, "AAAUSDT", "exit") == (SignalOutcome.REJECTED, "broker_bracket_open")
    assert rig.fake.submits(sell_id) == 0
    assert rig.position("AAAUSDT", 0).status is PositionStatus.OPEN
    assert "bracket_cancel_failed" in rig.kinds()
    await rig.close()


async def test_flatten_cancels_brackets_and_pending_entries_then_sells(tmp_path: Path) -> None:
    schedule: dict[datetime, list[Row]] = {
        at(0): [("AAAUSDT", 0.0), ("BBBUSDT", 0.0)],
        at(1): [("CCCUSDT", 0.0)],
    }
    rig = Rig(tmp_path / "s.sqlite", schedule)
    await rig.start()
    await rig.tick(0)
    rig.fake.hold.add("CCCUSDT")  # CCC's entry stays open at the exchange
    await rig.tick(1)
    assert rig.position("CCCUSDT", 1).status is PositionStatus.PENDING
    rig.engine.flatten()
    await rig.tick(2)
    controls = rig.store.controls()
    assert (controls.paused, controls.flatten_requested) == (True, False)
    ccc = rig.position("CCCUSDT", 1)
    assert (ccc.status, ccc.exit_reason) == (PositionStatus.CANCELLED, "flatten")
    assert rig.order(buy_id("CCCUSDT", 1)).status is OrderStatus.CANCELLED
    for symbol in ("AAAUSDT", "BBBUSDT"):
        base = client_order_id("s", symbol, at(2), Side.SELL)
        sell_id = child_order_id(base, "flatten")
        bracket_id = child_order_id(buy_id(symbol, 0), "bracket")
        assert rig.order(bracket_id).status is OrderStatus.CANCELLED
        assert rig.order(sell_id).status is OrderStatus.PENDING
        calls = rig.fake.calls
        assert calls.index(("cancel", bracket_id)) < calls.index(("submit", sell_id))
    await rig.tick(3)
    closed = {p.symbol: p.exit_reason for p in rig.store.closed_positions()}
    assert closed == {"AAAUSDT": "flatten", "BBBUSDT": "flatten"}
    assert rig.store.live_positions() == []
    assert rig.fake.positions == {}
    assert rig.book_matches()
    await rig.close()


# -- reconciliation and alerts --------------------------------------------------------------


async def test_reconciliation_at_start_reports_orphans_and_pauses_on_a_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.sqlite"
    first = Rig(path, {at(0): [("AAAUSDT", 0.0)]})
    await first.start()
    await first.tick(0)
    await first.tick(1)
    assert first.position("AAAUSDT", 0).status is PositionStatus.OPEN
    await first.close()

    fake = first.fake
    fake.restart()
    fake.balances["AAA"] = 0.0  # the AAA bought is gone from the account
    orphan = Order("DDDUSDT", Side.SELL, at(1), quantity=1.0, id="cms-not-ours-locally")
    fake.add_orphan(orphan)
    again = Rig(path, {at(2): [("BBBUSDT", 0.0)]}, fake=fake, now=at(2) + SECOND)
    await again.start()
    await again.recovered()
    events = {e.kind: e for e in again.store.events_since()}
    assert events["orphan_order"].level is EventLevel.WARNING
    assert events["orphan_order"].data is not None
    assert events["orphan_order"].data["order_id"] == orphan.id
    assert events["balance_mismatch"].level is EventLevel.ERROR
    assert again.store.controls().paused
    report = again.engine.reconciliation
    assert report is not None
    assert [o.order_id for o in report.orphans] == [orphan.id]
    assert [m.asset for m in report.mismatches] == ["AAA"]
    monitor = HealthMonitor(again.engine, again.feed, again.config.health, disk_free=_plenty)
    found = {f.key: f.level for f in monitor.check()}
    assert found["orphan_order"] is EventLevel.WARNING
    assert found["balance_mismatch"] is EventLevel.ERROR
    assert {a.key for a in again.store.alerts()} >= {"orphan_order", "balance_mismatch"}
    await again.tick(2)
    assert again.signal(2, "BBBUSDT") == (SignalOutcome.REJECTED, "paused")
    await again.close()


async def test_a_position_without_its_bracket_raises_an_alert_until_it_is_placed(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "s.sqlite", {at(0): [("AAAUSDT", 0.0)]})
    rig.fake.fail_bracket.add("AAAUSDT")
    await rig.start()
    await rig.tick(0)
    await rig.tick(1)
    position = rig.position("AAAUSDT", 0)
    assert (position.status, position.bracket_id) == (PositionStatus.OPEN, None)
    assert "bracket_failed" in rig.kinds()
    assert [p.symbol for p in rig.engine.unprotected_positions()] == ["AAAUSDT"]
    monitor = HealthMonitor(rig.engine, rig.feed, rig.config.health, disk_free=_plenty)
    found = {f.key: f.level for f in monitor.check()}
    assert found["unprotected_position:AAAUSDT"] is EventLevel.ERROR
    rig.fake.place_brackets()
    await rig.tick(2)
    bracket_id = child_order_id(buy_id("AAAUSDT", 0), "bracket")
    assert rig.position("AAAUSDT", 0).bracket_id == bracket_id
    assert rig.order(bracket_id).status is OrderStatus.PENDING
    assert rig.engine.unprotected_positions() == []
    monitor.check()
    assert "unprotected_position:AAAUSDT" not in {a.key for a in rig.store.alerts()}
    await rig.close()


def _plenty(path: Path) -> int:
    return 10**12


# -- crash recovery -------------------------------------------------------------------------

SCENARIO: dict[datetime, list[Row]] = {
    at(0): [("AAAUSDT", 0.0), ("BBBUSDT", 1.0)],
    at(2): [("CCCUSDT", 0.0)],
}
TICKS = 9


async def run_scenario(
    path: Path, crash: tuple[str, int] | None = None
) -> tuple[dict[str, list[tuple[object, ...]]], FakeExchangeBroker]:
    """Ticks 0..8 of `SCENARIO`: entries at 0 and 2, time exits at 4 and 6. With `crash`, the
    engine fails there and a new one (over the same exchange) resumes at the same clock time."""
    hook = None if crash is None else crash_at(*crash)
    rig = Rig(path, SCENARIO, hook=hook)
    await rig.start()
    k = 0
    while k < TICKS:
        try:
            await rig.tick(k)
        except Crash:
            assert crash is not None
            fake = rig.fake
            now = rig.clock()
            rig.store.close()
            fake.restart()
            rig = Rig(path, SCENARIO, fake=fake, now=now)
            await rig.start()
            await rig.recovered()
            continue  # the tick is processed again
        k += 1
    tables = dump_state(rig.store)
    await rig.close()
    return tables, rig.fake


def same_content(
    got: dict[str, list[tuple[object, ...]]], want: dict[str, list[tuple[object, ...]]]
) -> None:
    """Every row equal; rows may be inserted in another order (a resumed engine learns an
    outcome at another step), so a fill's surrogate id is left out."""
    assert got.keys() == want.keys()
    for table in want:
        skip = 1 if table == "fills" else 0
        mine = sorted((r[skip:] for r in got[table]), key=repr)
        theirs = sorted((r[skip:] for r in want[table]), key=repr)
        assert mine == theirs, table


def assert_never_placed_twice(fake: FakeExchangeBroker) -> None:
    submitted = [order_id for call, order_id in fake.calls if call == "submit"]
    assert len(submitted) == len(set(submitted))


@pytest.fixture
async def baseline(tmp_path: Path) -> dict[str, list[tuple[object, ...]]]:
    tables, fake = await run_scenario(tmp_path / "baseline.sqlite")
    assert_never_placed_twice(fake)
    reasons = sorted(str(r[16]) for r in tables["positions"])  # exit_reason
    assert reasons == ["time", "time", "time"]
    return tables


async def test_a_crash_after_the_intent_queries_and_never_places_the_order_twice(
    tmp_path: Path, baseline: dict[str, list[tuple[object, ...]]]
) -> None:
    path = tmp_path / "s.sqlite"
    tables, fake = await run_scenario(path, (Step.BEFORE_BROKER_CALL, 1))
    first = buy_id("AAAUSDT", 0)
    assert fake.submits(first) == 0
    assert ("query", first) in fake.calls
    assert_never_placed_twice(fake)
    with StateStore.open(path) as store:
        order = store.order(first)
        assert order is not None
        assert (order.status, order.reason) == (OrderStatus.REJECTED, "not_placed")
        position = store.position_for_order(first)
        assert position is not None
        assert (position.status, position.exit_reason) == (
            PositionStatus.CANCELLED,
            "broker_not_placed",
        )
        closed = sorted(p.symbol for p in store.closed_positions())
        assert closed == ["BBBUSDT", "CCCUSDT"]  # the rest of the run is unchanged
    # BBB and CCC trade exactly as in the uninterrupted run.
    fills = {r[1] for r in tables["fills"]}
    assert fills == {r[1] for r in baseline["fills"]} - {first, time_exit_id(first, at(5))}


AAA_TIME = time_exit_id(buy_id("AAAUSDT", 0), at(5))
BBB_TIME = time_exit_id(buy_id("BBBUSDT", 0), at(5))


@pytest.mark.parametrize(
    ("point", "n", "queried"),
    [
        # AAA's entry placed and filled, the outcome not recorded
        (Step.AFTER_BROKER_CALL, 1, buy_id("AAAUSDT", 0)),
        (Step.AFTER_BROKER_CALL, 3, buy_id("CCCUSDT", 2)),  # CCC's entry
        # AAA's time exit: the intent written, the bracket still live: the sell is finished
        (Step.BEFORE_BROKER_CALL, 4, AAA_TIME),
        (Step.AFTER_BROKER_CALL, 4, AAA_TIME),  # AAA's time exit placed, not recorded
        (Step.BEFORE_BROKER_CALL, 5, BBB_TIME),  # BBB's, with AAA's fill not reported yet
        (Step.AFTER_BROKER, 5, None),
        (Step.BEFORE_TICK_MARKER, 1, None),
    ],
)
async def test_a_crash_between_intent_and_outcome_ends_like_an_uninterrupted_run(
    tmp_path: Path,
    baseline: dict[str, list[tuple[object, ...]]],
    point: str,
    n: int,
    queried: str | None,
) -> None:
    tables, fake = await run_scenario(tmp_path / "s.sqlite", (point, n))
    assert_never_placed_twice(fake)
    queries = [order_id for call, order_id in fake.calls if call == "query"]
    assert queries == ([] if queried is None else [queried])
    same_content(tables, baseline)
