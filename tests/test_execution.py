"""`PaperExecution`: the paper broker behind the async execution interface."""

from __future__ import annotations

from datetime import timedelta

import pytest
from broker_fakes import T0, candles, flat

from crypto_moonshot.broker.base import Costs, ExitPlan, Order, Side, child_order_id
from crypto_moonshot.broker.execution import (
    CancelStatus,
    EntryFilled,
    ExecutionBroker,
    OrderState,
    PaperExecution,
    SubmitStatus,
    WorkingOrder,
)
from crypto_moonshot.broker.paper import BRACKET_ROLE, PaperSnapshot
from crypto_moonshot.models import Interval

pytestmark = pytest.mark.anyio

M5 = Interval.M5
PLAN = ExitPlan(take_profit=0.5, max_hold=timedelta(hours=1))


def buy(symbol: str = "AAAUSDT", k: int = 1, stake: float = 100.0, oid: str = "b1") -> Order:
    return Order(symbol, Side.BUY, T0 + k * M5.length, quote_amount=stake, exit_plan=PLAN, id=oid)


async def test_submit_accepts_rejects_and_events_advance() -> None:
    broker: ExecutionBroker = PaperExecution.create(150.0, Costs())
    assert (broker.needs_candles, broker.manages_time_exits, broker.sell_cancels_bracket) == (
        True,
        True,
        True,
    )
    accepted = await broker.submit(buy(), interval=M5)
    assert (accepted.status, accepted.events) == (SubmitStatus.ACCEPTED, ())
    again = await broker.submit(buy(), interval=M5)
    assert (again.status, again.reason) == (SubmitStatus.REJECTED, "duplicate_id")
    poor = await broker.submit(buy("BBBUSDT", oid="b2"), interval=M5)
    assert (poor.status, poor.reason) == (SubmitStatus.REJECTED, "cash")
    assert broker.candle_needs(M5) == {"AAAUSDT": T0 + M5.length}
    assert broker.candle_needs(Interval.H1) == {}
    broker.add_candles("AAAUSDT", candles(flat(3)))
    (event,) = await broker.events(T0 + 2 * M5.length, interval=M5)
    assert isinstance(event, EntryFilled)
    assert event.position.bracket_id == child_order_id("b1", BRACKET_ROLE)
    assert broker.candle_needs(M5) == {"AAAUSDT": T0 + M5.length}  # the position's candles


async def test_cancel_and_query_report_what_the_paper_broker_holds() -> None:
    broker = PaperExecution.create(1000.0)
    order = buy()
    assert (await broker.query(order)).state is OrderState.NOT_FOUND
    await broker.submit(order, interval=M5)
    assert (await broker.query(order)).state is OrderState.OPEN
    assert (await broker.cancel(order)).status is CancelStatus.CANCELLED
    assert broker.paper.cash == 1000.0  # the stake is released
    assert (await broker.cancel(order)).status is CancelStatus.NOT_OPEN
    assert (await broker.query(order)).state is OrderState.NOT_FOUND
    report = await broker.reconcile([WorkingOrder(order, M5, "s")], [], now=T0)
    assert report.clean
    assert report.time == T0


async def test_restore_rebuilds_the_paper_broker_in_place() -> None:
    broker = PaperExecution.create(1000.0, Costs(fee=0.002))
    pending = broker.paper.snapshot().orders
    assert pending == ()
    other = PaperExecution.create(500.0)
    await other.submit(buy(), interval=M5)
    broker.restore(other.paper.snapshot())
    assert broker.paper.cash == 400.0
    assert broker.paper.costs == Costs(fee=0.002)  # its own costs are kept
    assert (await broker.query(buy())).state is OrderState.OPEN
    broker.restore(PaperSnapshot(1000.0))
    assert broker.paper.pending == []
