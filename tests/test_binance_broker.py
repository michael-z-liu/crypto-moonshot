"""`BinanceBroker` against the fake exchange: entries, brackets, exits, partial fills, fees,
rejections, unknown outcomes, the user data stream and its gaps, reconciliation and halts."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal

import pytest
from binance_fakes import T0, BrokerRig, Clock, make_rig, until
from fake_exchange import API_KEY, API_SECRET, FakeExchange

from crypto_moonshot.broker.base import ExitPlan, ExitReason, Order, OrderType, Side
from crypto_moonshot.broker.binance import (
    BinanceBroker,
    ExitPlanError,
    bracket_id,
    check_exit_plan,
    exit_plan_problems,
    leg_ids,
)
from crypto_moonshot.broker.execution import (
    BracketFailed,
    BracketPlaced,
    BrokerEvent,
    BrokerHalted,
    CancelReason,
    CancelStatus,
    EntryFilled,
    ExitFilled,
    OrderCancelled,
    OrderState,
    Position,
    Reconciliation,
    SubmitStatus,
    WorkingOrder,
)
from crypto_moonshot.models import Interval

pytestmark = pytest.mark.anyio

D = Decimal
H1 = Interval.H1
SYMBOL = "AAAUSDT"
PLAN = ExitPlan(take_profit=0.1, stop=0.05)


def buy(
    symbol: str = SYMBOL, amount: float = 1000.0, plan: ExitPlan | None = PLAN, n: int = 1
) -> Order:
    return Order(symbol, Side.BUY, T0, quote_amount=amount, exit_plan=plan, id=f"cmb-test{n:028d}")


def sell(symbol: str = SYMBOL, n: int = 1) -> Order:
    return Order(symbol, Side.SELL, T0, id=f"cms-test{n:028d}")


def bracket_order(position: Position) -> Order:
    order = position.bracket()
    assert order is not None
    return order


def up(value: Decimal, tick: str = "0.01") -> Decimal:
    return value.quantize(D(tick), ROUND_UP)


def down(value: Decimal, tick: str = "0.01") -> Decimal:
    return value.quantize(D(tick), ROUND_DOWN)


@pytest.fixture
async def rig() -> AsyncIterator[BrokerRig]:
    rig = await make_rig()
    yield rig
    await rig.close()


async def entered(rig: BrokerRig, order: Order | None = None) -> EntryFilled:
    order = order or buy()
    result = await rig.broker.submit(order, interval=H1)
    assert result.status is SubmitStatus.ACCEPTED, result
    (event, *rest) = result.events
    assert isinstance(event, EntryFilled), result.events
    assert not [e for e in rest if isinstance(e, BracketFailed)], rest
    return event


async def drain(rig: BrokerRig) -> list[BrokerEvent]:
    await rig.delivered()
    return await rig.broker.events(rig.clock())


# -- entries and brackets ---------------------------------------------------------------------


async def test_an_entry_fills_and_its_oco_is_placed_with_rounded_levels(rig: BrokerRig) -> None:
    rig.fake.split = 2
    event = await entered(rig)
    # two fills: 5.000 at 100.00, then 4.999 at 100.01 (the budget's second half)
    quote = D("5.000") * D("100.00") + D("4.999") * D("100.01")
    executed = D("9.999")
    avg = quote / executed
    commission = D("0.005") + D("0.004999")
    assert event.fill.price == pytest.approx(float(avg))
    assert event.fill.quantity == pytest.approx(float(executed))
    position = event.position
    assert position.entry_ref == pytest.approx(float(avg))
    assert position.cost == pytest.approx(float(quote))
    assert position.quantity == pytest.approx(9.989)  # 9.989001 net, rounded down to the step
    assert rig.broker.dust[SYMBOL] == executed - commission - D("9.989")
    assert position.entry_time == T0  # the order's time, as in paper mode
    order_id = buy().id
    assert position.bracket_id == bracket_id(order_id)
    listed = rig.fake.lists[bracket_id(order_id)]
    above, below = listed.legs
    assert (above.client_id, below.client_id) == leg_ids(bracket_id(order_id))
    assert above.type == "LIMIT_MAKER"
    assert above.price == up(avg * D("1.1"))
    assert below.type == "STOP_LOSS"
    assert below.stop_price == down(avg * D("0.95"))
    assert above.quantity == below.quantity == D("9.989")
    assert rig.hosts() <= {"testnet.invalid", "127.0.0.1"}


async def test_the_take_profit_leg_fills(rig: BrokerRig) -> None:
    event = await entered(rig)
    tp = rig.fake.lists[bracket_id(buy().id)].legs[0].price
    assert await drain(rig) == []
    rig.fake.set_price(SYMBOL, tp)
    (exit_,) = await drain(rig)
    assert isinstance(exit_, ExitFilled)
    assert exit_.fill.order_id == event.position.bracket_id
    assert exit_.fill.reason is ExitReason.TAKE_PROFIT
    assert exit_.fill.price == pytest.approx(float(tp))
    assert exit_.fill.quantity == pytest.approx(event.position.quantity)
    notional = D(str(event.position.quantity)) * tp
    assert exit_.fill.fee == pytest.approx(float(notional * D("0.001")), abs=1e-7)
    assert exit_.proceeds == pytest.approx(float(notional * D("0.999")), abs=1e-6)
    assert await drain(rig) == []


async def test_the_stop_leg_fills_at_market_with_slippage(rig: BrokerRig) -> None:
    event = await entered(rig)
    stop = rig.fake.lists[bracket_id(buy().id)].legs[1].stop_price
    assert stop is not None
    rig.fake.slippage = D("0.002")
    rig.fake.set_price(SYMBOL, D("94.00"))  # gaps through the stop
    (exit_,) = await drain(rig)
    assert isinstance(exit_, ExitFilled)
    assert exit_.fill.reason is ExitReason.STOP
    assert exit_.fill.order_id == event.position.bracket_id
    assert exit_.fill.price == pytest.approx(float(down(D("94.00") * D("0.998"))))
    assert exit_.fill.price < float(stop)
    assert rig.fake.balance("AAA")[1] == 0  # nothing locked any more


async def test_a_trailing_stop_follows_the_peak(rig: BrokerRig) -> None:
    plan = ExitPlan(take_profit=0.5, trailing=0.05)
    event = await entered(rig, buy(plan=plan))
    below = rig.fake.lists[bracket_id(buy().id)].legs[1]
    assert below.trailing_delta == 500
    assert below.stop_price is None
    rig.fake.walk(SYMBOL, 100, 120, 99, 118)  # peak 120: the trailing level is 114
    assert await drain(rig) == []
    rig.fake.set_price(SYMBOL, D("114.50"))
    assert await drain(rig) == []
    rig.fake.set_price(SYMBOL, D("113.90"))
    (exit_,) = await drain(rig)
    assert isinstance(exit_, ExitFilled)
    assert exit_.fill.reason is ExitReason.TRAILING_STOP
    assert exit_.fill.price == pytest.approx(113.90)
    assert exit_.fill.order_id == event.position.bracket_id


async def test_a_plan_with_both_stops_uses_the_trailing_one(rig: BrokerRig) -> None:
    await entered(rig, buy(plan=ExitPlan(take_profit=0.2, stop=0.1, trailing=0.05)))
    below = rig.fake.lists[bracket_id(buy().id)].legs[1]
    assert (below.trailing_delta, below.stop_price) == (500, None)


@pytest.mark.parametrize(
    ("plan", "kind", "reason"),
    [
        (ExitPlan(take_profit=0.1), "LIMIT_MAKER", ExitReason.TAKE_PROFIT),
        (ExitPlan(stop=0.05), "STOP_LOSS", ExitReason.STOP),
        (ExitPlan(trailing=0.05), "STOP_LOSS", ExitReason.TRAILING_STOP),
    ],
)
async def test_a_single_leg_plan_places_one_order(
    rig: BrokerRig, plan: ExitPlan, kind: str, reason: ExitReason
) -> None:
    event = await entered(rig, buy(plan=plan))
    bid = bracket_id(buy().id)
    assert event.position.bracket_id == bid
    assert rig.fake.lists == {}
    order = rig.fake.orders[bid]
    assert order.type == kind
    rig.fake.set_price(SYMBOL, D("111") if reason is ExitReason.TAKE_PROFIT else D("94"))
    (exit_,) = await drain(rig)
    assert isinstance(exit_, ExitFilled)
    assert (exit_.fill.order_id, exit_.fill.reason) == (bid, reason)


async def test_a_plan_without_exits_places_no_bracket(rig: BrokerRig) -> None:
    for plan in (None, ExitPlan(max_hold=timedelta(hours=3))):
        rig.fake.orders.clear()
        n = 1 if plan is None else 2
        order = buy(plan=plan, n=n, symbol="AAAUSDT" if n == 1 else "BBBUSDT")
        result = await rig.broker.submit(order, interval=H1)
        (event,) = result.events
        assert isinstance(event, EntryFilled)
        assert event.position.bracket_id is None
        assert list(rig.fake.orders) == [order.id]
    assert await drain(rig) == []


# -- levels crossed at placement and bracket failures -----------------------------------------


@pytest.mark.parametrize(
    ("plan", "price", "reason"),
    [
        (PLAN, "111", ExitReason.TAKE_PROFIT),  # OCO price relationship refused
        (PLAN, "94", ExitReason.STOP),
        (ExitPlan(take_profit=0.1), "111", ExitReason.TAKE_PROFIT),  # LIMIT_MAKER would take
        (ExitPlan(stop=0.05), "94", ExitReason.STOP),  # STOP_LOSS would trigger
    ],
)
async def test_a_level_crossed_at_placement_sells_at_market(
    rig: BrokerRig, plan: ExitPlan, price: str, reason: ExitReason
) -> None:
    rig.fake.hold_market.add(SYMBOL)
    order = buy(plan=plan)
    result = await rig.broker.submit(order, interval=H1)
    assert (result.status, result.events) == (SubmitStatus.ACCEPTED, ())
    rig.fake.release_market(SYMBOL)  # fills at 100
    rig.fake.set_price(SYMBOL, D(price))  # moves past a level before the bracket is placed
    entry, exit_ = await drain(rig)
    assert isinstance(entry, EntryFilled)
    assert isinstance(exit_, ExitFilled)
    bid = bracket_id(order.id)
    assert entry.position.bracket_id == bid
    assert (exit_.fill.order_id, exit_.fill.reason) == (bid, reason)
    assert exit_.fill.price == pytest.approx(float(price))
    assert rig.fake.orders[bid].type == "MARKET"
    assert rig.broker.positions == {}
    assert await drain(rig) == []


async def test_a_failed_bracket_is_retried_with_backoff(rig: BrokerRig) -> None:
    rig.fake.fail(
        "/api/v3/orderList/oco", code=-1013, msg="Filter failure: PERCENT_PRICE_BY_SIDE", times=2
    )
    order = buy()
    result = await rig.broker.submit(order, interval=H1)
    entry, failed = result.events
    assert isinstance(entry, EntryFilled)
    assert entry.position.bracket_id is None
    assert isinstance(failed, BracketFailed)
    assert (failed.entry_order_id, failed.reason) == (order.id, "filter_percent_price_by_side")
    assert await drain(rig) == []  # not before the backoff
    rig.clock.advance(5)
    assert await drain(rig) == []  # failed again: the backoff doubles
    assert len(rig.fake.requests_to("/api/v3/orderList/oco")) == 2
    rig.clock.advance(5)
    assert await drain(rig) == []
    rig.clock.advance(5)
    (placed,) = await drain(rig)
    assert isinstance(placed, BracketPlaced)
    assert placed.position.bracket_id == bracket_id(order.id)  # never reached the exchange
    assert bracket_id(order.id) in rig.fake.lists


async def test_a_bracket_whose_placement_got_no_answer_retries_under_a_new_id(
    rig: BrokerRig,
) -> None:
    rig.fake.timeout("/api/v3/orderList/oco")
    order = buy()
    result = await rig.broker.submit(order, interval=H1)
    _, failed = result.events
    assert isinstance(failed, BracketFailed)
    assert failed.reason == "unknown"
    rig.clock.advance(5)
    (placed,) = await drain(rig)
    assert isinstance(placed, BracketPlaced)
    assert placed.position.bracket_id == bracket_id(order.id, 1)


async def test_a_bracket_placed_despite_a_lost_answer_is_adopted(rig: BrokerRig) -> None:
    rig.fake.timeout("/api/v3/orderList/oco", placed=True)
    order = buy()
    result = await rig.broker.submit(order, interval=H1)
    (entry,) = result.events
    assert isinstance(entry, EntryFilled)
    assert entry.position.bracket_id == bracket_id(order.id)
    assert len(rig.fake.lists) == 1


def test_plans_the_exchange_cannot_hold_are_refused() -> None:
    assert exit_plan_problems(PLAN) == []
    assert exit_plan_problems(ExitPlan(stop=0.1, trailing=0.05)) == []
    (problem,) = exit_plan_problems(ExitPlan(stop=0.05, trailing=0.1))
    assert "stop 0.05 is tighter than trailing 0.1" in problem
    (problem,) = exit_plan_problems(ExitPlan(take_profit=0.1, tp_fill="close"))
    assert "tp_fill = 'close'" in problem
    with pytest.raises(ExitPlanError, match="strategy s"):
        check_exit_plan(ExitPlan(stop=0.05, trailing=0.1), name="strategy s")


async def test_the_broker_refuses_such_plans_at_construction_and_submit(rig: BrokerRig) -> None:
    with pytest.raises(ExitPlanError):
        BinanceBroker(rig.client, plans=[PLAN, ExitPlan(take_profit=0.1, tp_fill="close")])
    result = await rig.broker.submit(buy(plan=ExitPlan(stop=0.05, trailing=0.1)), interval=H1)
    assert (result.status, result.reason) == (SubmitStatus.REJECTED, "unsupported_exit_plan")


# -- partial fills and fees -------------------------------------------------------------------


async def test_a_partly_filled_entry_is_reported_once_complete(rig: BrokerRig) -> None:
    rig.fake.hold_market.add(SYMBOL)
    order = buy()
    result = await rig.broker.submit(order, interval=H1)
    assert result.events == ()
    rig.fake.fill_part(order.id, D("0.4"))  # 4.000 at 100
    assert await drain(rig) == []
    rig.fake.set_price(SYMBOL, D("101.00"))
    rig.fake.release_market(SYMBOL, parts=2)  # the rest at 101.00 and 101.01
    (event, *_) = await drain(rig)
    assert isinstance(event, EntryFilled)
    fake_order = rig.fake.orders[order.id]
    assert event.fill.quantity == pytest.approx(float(fake_order.executed))
    avg = fake_order.cumulative_quote / fake_order.executed
    assert event.fill.price == pytest.approx(float(avg))
    assert event.position.cost == pytest.approx(float(fake_order.cumulative_quote))
    assert len(fake_order.fills) == 3


async def test_a_partly_filled_exit_leg_closes_once_filled(rig: BrokerRig) -> None:
    event = await entered(rig)
    tp = rig.fake.lists[bracket_id(buy().id)].legs[0].price
    rig.fake.liquidity[SYMBOL] = D("4")
    rig.fake.set_price(SYMBOL, tp)  # 4 of 9.990
    assert await drain(rig) == []
    rig.fake.set_price(SYMBOL, tp - D("1"))  # the stop leg expired at the first fill
    rig.fake.set_price(SYMBOL, tp)  # 4 more
    assert await drain(rig) == []
    rig.fake.set_price(SYMBOL, tp + D("1"))  # the last 1.990
    (exit_,) = await drain(rig)
    assert isinstance(exit_, ExitFilled)
    assert exit_.fill.reason is ExitReason.TAKE_PROFIT
    assert exit_.fill.quantity == pytest.approx(event.position.quantity)
    assert exit_.fill.price == pytest.approx(float(tp))


async def test_a_partly_filled_bracket_cancelled_for_a_sell_reports_the_whole_exit(
    rig: BrokerRig,
) -> None:
    event = await entered(rig)
    tp = rig.fake.lists[bracket_id(buy().id)].legs[0].price
    rig.fake.liquidity[SYMBOL] = D("4")
    rig.fake.set_price(SYMBOL, tp)
    assert await drain(rig) == []
    rig.fake.set_price(SYMBOL, D("105.00"))
    result = await rig.broker.cancel(bracket_order(event.position))
    assert (result.status, result.events) == (CancelStatus.CANCELLED, ())
    sold = await rig.broker.submit(sell(), interval=H1)
    (exit_,) = sold.events
    assert isinstance(exit_, ExitFilled)
    assert exit_.fill.order_id == sell().id
    assert exit_.fill.reason is None
    assert exit_.fill.quantity == pytest.approx(event.position.quantity)
    expected = (D(4) * tp + (D(str(event.position.quantity)) - 4) * D("105.00")) / D(
        str(event.position.quantity)
    )
    assert exit_.fill.price == pytest.approx(float(expected))
    assert rig.broker.positions == {}
    assert await drain(rig) == []


async def test_commissions_in_base_usdt_and_another_asset(rig: BrokerRig) -> None:
    event = await entered(rig)
    assert event.position.quantity == pytest.approx(9.99)  # 10 bought, 0.01 commission
    assert event.fill.fee == pytest.approx(0.01 * 100)  # the base commission's USDT value
    rig.fake.set_price(SYMBOL, D("111"))
    (exit_,) = await drain(rig)
    assert isinstance(exit_, ExitFilled)
    assert exit_.fill.fee == pytest.approx(9.99 * 110 * 0.001)  # in USDT
    # BNB: the base quantity is not reduced and the commission is recorded, not converted
    rig.fake.commission_asset = "BNB"
    rig.fake.deposit("BNB", D(1))
    other = await entered(rig, buy(symbol="BBBUSDT", n=2))
    assert other.position.quantity == pytest.approx(10.0)
    assert other.fill.fee == 0
    assert rig.broker.other_commissions == [(buy(n=2).id, "BNB", D("0.00333333"))]


# -- rejections and unknown outcomes ----------------------------------------------------------


async def test_rejections_carry_reasons(rig: BrokerRig) -> None:
    result = await rig.broker.submit(buy(amount=4.0), interval=H1)
    assert (result.status, result.reason) == (SubmitStatus.REJECTED, "min_notional")
    assert rig.fake.requests_to("/api/v3/order") == []
    rig.fake.fail("/api/v3/order", code=-1013, msg="Filter failure: LOT_SIZE")
    result = await rig.broker.submit(buy(n=2), interval=H1)
    assert (result.status, result.reason) == (SubmitStatus.REJECTED, "filter_lot_size")
    result = await rig.broker.submit(buy(amount=20_000.0, n=3), interval=H1)
    assert (result.status, result.reason) == (SubmitStatus.REJECTED, "insufficient_balance")
    result = await rig.broker.submit(buy(symbol="OLDUSDT", n=4), interval=H1)
    assert (result.status, result.reason) == (SubmitStatus.REJECTED, "symbol_not_trading")
    result = await rig.broker.submit(sell(), interval=H1)
    assert (result.status, result.reason) == (SubmitStatus.REJECTED, "not_holding")
    assert await drain(rig) == []


@pytest.mark.parametrize("how", ["error", "timeout"])
async def test_an_unknown_submit_that_was_placed_is_found_by_query_once(
    rig: BrokerRig, how: str
) -> None:
    if how == "error":
        rig.fake.fail_after("/api/v3/order", method="POST")
    else:
        rig.fake.timeout("/api/v3/order", method="POST", placed=True)
    order = buy()
    result = await rig.broker.submit(order, interval=H1)
    assert result.status is SubmitStatus.UNKNOWN
    found = await rig.broker.query(order)
    assert found.state is OrderState.FILLED
    (entry,) = found.events
    assert isinstance(entry, EntryFilled)
    assert entry.position.bracket_id == bracket_id(order.id)
    again = await rig.broker.query(order)
    assert (again.state, again.events) == (OrderState.FILLED, ())
    assert await drain(rig) == []
    assert len(rig.fake.requests_to("/api/v3/order", "POST")) == 1


async def test_an_unknown_submit_that_was_not_placed_is_not_found(rig: BrokerRig) -> None:
    rig.fake.fail("/api/v3/order", method="POST", status=503, code=-1007, msg="Timeout")
    order = buy()
    result = await rig.broker.submit(order, interval=H1)
    assert result.status is SubmitStatus.UNKNOWN
    found = await rig.broker.query(order)
    assert found.state is OrderState.NOT_FOUND
    assert await drain(rig) == []


async def test_a_duplicate_id_is_resolved_by_query(rig: BrokerRig) -> None:
    rig.fake.hold_market.add(SYMBOL)
    order = buy()
    await rig.broker.submit(order, interval=H1)
    other = BinanceBroker(rig.client, clock=rig.clock)  # a second process, same exchange
    result = await other.submit(order, interval=H1)
    assert result.status is SubmitStatus.ACCEPTED
    assert result.events == ()


async def test_an_auth_failure_halts_order_placement(rig: BrokerRig) -> None:
    rig.fake.fail("/api/v3/order", status=401, code=-2015, msg="Invalid API-key")
    result = await rig.broker.submit(buy(), interval=H1)
    assert (result.status, result.reason) == (SubmitStatus.REJECTED, "auth")
    (halted,) = await drain(rig)
    assert isinstance(halted, BrokerHalted)
    assert halted.reason == "auth"
    before = len(rig.fake.requests)
    result = await rig.broker.submit(buy(n=2), interval=H1)
    assert (result.status, result.reason) == (SubmitStatus.REJECTED, "auth")
    assert len(rig.fake.requests) == before
    assert rig.broker.halted == "auth"


async def test_a_ban_halts_order_placement(rig: BrokerRig) -> None:
    await entered(rig)
    rig.fake.ban()
    (halted,) = await rig.broker.events(rig.clock())  # the poll meets the ban
    assert isinstance(halted, BrokerHalted)
    assert halted.reason == "banned"
    before = len(rig.fake.requests)
    result = await rig.broker.submit(buy(symbol="BBBUSDT", n=2), interval=H1)
    assert (result.status, result.reason) == (SubmitStatus.REJECTED, "banned")
    assert len(rig.fake.requests) == before
    assert await rig.broker.events(rig.clock()) == []  # reported once


# -- cancels ----------------------------------------------------------------------------------


async def test_cancelling_a_bracket_frees_the_position_to_sell(rig: BrokerRig) -> None:
    event = await entered(rig)
    result = await rig.broker.cancel(bracket_order(event.position))
    assert (result.status, result.events) == (CancelStatus.CANCELLED, ())
    assert rig.fake.lists[bracket_id(buy().id)].status == "ALL_DONE"
    sold = await rig.broker.submit(sell(), interval=H1)
    (exit_,) = sold.events
    assert isinstance(exit_, ExitFilled)
    assert exit_.fill.quantity == pytest.approx(event.position.quantity)
    assert exit_.proceeds == pytest.approx(event.position.quantity * 100 * 0.999)
    assert await drain(rig) == []


async def test_a_sell_while_the_bracket_is_open_is_refused(rig: BrokerRig) -> None:
    await entered(rig)
    result = await rig.broker.submit(sell(), interval=H1)
    assert (result.status, result.reason) == (SubmitStatus.REJECTED, "bracket_open")


async def test_cancelling_a_bracket_that_filled_reports_the_fill(rig: BrokerRig) -> None:
    event = await entered(rig)
    rig.fake.omit_events(10)  # the stream misses the fill
    rig.fake.set_price(SYMBOL, D("111"))
    result = await rig.broker.cancel(bracket_order(event.position))
    assert result.status is CancelStatus.NOT_OPEN
    (exit_,) = result.events
    assert isinstance(exit_, ExitFilled)
    assert exit_.fill.reason is ExitReason.TAKE_PROFIT
    assert await drain(rig) == []
    again = await rig.broker.cancel(bracket_order(event.position))
    assert (again.status, again.events) == (CancelStatus.NOT_OPEN, ())


async def test_a_cancel_without_an_answer_is_unknown(rig: BrokerRig) -> None:
    event = await entered(rig)
    rig.fake.timeout("/api/v3/orderList", method="DELETE")
    result = await rig.broker.cancel(bracket_order(event.position))
    assert result.status is CancelStatus.UNKNOWN
    result = await rig.broker.cancel(bracket_order(event.position))
    assert result.status is CancelStatus.CANCELLED


async def test_a_cancel_whose_answer_was_lost_is_cancelled_on_retry(rig: BrokerRig) -> None:
    event = await entered(rig)
    rig.fake.timeout("/api/v3/orderList", method="DELETE", placed=True)
    result = await rig.broker.cancel(bracket_order(event.position))
    assert result.status is CancelStatus.UNKNOWN
    assert rig.fake.lists[bracket_id(buy().id)].status == "ALL_DONE"  # it did cancel
    result = await rig.broker.cancel(bracket_order(event.position))
    assert (result.status, result.events) == (CancelStatus.CANCELLED, ())
    sold = await rig.broker.submit(sell(), interval=H1)
    assert [type(e) for e in sold.events] == [ExitFilled]
    assert await drain(rig) == []


async def test_cancelling_a_bracket_the_exchange_lost_reports_it_cancelled(
    rig: BrokerRig,
) -> None:
    event = await entered(rig)
    rig.fake.reset()
    rig.fake.deposit("AAA", D("9.99"))
    result = await rig.broker.cancel(bracket_order(event.position))
    assert result.status is CancelStatus.NOT_OPEN
    (cancelled,) = result.events
    assert isinstance(cancelled, OrderCancelled)
    assert cancelled.order.id == event.position.bracket_id
    sold = await rig.broker.submit(sell(), interval=H1)
    assert [type(e) for e in sold.events] == [ExitFilled]


async def test_a_fill_whose_trades_cannot_be_fetched_is_reported_later() -> None:
    rig = await make_rig(stream=False)
    try:
        await entered(rig)
        rig.fake.set_price(SYMBOL, D("111"))
        rig.fake.fail("/api/v3/myTrades", status=500, times=10)
        assert await rig.broker.events(rig.clock()) == []  # commissions unknown: not yet
        rig.fake.faults.clear()
        (exit_,) = await rig.broker.events(rig.clock())
        assert isinstance(exit_, ExitFilled)
        assert exit_.fill.fee == pytest.approx(9.99 * 110 * 0.001)
        assert await rig.broker.events(rig.clock()) == []
    finally:
        await rig.close()


async def test_a_failing_poll_is_retried_at_the_next_call() -> None:
    rig = await make_rig(stream=False, poll_interval_down=timedelta(minutes=5))
    try:
        await entered(rig)
        rig.fake.set_price(SYMBOL, D("111"))
        rig.fake.fail("/api/v3/orderList", method="GET", status=500, times=10)
        assert await rig.broker.events(rig.clock()) == []
        rig.fake.faults.clear()
        (exit_,) = await rig.broker.events(rig.clock())  # not five minutes later: at once
        assert isinstance(exit_, ExitFilled)
    finally:
        await rig.close()


# -- the stream and its gaps ------------------------------------------------------------------


async def test_fills_while_the_stream_is_down_are_polled_once() -> None:
    rig = await make_rig(poll_interval=timedelta(minutes=1))
    try:
        event = await entered(rig)
        rig.fake.refuse = True
        await rig.fake.drop_connections()
        await until(lambda: not rig.broker.stream_connected)
        rig.fake.set_price(SYMBOL, D("111"))  # nobody listens: the events are lost
        (exit_,) = await drain(rig)  # the stream is down: polled
        assert isinstance(exit_, ExitFilled)
        assert exit_.fill.order_id == event.position.bracket_id
        rig.fake.refuse = False
        await until(lambda: rig.broker.stream_connected, timeout=5)
        (report,) = await drain(rig)  # after the reconnect: a reconciliation, no repeat
        assert isinstance(report, Reconciliation)
        assert report.clean
        assert await drain(rig) == []
        assert rig.stream.connects >= 2
    finally:
        await rig.close()


async def test_fills_missed_during_a_short_drop_come_from_the_reconnect_poll() -> None:
    rig = await make_rig(poll_interval=timedelta(minutes=1))
    try:
        event = await entered(rig)
        await rig.broker.events(rig.clock())  # polled now: the next regular poll is far off
        rig.fake.refuse = True
        await rig.fake.drop_connections()
        rig.fake.set_price(SYMBOL, D("94"))
        rig.fake.refuse = False
        await until(lambda: rig.stream.connects >= 2 and rig.broker.stream_connected, timeout=5)
        events = await drain(rig)
        exits = [e for e in events if isinstance(e, ExitFilled)]
        assert len(exits) == 1
        assert exits[0].fill.reason is ExitReason.STOP
        assert exits[0].fill.order_id == event.position.bracket_id
        assert await drain(rig) == []
    finally:
        await rig.close()


async def test_the_same_fill_from_stream_and_poll_is_reported_once() -> None:
    rig = await make_rig(poll_interval=timedelta(seconds=30))
    try:
        await entered(rig)
        await rig.broker.events(rig.clock())
        rig.fake.hold_events()
        rig.fake.set_price(SYMBOL, D("111"))
        rig.clock.advance(31)  # a regular poll is due
        (exit_,) = await drain(rig)
        assert isinstance(exit_, ExitFilled)
        rig.fake.release_events()  # now the stream tells the same story
        assert await drain(rig) == []
    finally:
        await rig.close()


async def test_stream_events_alone_report_fills() -> None:
    rig2 = await make_rig(poll_interval=timedelta(hours=1))
    try:
        await entered(rig2)
        await rig2.broker.events(rig2.clock())
        before = len(rig2.fake.requests)
        rig2.fake.set_price(SYMBOL, D("111"))
        (exit_,) = await drain(rig2)
        assert isinstance(exit_, ExitFilled)
        # the stream had everything, commissions included: no REST request was needed
        assert len(rig2.fake.requests) == before
    finally:
        await rig2.close()


async def test_an_event_stream_terminated_resubscribes_and_polls() -> None:
    rig = await make_rig(poll_interval=timedelta(hours=1))
    try:
        await entered(rig)
        await rig.broker.events(rig.clock())
        subscribes = len(rig.fake.ws_requests)
        rig.fake.terminate_stream()
        await until(lambda: len(rig.fake.ws_requests) > subscribes and rig.fake.subscribed == 1)
        await until(lambda: rig.broker.stream_connected)
        assert rig.stream.connects == 1  # the same connection
        before = len(rig.fake.requests_to("/api/v3/orderList"))
        (report,) = await drain(rig)
        assert isinstance(report, Reconciliation)
        assert len(rig.fake.requests_to("/api/v3/orderList")) > before  # polled after the gap
        rig.fake.set_price(SYMBOL, D("111"))
        (exit_,) = await drain(rig)
        assert isinstance(exit_, ExitFilled)
    finally:
        await rig.close()


async def test_a_server_shutdown_reconnects(rig: BrokerRig) -> None:
    rig.fake.server_shutdown()
    await until(lambda: rig.stream.connects == 2 and rig.broker.stream_connected)
    await entered(rig)
    rig.fake.set_price(SYMBOL, D("111"))
    events = await drain(rig)
    assert sorted(type(e).__name__ for e in events) == ["ExitFilled", "Reconciliation"]


async def test_a_rotation_reconnects_before_the_limit() -> None:
    rig = await make_rig(start=False)
    try:
        await rig.broker.start()
        rig.stream._max_age = timedelta(seconds=0.05)
        await until(lambda: rig.stream.connects >= 2, timeout=5)
    finally:
        await rig.close()


async def test_rest_polling_works_without_a_stream() -> None:
    rig = await make_rig(stream=False)
    try:
        event = await entered(rig)
        rig.fake.set_price(SYMBOL, D("94"))
        (exit_,) = await rig.broker.events(rig.clock())
        assert isinstance(exit_, ExitFilled)
        assert exit_.fill.order_id == event.position.bracket_id
    finally:
        await rig.close()


# -- reconciliation and restarts --------------------------------------------------------------


async def test_a_new_process_reconciles_and_finds_fills_from_while_it_was_down() -> None:
    clock = Clock()
    fake = FakeExchange(clock)
    first = await make_rig(clock=clock, fake=fake)
    event = await entered(first)
    await first.close()
    fake.set_price(SYMBOL, D("111"))  # fills while no broker runs
    second = await make_rig(clock=clock, fake=fake)
    try:
        position = event.position
        working = WorkingOrder(bracket_order(position), H1, "s")
        report = await second.broker.reconcile([working], [position], now=clock())
        assert report.clean
        (exit_,) = await drain(second)
        assert isinstance(exit_, ExitFilled)
        assert exit_.fill.reason is ExitReason.TAKE_PROFIT
        assert await drain(second) == []
    finally:
        await second.close()


async def test_reconcile_reports_orphans_and_balance_mismatches() -> None:
    clock = Clock()
    fake = FakeExchange(clock)
    first = await make_rig(clock=clock, fake=fake)
    a = await entered(first)
    b = await entered(first, buy(symbol="BBBUSDT", n=2))
    await first.close()
    fake.balances["AAA"] = [D(0), D("1.0")]  # coins gone from the account
    second = await make_rig(clock=clock, fake=fake)
    try:
        # the state file knows A only: B's bracket is an orphan
        report = await second.broker.reconcile(
            [WorkingOrder(bracket_order(a.position), H1, "s")], [a.position], now=clock()
        )
        (orphan,) = report.orphans
        assert orphan.order_id == b.position.bracket_id
        assert orphan.symbol == "BBBUSDT"
        (mismatch,) = report.mismatches
        assert mismatch.asset == "AAA"
        assert mismatch.expected == pytest.approx(a.position.quantity)
        assert mismatch.actual == pytest.approx(1.0)
    finally:
        await second.close()


async def test_reconcile_adopts_a_bracket_the_state_file_lost() -> None:
    clock = Clock()
    fake = FakeExchange(clock)
    fake.fail("/api/v3/orderList/oco", code=-1013, msg="Filter failure: PERCENT_PRICE_BY_SIDE")
    first = await make_rig(clock=clock, fake=fake)
    order = buy()
    result = await first.broker.submit(order, interval=H1)
    entry = result.events[0]
    assert isinstance(entry, EntryFilled)
    clock.advance(10)
    (placed,) = await drain(first)  # placed, but the engine crashed before recording it
    assert isinstance(placed, BracketPlaced)
    await first.close()
    second = await make_rig(clock=clock, fake=fake)
    try:
        report = await second.broker.reconcile([], [entry.position], now=clock())
        assert report.clean  # not an orphan: adopted
        (adopted,) = await drain(second)
        assert isinstance(adopted, BracketPlaced)
        assert adopted.position.bracket_id == bracket_id(order.id)
    finally:
        await second.close()


async def test_a_testnet_reset_loses_the_bracket(rig: BrokerRig) -> None:
    event = await entered(rig)
    rig.fake.reset()
    (lost,) = await drain(rig)  # the poll finds the bracket gone: the position is unprotected
    assert isinstance(lost, OrderCancelled)
    assert lost.order.id == event.position.bracket_id
    assert lost.reason is CancelReason.CANCELLED
    report = await rig.broker.reconcile([], [event.position], now=rig.clock())
    (mismatch,) = report.mismatches
    assert mismatch.actual == 0


# -- secrets ----------------------------------------------------------------------------------


async def test_keys_and_signatures_never_reach_logs_or_reprs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    rig = await make_rig()
    try:
        await entered(rig)
        rig.fake.set_price(SYMBOL, D("111"))
        await drain(rig)
        rig.fake.server_shutdown()
        await until(lambda: rig.stream.connects == 2 and rig.broker.stream_connected)
        await drain(rig)
        signatures = [
            r.url.params["signature"] for r in rig.fake.requests if "signature" in r.url.params
        ]
        signatures += [m["params"]["signature"] for m in rig.fake.ws_requests if "params" in m]
        assert len(signatures) > 5
        text = caplog.text + repr(rig.broker) + repr(rig.stream) + repr(rig.client)
        for secret in (API_KEY, API_SECRET, *signatures):
            assert secret not in text
        assert "signature=" not in caplog.text
        assert rig.hosts() == {"testnet.invalid", "127.0.0.1"}
    finally:
        await rig.close()


def test_broker_types_are_exported() -> None:
    assert OrderType.BRACKET.value == "bracket"


async def test_cancelling_a_resting_market_order_and_a_lost_cancel_answer(rig: BrokerRig) -> None:
    rig.fake.hold_market.add(SYMBOL)
    first, second = buy(), buy(symbol="BBBUSDT", n=2)
    rig.fake.hold_market.add("BBBUSDT")
    for order in (first, second):
        await rig.broker.submit(order, interval=H1)
    result = await rig.broker.cancel(first)
    assert (result.status, result.events) == (CancelStatus.CANCELLED, ())
    rig.fake.timeout("/api/v3/order", method="DELETE", placed=True)
    result = await rig.broker.cancel(second)
    assert result.status is CancelStatus.UNKNOWN
    result = await rig.broker.cancel(second)  # it was cancelled: the retry says so
    assert (result.status, result.events) == (CancelStatus.CANCELLED, ())
    assert await drain(rig) == []
    assert rig.fake.balance("USDT") == (D(10_000), D(0))


async def test_a_partly_filled_market_order_cancelled_reports_its_fill(rig: BrokerRig) -> None:
    rig.fake.hold_market.add(SYMBOL)
    order = buy()
    await rig.broker.submit(order, interval=H1)
    rig.fake.fill_part(order.id, D("0.5"))
    result = await rig.broker.cancel(order)
    assert result.status is CancelStatus.NOT_OPEN
    (entry, *_) = result.events
    assert isinstance(entry, EntryFilled)
    assert entry.fill.quantity == pytest.approx(5.0)
