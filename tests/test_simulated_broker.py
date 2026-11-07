from datetime import timedelta

import pytest
from broker_fakes import T0, candles, flat

from crypto_moonshot.broker.base import (
    Broker,
    Costs,
    ExitPlan,
    ExitReason,
    Order,
    OrderType,
    Side,
    base_asset,
)
from crypto_moonshot.broker.simulated import SimulatedBroker

M5 = timedelta(minutes=5)
COSTS = Costs(fee=0.001, slippage=0.0005, latency=0.0)
PLAN = ExitPlan(take_profit=0.2, stop=0.05, max_hold=timedelta(hours=1))


def buy(amount: float = 50.0, *, at: int = 0, plan: ExitPlan | None = PLAN) -> Order:
    return Order("ABCUSDT", Side.BUY, T0 + at * M5, quote_amount=amount, exit_plan=plan)


def broker_with(
    bars: list[tuple[float, float, float, float] | None], ended: bool = True
) -> SimulatedBroker:
    broker = SimulatedBroker(cash=1_000.0, costs=COSTS)
    broker.set_candles("ABCUSDT", candles(bars, ended=ended))
    return broker


def test_implements_the_broker_protocol() -> None:
    broker: Broker = SimulatedBroker(cash=1.0)
    assert broker.balances() == {"USDT": 1.0}


def test_market_buy_fills_at_the_open_with_costs() -> None:
    broker = broker_with([*flat(2), (100, 125, 99, 110), *flat(10)])
    fill = broker.place_order(buy(at=1))
    assert fill is not None
    assert (fill.side, fill.time) == (Side.BUY, T0 + M5)
    assert fill.price == pytest.approx(100.05)
    assert fill.quantity == pytest.approx(50 / (100.05 * 1.001))
    assert fill.fee == pytest.approx(fill.quantity * 100.05 * 0.001)
    assert broker.cash == pytest.approx(950)
    assert broker.balances() == {"USDT": pytest.approx(950), "ABC": pytest.approx(fill.quantity)}


def test_buy_fills_at_the_first_candle_after_a_hole() -> None:
    broker = broker_with([*flat(1), None, (101, 101, 101, 101)])
    fill = broker.place_order(buy(at=1, plan=None))
    assert fill is not None
    assert fill.time == T0 + 2 * M5
    assert fill.price == pytest.approx(101 * 1.0005)


def test_bracket_is_open_until_settled_then_take_profit_fills() -> None:
    broker = broker_with([*flat(2), (100, 125, 99, 110), *flat(10)])
    entry = broker.place_order(buy(at=1))
    assert entry is not None
    [bracket] = broker.open_orders()
    assert (bracket.type, bracket.side, bracket.exit_plan) == (OrderType.BRACKET, Side.SELL, PLAN)
    assert bracket.quantity == entry.quantity
    assert broker.positions["ABCUSDT"].bracket_id == bracket.id

    assert broker.settle(T0 + 3 * M5 - timedelta(microseconds=1)) == []
    assert broker.open_orders() == [bracket]

    [exit_fill] = broker.settle(T0 + 3 * M5)
    assert exit_fill.reason is ExitReason.TAKE_PROFIT
    assert exit_fill.order_id == bracket.id
    assert exit_fill.price == pytest.approx(120 * 0.9995)
    proceeds = entry.quantity * 120 * 0.9995 * 0.999
    assert broker.cash == pytest.approx(950 + proceeds)
    assert broker.positions == {}
    assert broker.open_orders() == []
    assert broker.balances() == {"USDT": pytest.approx(950 + proceeds)}
    assert broker.fills == [entry, exit_fill]
    net = proceeds / 50 - 1
    assert net == pytest.approx(120 * 0.9995 * 0.999 / (100.05 * 1.001) - 1)


def test_placing_an_order_settles_due_exits_first() -> None:
    broker = broker_with([(100, 100, 90, 92), *flat(20, 92)])
    broker.place_order(buy(at=0))
    # The stop filled at the close of the first candle, so a later buy sees the cash back.
    fill = broker.place_order(buy(at=3, plan=None))
    assert fill is not None
    stop = broker.fills[1]
    assert (stop.reason, stop.time) == (ExitReason.STOP, T0 + M5)
    assert stop.price == pytest.approx(95 * 0.9995)
    assert len(broker.fills) == 3


def test_time_exit_through_broker() -> None:
    broker = broker_with(flat(20))
    broker.place_order(buy())
    [fill] = broker.settle(T0 + timedelta(days=1))
    assert (fill.reason, fill.time) == (ExitReason.TIME, T0 + timedelta(hours=1))


def test_series_end_through_broker() -> None:
    broker = broker_with(flat(3))
    broker.place_order(buy())
    [fill] = broker.settle(T0 + timedelta(days=1))
    assert (fill.reason, fill.time) == (ExitReason.SERIES_END, T0 + 3 * M5)


def test_unresolved_bracket_is_resolved_when_more_candles_arrive() -> None:
    broker = broker_with(flat(3), ended=False)
    broker.place_order(buy(plan=ExitPlan(take_profit=0.2)))
    assert broker.settle(T0 + timedelta(days=1)) == []
    broker.set_candles("ABCUSDT", candles([*flat(5), (100, 130, 100, 125)]))
    [fill] = broker.settle(T0 + timedelta(days=1))
    assert (fill.reason, fill.time) == (ExitReason.TAKE_PROFIT, T0 + 6 * M5)


def test_market_sell_cancels_the_bracket() -> None:
    broker = broker_with([*flat(3), (110, 110, 110, 110), *flat(20)])
    entry = broker.place_order(buy())
    assert entry is not None
    sell = broker.place_order(Order("ABCUSDT", Side.SELL, T0 + 3 * M5))
    assert sell is not None
    assert sell.reason is None
    assert sell.price == pytest.approx(110 * 0.9995)
    assert sell.quantity == entry.quantity
    assert broker.open_orders() == []
    assert broker.positions == {}
    assert broker.settle(T0 + timedelta(days=1)) == []


def test_partial_sell_after_cancelling_the_bracket() -> None:
    broker = broker_with(flat(20))
    entry = broker.place_order(buy())
    assert entry is not None
    [bracket] = broker.open_orders()
    broker.cancel_order(bracket.id)
    assert broker.open_orders() == []
    assert broker.positions["ABCUSDT"].bracket_id is None
    with pytest.raises(KeyError):
        broker.cancel_order(bracket.id)
    broker.place_order(Order("ABCUSDT", Side.SELL, T0 + M5, quantity=entry.quantity / 2))
    assert broker.positions["ABCUSDT"].quantity == pytest.approx(entry.quantity / 2)
    assert broker.positions["ABCUSDT"].cost == pytest.approx(25)
    assert broker.settle(T0 + timedelta(days=1)) == []  # cancelled: no time exit


def test_partial_sell_with_an_open_bracket_is_refused() -> None:
    broker = broker_with(flat(20))
    entry = broker.place_order(buy())
    assert entry is not None
    with pytest.raises(ValueError, match="cancel the bracket"):
        broker.place_order(Order("ABCUSDT", Side.SELL, T0 + M5, quantity=entry.quantity / 2))


@pytest.mark.parametrize(
    ("order", "match"),
    [
        (buy(2_000), "not enough cash"),
        (buy(at=30), "no ABCUSDT candle"),
        (Order("ABCUSDT", Side.SELL, T0), "not holding"),
        (Order("XYZUSDT", Side.BUY, T0, quote_amount=1.0), "no candles for XYZUSDT"),
    ],
)
def test_refused_orders(order: Order, match: str) -> None:
    broker = broker_with(flat(20))
    with pytest.raises(ValueError, match=match):
        broker.place_order(order)


def test_one_position_per_symbol() -> None:
    broker = broker_with(flat(20))
    broker.place_order(buy())
    with pytest.raises(ValueError, match="already holding"):
        broker.place_order(buy(at=1))


def test_order_validation() -> None:
    with pytest.raises(ValueError, match="quote_amount"):
        Order("ABCUSDT", Side.BUY, T0)
    with pytest.raises(ValueError, match="exit plan"):
        Order("ABCUSDT", Side.SELL, T0, exit_plan=PLAN)
    with pytest.raises(ValueError, match="timezone-aware"):
        Order("ABCUSDT", Side.BUY, T0.replace(tzinfo=None), quote_amount=1.0)
    with pytest.raises(ValueError, match="USDT-quoted"):
        base_asset("BTCEUR")
    bracket = Order("ABCUSDT", Side.SELL, T0, OrderType.BRACKET, exit_plan=PLAN)
    broker = broker_with(flat(2))
    with pytest.raises(ValueError, match="opened by a buy"):
        broker.place_order(bracket)


def test_pending_exit_is_known_once_the_window_covers_it() -> None:
    plan = ExitPlan(take_profit=0.2)
    broker = broker_with([*flat(3)], ended=False)
    assert broker.pending_exit("ABCUSDT") is None
    broker.place_order(buy(at=1, plan=plan))
    assert broker.pending_exit("ABCUSDT") is None
    broker.set_candles("ABCUSDT", candles([*flat(3), (100, 130, 100, 125)], ended=False))
    pending = broker.pending_exit("ABCUSDT")
    assert pending is not None
    assert (pending.reason, pending.time) == (ExitReason.TAKE_PROFIT, T0 + 4 * M5)
    broker.settle(T0 + 4 * M5)
    assert broker.pending_exit("ABCUSDT") is None
