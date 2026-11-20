import re
from datetime import UTC, datetime, timedelta

import pytest
from broker_fakes import T0, Bar, candles, flat

from crypto_moonshot.broker.base import (
    CLIENT_ORDER_ID,
    Broker,
    Costs,
    ExitPlan,
    ExitReason,
    Fill,
    Order,
    OrderRejected,
    OrderType,
    RejectReason,
    Side,
    child_order_id,
    client_order_id,
)
from crypto_moonshot.broker.candles import Candles, to_us
from crypto_moonshot.broker.paper import (
    BRACKET_ROLE,
    BrokerEvent,
    CancelReason,
    EntryFilled,
    ExitFilled,
    OrderCancelled,
    PaperBroker,
    PaperSnapshot,
)
from crypto_moonshot.broker.simulated import SimulatedBroker, resolve_exit
from crypto_moonshot.data.stream import Candle
from crypto_moonshot.models import Interval

M5 = timedelta(minutes=5)
COSTS = Costs(fee=0.001, slippage=0.0005, latency=0.0)
PLAN = ExitPlan(take_profit=0.2, stop=0.05, max_hold=timedelta(hours=1))
SYM = "ABCUSDT"


def at(k: int) -> datetime:
    return T0 + k * M5


def buy(
    amount: float = 50.0,
    *,
    k: int = 0,
    plan: ExitPlan | None = PLAN,
    symbol: str = SYM,
    id: str | None = None,
) -> Order:
    return Order(
        symbol,
        Side.BUY,
        at(k),
        quote_amount=amount,
        exit_plan=plan,
        id=id or f"{symbol}-buy-{k}",
    )


def feed(broker: PaperBroker, bars: list[Bar | None], *, start: int = 0, symbol: str = SYM) -> None:
    broker.add_candles(symbol, candles(bars, start=at(start)))


def fills(events: list[BrokerEvent]) -> list[Fill]:
    return [e.fill for e in events if isinstance(e, EntryFilled | ExitFilled)]


def test_implements_the_broker_protocol() -> None:
    broker: Broker = PaperBroker(cash=1.0)
    assert broker.balances() == {"USDT": 1.0}
    assert broker.open_orders() == []
    assert broker.place_order(buy(1.0)) is None  # fills come from `advance`


# Entries ---------------------------------------------------------------------------------


def test_buy_fills_at_the_open_of_its_candle_once_it_has_closed() -> None:
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    broker.place_order(buy(k=1, plan=None))
    assert broker.cash == pytest.approx(950)
    assert broker.reserved == pytest.approx(50)
    feed(broker, [*flat(1), (100, 125, 99, 110), *flat(3)])
    assert broker.advance(at(2) - timedelta(microseconds=1)) == []  # candle 1 still open
    [event] = broker.advance(at(2))
    assert isinstance(event, EntryFilled)
    reference = SimulatedBroker(cash=1_000.0, costs=COSTS)
    reference.set_candles(SYM, candles([*flat(1), (100, 125, 99, 110), *flat(3)]))
    assert event.fill == reference.place_order(buy(k=1, plan=None))
    assert event.fill.time == at(1)
    assert event.fill.price == pytest.approx(100.05)
    position = event.position
    assert (position.entry_ref, position.cost, position.bracket_id) == (100, 50, None)
    assert broker.positions[SYM] == position
    assert broker.reserved == 0
    assert broker.balances() == {"USDT": pytest.approx(950), "ABC": event.fill.quantity}


def test_a_hole_at_the_entry_time_cancels_the_buy_and_releases_its_cash() -> None:
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    broker.place_order(buy(k=1))
    feed(broker, [*flat(1), None, *flat(1)])
    assert broker.advance(at(2)) == []  # nothing at or after the entry has closed yet
    [event] = broker.advance(at(3))
    assert event == OrderCancelled(buy(k=1), at(1), CancelReason.NO_ENTRY_CANDLE)
    assert broker.cash == pytest.approx(1_000)
    assert broker.open_orders() == []
    assert broker.positions == {}


def test_a_late_entry_candle_still_fills_if_it_arrives_before_the_advance() -> None:
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    broker.place_order(buy(k=1, plan=None))
    feed(broker, flat(1, 101), start=2)  # the next candle arrives first
    feed(broker, flat(1, 100), start=1)
    [event] = broker.advance(at(3))
    assert isinstance(event, EntryFilled)
    assert event.fill.time == at(1)


def test_a_buy_without_candles_stays_pending_until_cancelled() -> None:
    broker = PaperBroker(cash=1_000.0)
    order = buy(k=1)
    broker.place_order(order)
    assert broker.advance(at(100)) == []
    assert broker.open_orders() == [order]
    broker.cancel_order(order.id)
    assert broker.cash == 1_000
    assert broker.open_orders() == []
    with pytest.raises(KeyError):
        broker.cancel_order(order.id)


def test_cash_is_reserved_at_placement_so_it_is_never_over_committed() -> None:
    broker = PaperBroker(cash=100.0)
    broker.place_order(buy(60.0, symbol="AAAUSDT"))
    with pytest.raises(OrderRejected, match="not enough cash") as info:
        broker.place_order(buy(60.0, symbol="BBBUSDT"))
    assert info.value.reason is RejectReason.CASH
    broker.place_order(buy(40.0, symbol="BBBUSDT"))
    assert broker.balances()["USDT"] == pytest.approx(0)


@pytest.mark.parametrize(
    ("setup", "order", "reason"),
    [
        ([buy()], buy(k=1), RejectReason.SYMBOL_HELD),
        ([], Order(SYM, Side.SELL, at(0)), RejectReason.NOT_HOLDING),
        ([buy(id="same")], buy(symbol="XYZUSDT", id="same"), RejectReason.DUPLICATE_ID),
    ],
)
def test_refused_orders(setup: list[Order], order: Order, reason: RejectReason) -> None:
    broker = PaperBroker(cash=1_000.0)
    for placed in setup:
        broker.place_order(placed)
    with pytest.raises(OrderRejected) as info:
        broker.place_order(order)
    assert info.value.reason is reason


def test_invalid_orders() -> None:
    broker = PaperBroker(cash=1_000.0)
    with pytest.raises(ValueError, match="candle open"):
        broker.place_order(Order(SYM, Side.BUY, T0 + timedelta(minutes=1), quote_amount=1.0))
    bracket = Order(SYM, Side.SELL, T0, OrderType.BRACKET, exit_plan=PLAN)
    with pytest.raises(ValueError, match="opened by a buy"):
        broker.place_order(bracket)
    with pytest.raises(ValueError, match="aligned"):
        broker.add_candles(SYM, candles(flat(1), start=T0 + timedelta(minutes=1)))
    with pytest.raises(ValueError, match="no fixed length"):
        PaperBroker(cash=1.0, interval=Interval.MO1)


# Brackets ----------------------------------------------------------------------------------


def test_bracket_id_is_derived_from_the_buy() -> None:
    broker = PaperBroker(cash=1_000.0)
    broker.place_order(buy())
    feed(broker, flat(1))
    [entry] = broker.advance(at(1))
    assert isinstance(entry, EntryFilled)
    [bracket] = broker.open_orders()
    assert bracket.id == child_order_id(buy().id, BRACKET_ROLE) == entry.position.bracket_id
    assert (bracket.type, bracket.side, bracket.exit_plan) == (OrderType.BRACKET, Side.SELL, PLAN)
    assert bracket.quantity == entry.fill.quantity


def test_take_profit_is_applied_when_its_candle_closes() -> None:
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    broker.place_order(buy(k=1))
    feed(broker, [*flat(2), (100, 125, 99, 110), *flat(2)])
    [entry] = broker.advance(at(2))
    assert broker.advance(at(3) - timedelta(microseconds=1)) == []
    [event] = broker.advance(at(3))
    assert isinstance(event, ExitFilled)
    assert isinstance(entry, EntryFilled)
    assert event.closed
    assert (event.fill.reason, event.fill.time) == (ExitReason.TAKE_PROFIT, at(3))
    assert event.fill.order_id == entry.position.bracket_id
    assert event.fill.price == pytest.approx(120 * 0.9995)
    assert event.fill.fee == pytest.approx(event.fill.notional * 0.001)
    assert broker.cash == pytest.approx(950 + event.proceeds)
    assert broker.positions == {}
    assert broker.open_orders() == []


def test_a_gap_fill_is_applied_at_the_candles_close_with_its_open_time() -> None:
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    broker.place_order(buy())
    feed(broker, [*flat(2), (90, 92, 88, 91)])
    broker.advance(at(2))
    assert broker.advance(at(2)) == []
    [event] = broker.advance(at(3))
    assert isinstance(event, ExitFilled)
    assert (event.fill.reason, event.fill.time) == (ExitReason.STOP, at(2))
    assert event.fill.price == pytest.approx(90 * 0.9995)


def test_a_time_exit_waits_for_the_deadline_although_a_prefix_says_time() -> None:
    plan = ExitPlan(take_profit=0.5, max_hold=timedelta(hours=1))
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    broker.place_order(buy(plan=plan))
    feed(broker, flat(3))
    # resolve_exit on the first three candles alone returns a time exit...
    found = resolve_exit(candles(flat(3), ended=False), plan, COSTS)
    assert found is not None
    assert found.reason is ExitReason.TIME
    # ...but more candles can still hit the target before the deadline.
    assert fills(broker.advance(at(3)))[0].side is Side.BUY
    assert broker.advance(at(12) - timedelta(microseconds=1)) == []
    [event] = broker.advance(at(12))  # no candles since the third: a hole up to the deadline
    assert isinstance(event, ExitFilled)
    assert (event.fill.reason, event.fill.time) == (ExitReason.TIME, at(12))
    assert event.fill.price == pytest.approx(100 * 0.9995)


def test_a_time_exit_inside_a_candle_waits_for_that_candles_close() -> None:
    plan = ExitPlan(max_hold=timedelta(minutes=12))
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    broker.place_order(buy(plan=plan))
    feed(broker, [*flat(2), (100, 100, 100, 97), *flat(2)])
    broker.advance(at(1))
    assert broker.advance(T0 + timedelta(minutes=13)) == []  # past the deadline, candle open
    [event] = broker.advance(at(3))
    assert isinstance(event, ExitFilled)
    assert (event.fill.reason, event.fill.time) == (ExitReason.TIME, T0 + timedelta(minutes=12))
    assert event.fill.price == pytest.approx(97 * 0.9995)  # the close of the 10-15 candle


def test_without_an_exit_yet_the_position_stays_open() -> None:
    broker = PaperBroker(cash=1_000.0)
    broker.place_order(buy(plan=ExitPlan(take_profit=0.5)))
    feed(broker, flat(50))
    events = broker.advance(at(50))
    assert [type(e) for e in events] == [EntryFilled]
    assert SYM in broker.positions


def test_a_cancelled_bracket_never_exits() -> None:
    broker = PaperBroker(cash=1_000.0)
    broker.place_order(buy())
    feed(broker, flat(1))
    broker.advance(at(1))
    [bracket] = broker.open_orders()
    broker.cancel_order(bracket.id)
    assert broker.positions[SYM].bracket_id is None
    feed(broker, flat(30, 50), start=1)
    assert broker.advance(at(31)) == []


# Market sells ------------------------------------------------------------------------------


def test_market_sell_fills_at_the_next_open_and_cancels_the_bracket() -> None:
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    broker.place_order(buy())
    feed(broker, flat(2))
    [entry] = broker.advance(at(2))
    assert isinstance(entry, EntryFilled)
    sell = Order(SYM, Side.SELL, at(2), id="sell")
    broker.place_order(sell)
    assert broker.open_orders()[0] == sell
    feed(broker, [None, (110, 111, 109, 110)], start=2)  # a hole at the sell time is skipped
    assert broker.advance(at(3)) == []
    [event] = broker.advance(at(4))
    assert isinstance(event, ExitFilled)
    assert (event.fill.order_id, event.fill.reason, event.fill.time) == ("sell", None, at(3))
    assert event.fill.price == pytest.approx(110 * 0.9995)
    assert event.fill.quantity == entry.fill.quantity
    assert broker.positions == {}
    assert broker.open_orders() == []
    assert broker.cash == pytest.approx(950 + event.proceeds)


def test_partial_sells_need_the_bracket_cancelled_first() -> None:
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    broker.place_order(buy())
    feed(broker, flat(3))
    [entry] = broker.advance(at(1))
    assert isinstance(entry, EntryFilled)
    half = entry.fill.quantity / 2
    with pytest.raises(OrderRejected, match="cancel the bracket") as info:
        broker.place_order(Order(SYM, Side.SELL, at(1), quantity=half))
    assert info.value.reason is RejectReason.BRACKET_OPEN
    with pytest.raises(OrderRejected) as too_much:
        broker.place_order(Order(SYM, Side.SELL, at(1), quantity=3 * half))
    assert too_much.value.reason is RejectReason.QUANTITY
    assert entry.position.bracket_id is not None
    broker.cancel_order(entry.position.bracket_id)
    broker.place_order(Order(SYM, Side.SELL, at(1), quantity=half, id="s1"))
    with pytest.raises(OrderRejected) as second:
        broker.place_order(Order(SYM, Side.SELL, at(1), id="s2"))
    assert second.value.reason is RejectReason.SELL_PENDING
    [event] = broker.advance(at(2))
    assert isinstance(event, ExitFilled)
    assert not event.closed
    assert event.remaining is not None
    assert event.remaining.quantity == pytest.approx(half)
    assert event.remaining.cost == pytest.approx(25)
    assert broker.positions[SYM] == event.remaining


def test_a_bracket_gap_at_the_sells_open_wins_and_cancels_the_sell() -> None:
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    broker.place_order(buy())
    feed(broker, [*flat(2), (90, 91, 89, 90)])
    broker.advance(at(2))
    sell = Order(SYM, Side.SELL, at(2), id="sell")
    broker.place_order(sell)
    exit_, cancelled = broker.advance(at(3))
    assert isinstance(exit_, ExitFilled)
    assert (exit_.fill.reason, exit_.fill.time) == (ExitReason.STOP, at(2))
    assert cancelled == OrderCancelled(sell, at(2), CancelReason.POSITION_CLOSED)
    assert broker.open_orders() == []


def test_a_sell_fills_before_a_stop_touched_later_in_its_candle() -> None:
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    broker.place_order(buy())
    feed(broker, [*flat(2), (99, 99, 90, 91)])
    broker.advance(at(2))
    broker.place_order(Order(SYM, Side.SELL, at(2), id="sell"))
    [event] = broker.advance(at(3))
    assert isinstance(event, ExitFilled)
    assert (event.fill.order_id, event.fill.reason, event.fill.time) == ("sell", None, at(2))
    assert event.fill.price == pytest.approx(99 * 0.9995)


# Time order, candles and intervals ---------------------------------------------------------


def test_events_come_in_time_order_across_symbols() -> None:
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    broker.place_order(buy(symbol="AAAUSDT", k=0))
    broker.place_order(buy(symbol="BBBUSDT", k=1))
    feed(broker, [*flat(4), (100, 100, 90, 92)], symbol="AAAUSDT")  # stop at the close of 4
    feed(broker, [None, (100, 125, 99, 110)], symbol="BBBUSDT")  # entry and target in 1
    events = broker.advance(at(10))
    assert [(type(e).__name__, e.time) for e in events] == [
        ("EntryFilled", at(0)),
        ("EntryFilled", at(1)),
        ("ExitFilled", at(2)),
        ("ExitFilled", at(5)),
    ]


def test_candles_of_unused_series_are_dropped_once_closed() -> None:
    broker = PaperBroker(cash=1_000.0)
    feed(broker, flat(10), symbol="XYZUSDT")
    broker.place_order(buy(k=3))
    feed(broker, flat(10))
    broker.advance(at(5))
    assert broker.required_history() == []
    # Unused: only the candles not closed yet (5 to 9) are kept.
    assert min(broker._series[("XYZUSDT", Interval.M5)].bars) == to_us(at(5))
    assert min(broker._series[(SYM, Interval.M5)].bars) == to_us(at(3))  # from the entry
    broker.advance(at(10))
    assert ("XYZUSDT", Interval.M5) not in broker._series


def test_feed_candles_can_be_merged() -> None:
    bars = [Candle(at(k), 100.0, 101.0, 99.0, 100.5, 1.0, 1.0, 1, 0.5, 0.5) for k in (2, 0, 1, 1)]
    series = Candles.from_bars(bars, Interval.M5)
    assert series.open_time.tolist() == [int(candles(flat(3)).open_time[k]) for k in range(3)]
    broker = PaperBroker(cash=1_000.0)
    broker.place_order(buy(plan=None))
    broker.add_candles(SYM, series)
    broker.add_candles(SYM, series)  # duplicates are ignored
    assert len(broker.advance(at(3))) == 1


def test_orders_can_fill_on_another_interval() -> None:
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    hour = timedelta(hours=1)
    order = Order(SYM, Side.BUY, T0 + hour, quote_amount=50.0, exit_plan=ExitPlan(stop=0.05))
    broker.place_order(order, interval=Interval.H1)
    feed(broker, flat(24))  # 5m candles do not count for an hourly order
    assert broker.advance(T0 + 2 * hour) == []
    broker.add_candles(
        SYM,
        candles([(100, 100, 100, 100), (100, 100, 90, 92)], interval=Interval.H1, start=T0 + hour),
    )
    entry, exit_ = broker.advance(T0 + 3 * hour)
    assert isinstance(entry, EntryFilled)
    assert isinstance(exit_, ExitFilled)
    assert entry.position.interval is Interval.H1
    assert (exit_.fill.reason, exit_.fill.time) == (ExitReason.STOP, T0 + 3 * hour)
    broker2 = PaperBroker(cash=1_000.0)
    broker2.place_order(buy(plan=None))
    feed(broker2, flat(1))
    broker2.advance(at(1))
    with pytest.raises(ValueError, match="held on"):
        broker2.place_order(Order(SYM, Side.SELL, at(1)), interval=Interval.H1)


# Recovery ----------------------------------------------------------------------------------


def run_uninterrupted(bars: list[Bar | None], until: int) -> tuple[PaperBroker, list[BrokerEvent]]:
    broker = PaperBroker(cash=1_000.0, costs=COSTS)
    broker.place_order(buy(k=1, plan=ExitPlan(take_profit=0.1, trailing=0.03)))
    broker.place_order(buy(k=3, symbol="XYZUSDT", plan=None))
    events = []
    for k, bar in enumerate(bars[:until]):
        if bar is not None:
            feed(broker, [bar], start=k)
            feed(broker, [bar], start=k, symbol="XYZUSDT")
        events.extend(broker.advance(at(k + 1)))
    return broker, events


BARS: list[Bar | None] = [
    *flat(4),
    (101, 104, 100, 103),
    (103, 106, 102, 105),
    None,
    (105, 105, 101, 102),  # trailing stop: peak 106, level 102.82
    *flat(5, 102),
]


def test_restore_applies_exits_that_happened_while_down_at_their_real_times() -> None:
    whole, whole_events = run_uninterrupted(BARS, len(BARS))
    assert [type(e).__name__ for e in whole_events] == ["EntryFilled", "EntryFilled", "ExitFilled"]
    stop = whole_events[2]
    assert isinstance(stop, ExitFilled)
    assert (stop.fill.reason, stop.fill.time) == (ExitReason.TRAILING_STOP, at(8))

    before, before_events = run_uninterrupted(BARS, 3)  # down after the first entry filled
    assert [type(e) for e in before_events] == [EntryFilled]
    snapshot = before.snapshot()
    assert snapshot == PaperSnapshot(
        before.cash, tuple(before.positions.values()), tuple(before.pending)
    )

    restored = PaperBroker.restore(snapshot, COSTS)
    assert restored.required_history() == [(SYM, Interval.M5, at(1))]
    assert restored.open_orders() == before.open_orders()
    # Only the latest candle is back: without the entry candle nothing can be decided.
    restored.add_candles(SYM, candles(BARS[-1:], start=at(len(BARS) - 1)))
    assert restored.advance(at(len(BARS))) == []
    # Back up: candles from the entry on, all at once (e.g. a REST backfill), then now.
    restored.add_candles(SYM, candles(BARS[1:], start=at(1)))
    restored.add_candles("XYZUSDT", candles(BARS[3:], start=at(3)))
    after = restored.advance(at(len(BARS)))
    assert [e.time for e in after] == [e.time for e in whole_events[1:]]
    assert fills(after) == fills(whole_events[1:])
    assert restored.cash == pytest.approx(whole.cash)
    assert restored.positions == whole.positions


def test_restore_rejects_inconsistent_snapshots() -> None:
    broker = PaperBroker(cash=1_000.0)
    broker.place_order(buy(plan=None))
    feed(broker, flat(1))
    broker.advance(at(1))
    position = broker.positions[SYM]
    with pytest.raises(ValueError, match="twice"):
        PaperBroker.restore(PaperSnapshot(1.0, (position, position)))
    broker.place_order(Order(SYM, Side.SELL, at(1)))
    orphan = PaperSnapshot(1.0, (), broker.snapshot().orders)
    with pytest.raises(ValueError, match="without a position"):
        PaperBroker.restore(orphan)


# Client order IDs --------------------------------------------------------------------------


def test_client_order_ids_are_deterministic_short_and_distinct() -> None:
    t = datetime(2024, 5, 1, 12, 5, tzinfo=UTC)
    first = client_order_id("capitulation", "BTCUSDT", t, Side.BUY)
    assert first == client_order_id("capitulation", "BTCUSDT", t.astimezone(), Side.BUY)
    assert len(first) <= 36
    assert CLIENT_ORDER_ID.fullmatch(first)
    assert re.fullmatch(r"cmb-[0-9a-f]{32}", first)
    others = {
        client_order_id("capitulation", "BTCUSDT", t, Side.SELL),
        client_order_id("capitulation", "ETHUSDT", t, Side.BUY),
        client_order_id("capitulation", "BTCUSDT", t + M5, Side.BUY),
        client_order_id("other", "BTCUSDT", t, Side.BUY),
        client_order_id("capitulationB", "TCUSDT", t, Side.BUY),
        child_order_id(first, BRACKET_ROLE),
    }
    assert first not in others
    assert len(others) == 6
    assert all(CLIENT_ORDER_ID.fullmatch(i) for i in others)
    with pytest.raises(ValueError, match="timezone-aware"):
        client_order_id("s", "BTCUSDT", t.replace(tzinfo=None), Side.BUY)


def test_advance_can_be_limited_to_one_interval() -> None:
    broker = PaperBroker(cash=1_000.0, costs=Costs(fee=0.0, slippage=0.0))
    h1 = Interval.H1
    broker.place_order(
        Order("AAAUSDT", Side.BUY, T0, quote_amount=100.0, id="a"), interval=Interval.M5
    )
    broker.place_order(Order("BBBUSDT", Side.BUY, T0, quote_amount=100.0, id="b"), interval=h1)
    broker.add_candles("AAAUSDT", candles(flat(1)))
    broker.add_candles("BBBUSDT", candles(flat(1), interval=h1))
    assert [
        e.order.symbol if isinstance(e, OrderCancelled) else e.fill.symbol
        for e in broker.advance(T0 + h1.length, interval=Interval.M5)
    ] == ["AAAUSDT"]
    assert [
        e.order.symbol if isinstance(e, OrderCancelled) else e.fill.symbol
        for e in broker.advance(T0 + h1.length, interval=h1)
    ] == ["BBBUSDT"]
