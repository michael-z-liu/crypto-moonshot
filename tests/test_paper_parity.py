"""Parity of the incremental `PaperBroker` with `SimulatedBroker` over random candle paths.

`SimulatedBroker` gets each position's whole window at once (as the backtest does); the paper
broker gets the same candles one at a time, in bursts, out of order or ahead of time, and is
advanced as time passes. Its fills must be identical, and each exit must be applied at the first
advance that can know it (the close of the hit candle, or of the last candle before a time
exit's deadline), never earlier.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pytest
from broker_fakes import T0, Bar, candles

from crypto_moonshot.broker.base import Costs, ExitPlan, ExitReason, Fill, Order, Side
from crypto_moonshot.broker.paper import (
    BrokerEvent,
    CancelReason,
    EntryFilled,
    ExitFilled,
    OrderCancelled,
    PaperBroker,
)
from crypto_moonshot.broker.simulated import SimulatedBroker

M5 = timedelta(minutes=5)
COSTS = Costs(fee=0.001, slippage=0.0005, latency=0.0002)
STAKE = 100.0
SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")


@dataclass(frozen=True)
class Case:
    symbol: str
    bars: list[Bar | None]
    entry: int
    plan: ExitPlan

    @property
    def entry_time(self) -> datetime:
        return T0 + self.entry * M5

    def order(self) -> Order:
        return Order(
            self.symbol,
            Side.BUY,
            self.entry_time,
            quote_amount=STAKE,
            exit_plan=self.plan,
            id=f"{self.symbol}-buy",
        )


def random_bars(rng: np.random.Generator, n: int) -> list[Bar | None]:
    """A random walk with gaps through levels, holes and wide candles touching both levels."""
    bars: list[Bar | None] = []
    price = 100.0
    for _ in range(n):
        if rng.random() < 0.07:
            bars.append(None)
            continue
        jump = rng.normal(0, 0.05) if rng.random() < 0.08 else rng.normal(0, 0.003)
        o = price * float(np.exp(jump))
        c = o * float(np.exp(rng.normal(0, 0.015)))
        hi = max(o, c) * float(np.exp(abs(rng.normal(0, 0.01))))
        lo = min(o, c) * float(np.exp(-abs(rng.normal(0, 0.01))))
        if rng.random() < 0.05:
            hi, lo = hi * 1.2, lo * 0.8
        bars.append((o, hi, lo, c))
        price = c
    return bars


def random_plan(rng: np.random.Generator) -> ExitPlan:
    while True:
        hold = None
        if rng.random() < 0.7:
            # Mostly whole candles; some deadlines fall inside a candle (e.g. 37 minutes).
            minutes = int(rng.integers(1, 40)) * 5
            if rng.random() < 0.25:
                minutes += int(rng.integers(1, 5))
            hold = timedelta(minutes=minutes)
        plan = ExitPlan(
            take_profit=float(rng.uniform(0.01, 0.15)) if rng.random() < 0.7 else None,
            stop=float(rng.uniform(0.01, 0.1)) if rng.random() < 0.6 else None,
            trailing=float(rng.uniform(0.01, 0.08)) if rng.random() < 0.35 else None,
            max_hold=hold,
            tp_fill="close" if rng.random() < 0.3 else "touch",
        )
        if any(v is not None for v in (plan.take_profit, plan.stop, plan.trailing, hold)):
            return plan


def random_case(rng: np.random.Generator, symbol: str, *, hole_at_entry: bool = False) -> Case:
    plan = random_plan(rng)
    entry = int(rng.integers(0, 6))
    if plan.max_hold is not None:
        # The whole window must reach past the deadline, as the backtest's windows do.
        span = -(-plan.max_hold // M5) + int(rng.integers(1, 8))
    else:
        span = int(rng.integers(5, 60))
    bars = random_bars(rng, entry + span)
    if not hole_at_entry and bars[entry] is None:
        bars[entry] = (100.0, 100.5, 99.5, 100.0)
    if hole_at_entry:
        bars[entry] = None
        bars[-1] = bars[-1] or (100.0, 100.0, 100.0, 100.0)
    return Case(symbol, bars, entry, plan)


def reference(case: Case) -> list[Fill]:
    """`SimulatedBroker` with the whole window at once (not ended: the series goes on)."""
    broker = SimulatedBroker(cash=10 * STAKE, costs=COSTS)
    broker.set_candles(case.symbol, candles(case.bars, ended=False))
    entry = broker.place_order(case.order())
    assert entry is not None
    return [entry, *broker.settle(datetime.max.replace(tzinfo=T0.tzinfo))]


def known_at(case: Case, fill: Fill) -> datetime:
    """The earliest `until` at which a fill can be known."""
    if fill.side is Side.BUY:
        return fill.time + M5
    if fill.reason is ExitReason.TIME:
        assert case.plan.max_hold is not None
        # The close of the last candle that opens before the deadline.
        return case.entry_time + -(-case.plan.max_hold // M5) * M5
    k = (fill.time - T0) // M5
    bar = case.bars[k] if (fill.time - T0) % M5 == timedelta(0) and k < len(case.bars) else None
    market = fill.reason is not ExitReason.TAKE_PROFIT
    gap = bar is not None and fill.price == COSTS.sell_price(bar[0], market=market)
    return fill.time + M5 if gap else fill.time


def comparable(fill: Fill) -> tuple[object, ...]:
    return (fill.symbol, fill.side, fill.time, fill.price, fill.quantity, fill.fee, fill.reason)


def fills(events: list[BrokerEvent]) -> list[Fill]:
    return [e.fill for e in events if isinstance(e, EntryFilled | ExitFilled)]


Schedule = list[tuple[datetime, list[int]]]
"""Steps of (advance to this time, after adding the candles of these slots)."""


def one_at_a_time(n: int) -> Schedule:
    return [(T0 + (k + 1) * M5, [k]) for k in range(n)]


def in_bursts(rng: np.random.Generator, n: int) -> Schedule:
    """Random chunks of slots, each added in shuffled order, advancing at the chunk's end."""
    out: Schedule = []
    k = 0
    while k < n:
        size = int(rng.integers(1, 12))
        chunk = list(range(k, min(k + size, n)))
        rng.shuffle(chunk)
        k += size
        out.append((T0 + min(k, n) * M5, chunk))
    return out


def ahead_of_time(rng: np.random.Generator, n: int) -> Schedule:
    """Chunks added up front, then advanced slot by slot: candles not yet closed at an advance
    must be ignored."""
    out: Schedule = []
    k = 0
    while k < n:
        size = int(rng.integers(1, 12))
        chunk = list(range(k, min(k + size, n)))
        rng.shuffle(chunk)
        out.append((T0 + (k + 1) * M5, chunk))
        out.extend((T0 + (j + 1) * M5, []) for j in range(k + 1, min(k + size, n)))
        k += size
    return out


def late_arrivals(rng: np.random.Generator, n: int) -> Schedule:
    """One slot at a time, but some candles arrive a few steps late, still before the advance
    that needs them (the steps in between do not move time forward)."""
    out: Schedule = []
    k = 0
    while k < n:
        late = min(int(rng.integers(1, 4)), n - k)
        slots = list(range(k, k + late))
        until = T0 + k * M5 if k else T0
        for slot in reversed(slots):  # newest first: the older ones are late
            out.append((until, [slot]))
        k += late
        out.append((T0 + k * M5, []))
    return out


def irregular_advances(rng: np.random.Generator, n: int) -> Schedule:
    """Each candle handed over before it closes, with advances at random times inside candles
    (as when another interval's ticks drive the broker), plus one at each close."""
    out: Schedule = []
    for k in range(n):
        start = T0 + k * M5
        out.append((start, [k]))
        for minute in sorted(rng.uniform(0.1, 4.9, int(rng.integers(0, 3)))):
            out.append((start + timedelta(minutes=float(minute)), []))
        out.append((start + M5, []))
    return out


def run_paper(
    cases: list[Case], schedule: Schedule, n: int
) -> tuple[PaperBroker, list[tuple[datetime, BrokerEvent]]]:
    broker = PaperBroker(cash=len(cases) * STAKE, costs=COSTS)
    for case in cases:
        broker.place_order(case.order())
    seen: list[tuple[datetime, BrokerEvent]] = []
    for until, slots in schedule:
        for slot in slots:
            for case in cases:
                if slot < len(case.bars) and case.bars[slot] is not None:
                    bar = case.bars[slot]
                    broker.add_candles(case.symbol, candles([bar], start=T0 + slot * M5))
        events = broker.advance(until)
        times = [e.time for e in events]
        assert times == sorted(times)
        assert all(t <= until for t in times)
        seen.extend((until, e) for e in events)
    assert schedule[-1][0] == T0 + n * M5
    return broker, seen


def symbol_of(event: BrokerEvent) -> str:
    return event.order.symbol if isinstance(event, OrderCancelled) else event.fill.symbol


def check(
    cases: list[Case],
    broker: PaperBroker,
    seen: list[tuple[datetime, BrokerEvent]],
    schedule: Schedule,
) -> None:
    untils = sorted({u for u, _ in schedule})
    expected_cash = len(cases) * STAKE
    for case in cases:
        mine = [(u, e) for u, e in seen if symbol_of(e) == case.symbol]
        if case.bars[case.entry] is None:
            [(_, event)] = mine
            assert isinstance(event, OrderCancelled)
            assert (event.reason, event.time) == (CancelReason.NO_ENTRY_CANDLE, case.entry_time)
            continue
        ref = reference(case)
        got = [(u, e.fill) for u, e in mine if isinstance(e, EntryFilled | ExitFilled)]
        assert [comparable(f) for _, f in got] == [comparable(f) for f in ref], case
        for until, fill in got:
            # Applied at the first advance that can know it: never earlier, never later.
            first = min(u for u in untils if u >= known_at(case, fill))
            assert until == first, (case, fill)
        expected_cash -= STAKE
        if len(ref) == 2:
            expected_cash += ref[1].notional - ref[1].fee
        else:
            assert case.symbol in broker.positions  # no exit yet: still open
    assert broker.cash == pytest.approx(expected_cash, rel=1e-12)


@pytest.mark.parametrize("seed", range(40))
def test_one_candle_at_a_time_matches_the_simulated_broker(seed: int) -> None:
    rng = np.random.default_rng(seed)
    cases = [random_case(rng, symbol, hole_at_entry=rng.random() < 0.1) for symbol in SYMBOLS]
    n = max(len(c.bars) for c in cases)
    schedule = one_at_a_time(n)
    broker, seen = run_paper(cases, schedule, n)
    check(cases, broker, seen, schedule)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("make", [in_bursts, ahead_of_time, late_arrivals, irregular_advances])
def test_bursts_and_out_of_order_candles_give_the_same_fills(
    seed: int, make: Callable[[np.random.Generator, int], Schedule]
) -> None:
    rng = np.random.default_rng(1000 + seed)
    cases = [random_case(rng, symbol, hole_at_entry=rng.random() < 0.1) for symbol in SYMBOLS]
    n = max(len(c.bars) for c in cases)
    schedule = make(rng, n)
    broker, seen = run_paper(cases, schedule, n)
    check(cases, broker, seen, schedule)


def test_the_random_cases_cover_every_exit_kind() -> None:
    reasons: set[ExitReason | None] = set()
    gaps = unresolved = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        for symbol in SYMBOLS:
            case = random_case(rng, symbol)
            ref = reference(case)
            if len(ref) == 1:
                unresolved += 1
                continue
            reasons.add(ref[1].reason)
            gaps += known_at(case, ref[1]) != ref[1].time and ref[1].reason is not ExitReason.TIME
    assert reasons >= {
        ExitReason.TAKE_PROFIT,
        ExitReason.STOP,
        ExitReason.TRAILING_STOP,
        ExitReason.TIME,
    }
    assert gaps > 0
    assert unresolved > 0
