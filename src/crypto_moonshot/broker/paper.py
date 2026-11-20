"""The paper broker: fills orders against closed candles as they arrive, with the backtest's rules.

`PaperBroker` is what the live engine trades with in paper mode and in replay. It never reads a
clock: the engine gives it candles (`add_candles`) and moves it forward (`advance(until)`), and
every decision uses only candles that have *closed* by `until` (open time + length <= `until`),
so candles handed over early never leak into a decision. Fill prices, quantities and fees are
exactly `SimulatedBroker`'s (see `broker.simulated` for the fill rules):

- **Market buy** for time *t* (aligned to the order's candle interval): cash is reserved at
  placement. It fills at the open of candle *t* once that candle has closed, with the fill time
  *t*. If the first closed candle at or after *t* opens later (a hole at *t*) the buy is
  cancelled with reason `no_entry_candle` and its cash released, as the backtest skips such a
  signal. While no candle at or after *t* has closed, the buy stays pending (a late candle may
  still arrive); the engine decides how long to wait and cancels it with `cancel_order`.
- **Market sell** (the whole position, or a quantity of one without a bracket): fills at the
  open of the first candle at or after its time once that candle has closed (a hole is skipped,
  as in `SimulatedBroker`). Selling the whole position cancels its bracket.
- **Brackets.** A buy with an `ExitPlan` opens a bracket for the bought quantity at its fill.
  On every `advance`, `resolve_exit` runs on the position's closed candles from the entry candle
  (`ended=False`), the same function on a prefix of the window the backtest uses, and its exit
  is applied only once it is certain:

  - a take profit, stop or trailing stop found in the prefix is the first hit of the full
    window too (the scan goes forward and the trailing peak looks only backwards), and its
    candle has closed, so it is certain at once; a gap fill at a candle's open is applied when
    that candle closes, with the open time as its fill time;
  - a `TIME` exit is certain only once every candle that opens before the deadline has closed:
    `until >= entry + ceil(max_hold / interval) x interval` (the deadline itself when it falls
    on a candle boundary). `resolve_exit` returns a time exit for *any* unended window without
    a hit, even one that stops short of the deadline, so a prefix's time exit means nothing
    before that;
  - `series_end` never happens: a live series never ends. A delisted symbol's position stays
    open until the engine sells it.

  When a bracket exit and a pending market sell of the same position are both known, the
  earlier fills; on a tie (a gap at the open of the sell's candle) the bracket fills and the
  sell is cancelled with reason `position_closed`.
- **Events.** `advance` returns `EntryFilled`, `ExitFilled` and `OrderCancelled` events in time
  order across symbols and credits the cash in that order. An event's time is when it happened
  (a gap fill: the candle's open), which can be before `until` but never after it.
- **Recovery.** `snapshot()` and `PaperBroker.restore()` save and rebuild the cash, positions
  (with brackets) and pending orders. After a restore, add each position's candles from its
  entry candle on (`required_history` lists what is missing) and `advance` to now: exits that
  happened meanwhile are applied at their real times.

Candles are kept per (symbol, interval) only while that series has a position or a pending
order, from the earliest time it needs (the entry candle, or the order time); `advance` drops
the rest of the closed candles. Duplicate candles (same open time) keep the first one held.
Orders keep the `Order.id` they were given (the engine's deterministic client order IDs); a
bracket's ID is derived from its buy's (`child_order_id`). One position per symbol.

Known differences from a backtest, all from information the backtest has and a live run does
not: a candle that arrives after the advance that needed it (the broker has already decided
without it), and nothing else in the broker itself (see `docs/architecture.md`, Paper broker).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import StrEnum

import numpy as np

from crypto_moonshot.broker.base import (
    QUOTE_ASSET,
    Costs,
    ExitPlan,
    ExitReason,
    Fill,
    Order,
    OrderRejected,
    OrderType,
    RejectReason,
    Side,
    base_asset,
    child_order_id,
)
from crypto_moonshot.broker.candles import Candles, from_us, to_us
from crypto_moonshot.broker.simulated import ExitFill, resolve_exit
from crypto_moonshot.models import Interval, as_utc

_US = timedelta(microseconds=1)
_TOLERANCE = 1e-9
BRACKET_ROLE = "bracket"


class CancelReason(StrEnum):
    """Why the broker cancelled an order by itself."""

    NO_ENTRY_CANDLE = "no_entry_candle"
    """A buy's candle is missing: the first candle at or after its time opens later."""
    POSITION_CLOSED = "position_closed"
    """A sell's position was closed by its bracket first."""


@dataclass(frozen=True, slots=True)
class PaperPosition:
    """An open long position. `cost` is the USDT spent, entry fee included; `entry_ref` the raw
    open of the entry candle (the exit levels' reference); `interval` the candles its bracket
    is evaluated on."""

    symbol: str
    interval: Interval
    quantity: float
    entry_time: datetime
    entry_ref: float
    entry_price: float
    cost: float
    entry_order_id: str
    exit_plan: ExitPlan | None = None
    bracket_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_time", as_utc(self.entry_time))
        if (self.bracket_id is not None) and self.exit_plan is None:
            msg = "a position with a bracket needs its exit plan"
            raise ValueError(msg)

    def bracket(self) -> Order | None:
        """The open bracket sell as an order, or `None` without one."""
        if self.bracket_id is None or self.exit_plan is None:
            return None
        return Order(
            self.symbol,
            Side.SELL,
            self.entry_time,
            OrderType.BRACKET,
            quantity=self.quantity,
            exit_plan=self.exit_plan,
            id=self.bracket_id,
        )


@dataclass(frozen=True, slots=True)
class PendingOrder:
    """A market order not yet filled, with the candle interval it fills on."""

    order: Order
    interval: Interval


@dataclass(frozen=True, slots=True)
class EntryFilled:
    """A buy filled and opened `position` (with its bracket, if the buy carried an exit plan)."""

    fill: Fill
    position: PaperPosition

    @property
    def time(self) -> datetime:
        return self.fill.time


@dataclass(frozen=True, slots=True)
class ExitFilled:
    """A sell filled: a bracket exit (`fill.reason` set, `fill.order_id` the bracket's ID) or a
    market sell (`fill.reason` None). `position` is the position before the sell, `remaining`
    what is left after a partial sell (`None` once closed)."""

    fill: Fill
    position: PaperPosition
    remaining: PaperPosition | None = None

    @property
    def time(self) -> datetime:
        return self.fill.time

    @property
    def closed(self) -> bool:
        return self.remaining is None

    @property
    def proceeds(self) -> float:
        """USDT returned, exit fee deducted."""
        return self.fill.notional - self.fill.fee


@dataclass(frozen=True, slots=True)
class OrderCancelled:
    """The broker cancelled `order` by itself (`time` is the order's time for a missing entry
    candle, the closing exit's time for a sell)."""

    order: Order
    time: datetime
    reason: CancelReason


BrokerEvent = EntryFilled | ExitFilled | OrderCancelled


@dataclass(frozen=True, slots=True)
class PaperSnapshot:
    """Everything a `PaperBroker` needs to resume: free cash (reserved stakes of pending buys
    already deducted), open positions and pending market orders. Candles are not included."""

    cash: float
    positions: tuple[PaperPosition, ...] = ()
    orders: tuple[PendingOrder, ...] = ()


@dataclass(slots=True)
class _Series:
    """One symbol's closed candles at one interval, keyed by open time (µs)."""

    interval: Interval
    bars: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    cache: Candles | None = None

    @property
    def length_us(self) -> int:
        return self.interval.length // _US

    def add(self, candles: Candles) -> None:
        times = candles.open_time
        for k in range(len(candles)):
            key = int(times[k])
            if key not in self.bars:
                self.bars[key] = (
                    float(candles.open[k]),
                    float(candles.high[k]),
                    float(candles.low[k]),
                    float(candles.close[k]),
                )
                self.cache = None

    def candles(self) -> Candles:
        if self.cache is None:
            times = sorted(self.bars)
            prices = np.array([self.bars[t] for t in times], dtype=np.float64).reshape(-1, 4)
            self.cache = Candles(
                interval=self.interval,
                open_time=np.array(times, dtype=np.int64),
                open=prices[:, 0].copy(),
                high=prices[:, 1].copy(),
                low=prices[:, 2].copy(),
                close=prices[:, 3].copy(),
                ended=False,
            )
        return self.cache

    def closed(self, until_us: int) -> Candles:
        """The candles that have closed by `until_us`, oldest first (`ended=False`)."""
        candles = self.candles()
        hi = int(np.searchsorted(candles.open_time, until_us - self.length_us, side="right"))
        return Candles(
            interval=candles.interval,
            open_time=candles.open_time[:hi],
            open=candles.open[:hi],
            high=candles.high[:hi],
            low=candles.low[:hi],
            close=candles.close[:hi],
            ended=False,
        )

    def drop(self, keep: dict[int, tuple[float, float, float, float]]) -> None:
        if len(keep) != len(self.bars):
            self.bars = keep
            self.cache = None


class PaperBroker:
    """Fills orders against closed candles as they arrive; see the module docstring.

    `interval` is the candle interval orders fill on unless `place_order` is given another.
    `cash` is the free USDT (not reserved by a pending buy).
    """

    def __init__(
        self, cash: float, costs: Costs | None = None, *, interval: Interval = Interval.M5
    ) -> None:
        _ = interval.length  # raises for 1M, which has no fixed length
        self.cash = cash
        self.costs = costs or Costs()
        self.interval = interval
        self._positions: dict[str, PaperPosition] = {}
        self._orders: dict[str, PendingOrder] = {}
        self._series: dict[tuple[str, Interval], _Series] = {}

    # -- state --------------------------------------------------------------------------------

    @property
    def positions(self) -> dict[str, PaperPosition]:
        """Open positions by symbol (a copy)."""
        return dict(self._positions)

    @property
    def pending(self) -> list[PendingOrder]:
        """Pending market orders, in placement order."""
        return list(self._orders.values())

    @property
    def reserved(self) -> float:
        """USDT reserved by pending buys."""
        return sum(
            p.order.quote_amount or 0.0 for p in self._orders.values() if p.order.side is Side.BUY
        )

    def snapshot(self) -> PaperSnapshot:
        return PaperSnapshot(self.cash, tuple(self._positions.values()), tuple(self.pending))

    @classmethod
    def restore(
        cls,
        snapshot: PaperSnapshot,
        costs: Costs | None = None,
        *,
        interval: Interval = Interval.M5,
    ) -> PaperBroker:
        """A broker in the state `snapshot` describes, holding no candles yet."""
        broker = cls(snapshot.cash, costs, interval=interval)
        for position in snapshot.positions:
            if position.symbol in broker._positions:
                msg = f"snapshot holds {position.symbol} twice"
                raise ValueError(msg)
            broker._positions[position.symbol] = position
        for pending in snapshot.orders:
            order = pending.order
            if order.id in broker._orders:
                msg = f"snapshot has order {order.id} twice"
                raise ValueError(msg)
            if order.side is Side.SELL and order.symbol not in broker._positions:
                msg = f"snapshot has a sell of {order.symbol} without a position"
                raise ValueError(msg)
            broker._orders[order.id] = pending
        return broker

    def required_history(self) -> list[tuple[str, Interval, datetime]]:
        """(symbol, interval, entry time) of each open position whose entry candle the broker
        does not hold: add its candles from that time on before advancing."""
        out = []
        for position in self._positions.values():
            series = self._series.get((position.symbol, position.interval))
            if series is None or to_us(position.entry_time) not in series.bars:
                out.append((position.symbol, position.interval, position.entry_time))
        return out

    # -- Broker protocol ----------------------------------------------------------------------

    def place_order(self, order: Order, *, interval: Interval | None = None) -> None:
        """Accept a market order; it fills in a later `advance`, so this returns `None`.

        Raises `OrderRejected` when the order cannot be accepted: not enough free cash, the
        symbol already held or with a buy pending, a sell without a position or with another
        sell pending, a partial sell of a position with an open bracket, or a reused order ID.
        """
        if order.type is OrderType.BRACKET:
            msg = "bracket orders are opened by a buy with an exit plan"
            raise ValueError(msg)
        if order.id in self._orders or any(
            p.bracket_id == order.id for p in self._positions.values()
        ):
            raise OrderRejected(RejectReason.DUPLICATE_ID, f"order {order.id} already exists")
        if order.side is Side.BUY:
            self._accept_buy(order, interval or self.interval)
        else:
            self._accept_sell(order, interval)

    def cancel_order(self, order_id: str) -> None:
        """Cancel a pending market order (a buy's reserved cash is released) or an open bracket
        (its position stays open, without exits). Raises `KeyError` for no such open order."""
        pending = self._orders.pop(order_id, None)
        if pending is not None:
            if pending.order.side is Side.BUY:
                self.cash += pending.order.quote_amount or 0.0
            return
        for symbol, position in self._positions.items():
            if position.bracket_id == order_id:
                self._positions[symbol] = replace(position, bracket_id=None)
                return
        raise KeyError(order_id)

    def balances(self) -> dict[str, float]:
        """Free USDT (reserved stakes excluded) and each base asset held."""
        out = {QUOTE_ASSET: self.cash}
        for symbol, position in self._positions.items():
            out[base_asset(symbol)] = position.quantity
        return out

    def open_orders(self) -> list[Order]:
        """Pending market orders, then open brackets."""
        brackets = [p.bracket() for p in self._positions.values()]
        return [p.order for p in self._orders.values()] + [b for b in brackets if b is not None]

    # -- candles and time ---------------------------------------------------------------------

    def add_candles(self, symbol: str, candles: Candles) -> None:
        """Merge closed candles of `symbol` (any order; an open time already held is ignored).

        Candles of series without a position or pending order are dropped by the next
        `advance` once closed, so the engine may hand over every symbol's candles.
        """
        if len(candles) == 0:
            return
        interval = candles.interval
        length = interval.length // _US
        first = int(candles.open_time[0])
        aligned = to_us(interval.floor(from_us(first))) == first
        if not aligned or bool(np.any((candles.open_time - first) % length)):
            msg = f"{symbol} candle open times must be aligned to {interval.value}"
            raise ValueError(msg)
        series = self._series.setdefault((symbol, interval), _Series(interval))
        series.add(candles)

    def advance(self, until: datetime) -> list[BrokerEvent]:
        """Apply everything the closed candles decide up to `until`; returns the events in time
        order (cash credited in that order). Candles closing after `until` are not used."""
        until_us = to_us(until)
        events: list[BrokerEvent] = []
        symbols = {p.symbol for p in self._positions.values()}
        symbols.update(p.order.symbol for p in self._orders.values())
        for symbol in sorted(symbols):
            events.extend(self._advance_symbol(symbol, until_us))
        events.sort(key=operator.attrgetter("time"))  # stable: per-symbol order is kept
        for event in events:
            if isinstance(event, ExitFilled):
                self.cash += event.fill.quantity * event.fill.price - event.fill.fee
            elif isinstance(event, OrderCancelled) and event.order.side is Side.BUY:
                self.cash += event.order.quote_amount or 0.0
        self._prune(until_us)
        return events

    # -- internals ----------------------------------------------------------------------------

    def _pending(self, symbol: str, side: Side) -> PendingOrder | None:
        for pending in self._orders.values():
            if pending.order.symbol == symbol and pending.order.side is side:
                return pending
        return None

    def _accept_buy(self, order: Order, interval: Interval) -> None:
        assert order.quote_amount is not None
        _ = interval.length
        if not interval.is_aligned(order.time):
            msg = f"a buy's time must be a {interval.value} candle open, got {order.time}"
            raise ValueError(msg)
        if order.symbol in self._positions or self._pending(order.symbol, Side.BUY):
            raise OrderRejected(RejectReason.SYMBOL_HELD, f"already holding {order.symbol}")
        if order.quote_amount > self.cash * (1 + _TOLERANCE):
            msg = f"not enough cash: {order.quote_amount:.2f} > {self.cash:.2f}"
            raise OrderRejected(RejectReason.CASH, msg)
        self.cash = max(self.cash - order.quote_amount, 0.0)
        self._orders[order.id] = PendingOrder(order, interval)

    def _accept_sell(self, order: Order, interval: Interval | None) -> None:
        position = self._positions.get(order.symbol)
        if position is None:
            raise OrderRejected(RejectReason.NOT_HOLDING, f"not holding {order.symbol}")
        if interval is not None and interval is not position.interval:
            msg = f"{order.symbol} is held on {position.interval.value} candles, not {interval}"
            raise ValueError(msg)
        if self._pending(order.symbol, Side.SELL) is not None:
            msg = f"a sell of {order.symbol} is already pending"
            raise OrderRejected(RejectReason.SELL_PENDING, msg)
        quantity = position.quantity if order.quantity is None else order.quantity
        if quantity > position.quantity * (1 + _TOLERANCE):
            msg = f"cannot sell {quantity} {order.symbol}, holding {position.quantity}"
            raise OrderRejected(RejectReason.QUANTITY, msg)
        whole = quantity >= position.quantity * (1 - _TOLERANCE)
        if position.bracket_id is not None and not whole:
            msg = "cancel the bracket before selling part of a position"
            raise OrderRejected(RejectReason.BRACKET_OPEN, msg)
        self._orders[order.id] = PendingOrder(order, position.interval)

    def _closed(self, symbol: str, interval: Interval, until_us: int) -> Candles | None:
        series = self._series.get((symbol, interval))
        if series is None or not series.bars:
            return None
        return series.closed(until_us)

    def _advance_symbol(self, symbol: str, until_us: int) -> list[BrokerEvent]:
        out: list[BrokerEvent] = []
        while True:
            buy = self._pending(symbol, Side.BUY)
            if buy is not None:
                event = self._try_buy(buy, until_us)
                if event is None:
                    break
                out.append(event)
                continue
            position = self._positions.get(symbol)
            if position is None:
                break
            closed = self._closed(symbol, position.interval, until_us)
            if closed is None or len(closed) == 0:
                break
            exit_ = self._certain_exit(position, closed, until_us)
            sell = self._pending(symbol, Side.SELL)
            sell_at: tuple[int, float] | None = None
            if sell is not None:
                i = closed.index_at_or_after(sell.order.time)
                if i < len(closed):
                    sell_at = (int(closed.open_time[i]), float(closed.open[i]))
            if exit_ is not None and (sell_at is None or to_us(exit_.time) <= sell_at[0]):
                out.append(self._fill_exit(position, exit_))
                if sell is not None:
                    del self._orders[sell.order.id]
                    out.append(OrderCancelled(sell.order, exit_.time, CancelReason.POSITION_CLOSED))
                continue
            if sell is not None and sell_at is not None:
                out.append(self._fill_sell(position, sell, *sell_at))
                continue
            break
        return out

    def _try_buy(self, buy: PendingOrder, until_us: int) -> BrokerEvent | None:
        order = buy.order
        closed = self._closed(order.symbol, buy.interval, until_us)
        if closed is None:
            return None
        i = closed.index_at_or_after(order.time)
        if i == len(closed):
            return None
        del self._orders[order.id]
        if int(closed.open_time[i]) != to_us(order.time):
            return OrderCancelled(order, order.time, CancelReason.NO_ENTRY_CANDLE)
        assert order.quote_amount is not None
        ref = float(closed.open[i])
        price = self.costs.buy_price(ref)
        quantity = order.quote_amount / (price * (1 + self.costs.fee))
        fill = Fill(
            order.id,
            order.symbol,
            Side.BUY,
            order.time,
            price,
            quantity,
            fee=order.quote_amount - quantity * price,
        )
        plan = order.exit_plan
        position = PaperPosition(
            symbol=order.symbol,
            interval=buy.interval,
            quantity=quantity,
            entry_time=order.time,
            entry_ref=ref,
            entry_price=price,
            cost=order.quote_amount,
            entry_order_id=order.id,
            exit_plan=plan,
            bracket_id=None if plan is None else child_order_id(order.id, BRACKET_ROLE),
        )
        self._positions[order.symbol] = position
        return EntryFilled(fill, position)

    def _certain_exit(
        self, position: PaperPosition, closed: Candles, until_us: int
    ) -> ExitFill | None:
        """The bracket's exit if the closed candles already decide it, else `None`."""
        plan = position.exit_plan
        if position.bracket_id is None or plan is None:
            return None
        entry_us = to_us(position.entry_time)
        window = closed.window(position.entry_time)
        if len(window) == 0 or int(window.open_time[0]) != entry_us:
            return None  # the entry candle is not held (see `required_history`)
        found = resolve_exit(window, plan, self.costs)
        if found is None:
            return None
        if found.reason is ExitReason.TIME:
            assert plan.max_hold is not None
            if until_us < time_exit_known_us(entry_us, plan.max_hold, window.length_us):
                return None
        elif found.reason is ExitReason.SERIES_END:  # pragma: no cover - windows never end
            msg = "a paper position's candle window never ends"
            raise AssertionError(msg)
        return found

    def _fill_exit(self, position: PaperPosition, exit_: ExitFill) -> ExitFilled:
        assert position.bracket_id is not None
        quantity = position.quantity
        fill = Fill(
            position.bracket_id,
            position.symbol,
            Side.SELL,
            exit_.time,
            exit_.price,
            quantity,
            quantity * exit_.price * self.costs.fee,
            exit_.reason,
        )
        del self._positions[position.symbol]
        return ExitFilled(fill, position)

    def _fill_sell(
        self, position: PaperPosition, sell: PendingOrder, open_us: int, raw: float
    ) -> ExitFilled:
        del self._orders[sell.order.id]
        price = self.costs.sell_price(raw, market=True)
        quantity = position.quantity if sell.order.quantity is None else sell.order.quantity
        remaining: PaperPosition | None = None
        if quantity >= position.quantity * (1 - _TOLERANCE):
            quantity = position.quantity
            del self._positions[position.symbol]
        else:
            remaining = replace(
                position,
                quantity=position.quantity - quantity,
                cost=position.cost * (1 - quantity / position.quantity),
            )
            self._positions[position.symbol] = remaining
        fee = quantity * price * self.costs.fee
        fill = Fill(
            sell.order.id, position.symbol, Side.SELL, from_us(open_us), price, quantity, fee
        )
        return ExitFilled(fill, position, remaining)

    def _prune(self, until_us: int) -> None:
        need: dict[tuple[str, Interval], int] = {}
        for position in self._positions.values():
            key, start = (position.symbol, position.interval), to_us(position.entry_time)
            need[key] = min(need.get(key, start), start)
        for pending in self._orders.values():
            key, start = (pending.order.symbol, pending.interval), to_us(pending.order.time)
            need[key] = min(need.get(key, start), start)
        for key, series in list(self._series.items()):
            if key in need:
                start = need[key]
                series.drop({t: bar for t, bar in series.bars.items() if t >= start})
            else:
                closes = until_us - series.length_us
                series.drop({t: bar for t, bar in series.bars.items() if t > closes})
                if not series.bars:
                    del self._series[key]


def time_exit_known_us(entry_us: int, max_hold: timedelta, length_us: int) -> int:
    """When a time exit becomes certain (µs): the close of the last candle that opens before
    the deadline, `entry + ceil(max_hold / length) x length`."""
    hold_us = max_hold // _US
    return entry_us + -(-hold_us // length_us) * length_us
