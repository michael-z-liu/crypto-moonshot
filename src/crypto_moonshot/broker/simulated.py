"""Simulated fills against candles: the exit resolver and the `SimulatedBroker`.

`resolve_exit` is the core. It simulates one position's bracket exits over its whole candle
window in a single vectorised numpy pass. The rules:

1. Entry is a market buy at the open of the window's first candle (the entry candle). That
   open, before costs, is the *reference price* every exit level is relative to. The entry
   candle can itself trigger exits (bar 0 of the research's forward paths).
2. Take profit at `target = ref * (1 + take_profit)`. `touch`: hit when a candle's high
   reaches the target. `close`: hit only when a candle closes at or above it. Fills at the
   target; if the candle *opens* at or above it (a gap), at the open.
3. Stop at `ref * (1 - stop)`: hit when a candle's low reaches it; fills at the level, or at the
   open if the candle opens at or below it.
4. Trailing stop at `peak * (1 - trailing)`, `peak` being the highest high of the *earlier*
   candles since entry, starting from `ref`; the candle that triggers it never lifts it. Fills
   as a stop. When a fixed stop and a trailing stop are both active, the higher level is the
   effective one (falling prices cross it first); a tie counts as `stop`.
5. When a stop-like level and the take profit are both hit in one candle, the stop fills first
   (we cannot know the order inside a candle), unless the candle opens beyond the target, which
   fills the take profit at the open. A gap through the stop always fills the stop at the open.
6. Time exit at `deadline = entry time + max_hold`: if nothing else filled, sell at the close of
   the last candle that opens before the deadline; the exit time is the deadline. For a 1h hold
   on 5m candles that is the close of the 12th candle.
7. Multi-resolution: with `fine` candles (e.g. 1m under 5m), each coarse candle fully covered
   by fine candles (all of them present) is replaced by them, the others are kept, and every
   rule above runs on that mixed path. Same-candle ambiguity is then resolved by the fine
   order (still stop-first within one fine candle).
8. Holes and ended series: no fills happen inside a hole; the first candle after it is checked
   normally, so its open applies the gap rules. A deadline inside a hole fills at the last close
   before the hole (missing candles carry the last close), at the deadline. A series that ends
   (`Candles.ended`) before any exit closes at its last close, reason `series_end`, at that
   candle's close time.

Exit times: a fill at a candle's open (gap) happens at its open time; a fill inside a candle
(a level touched, or a close) at its close time, the first moment the simulation knows of it.

Costs are applied by `Costs` (see its docstring): stops, trailing stops, time and series-end
exits are market orders (slippage + latency), a take profit is a limit (slippage only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import inf

import numpy as np

from crypto_moonshot.broker.base import (
    QUOTE_ASSET,
    Costs,
    ExitPlan,
    ExitReason,
    Fill,
    Order,
    OrderType,
    Side,
    base_asset,
)
from crypto_moonshot.broker.candles import Candles, Prices, Times, from_us, to_us

_US = timedelta(microseconds=1)


@dataclass(frozen=True, slots=True)
class FillPath:
    """The candles a position's exits are checked against, in time order, possibly of mixed
    lengths (coarse candles with fine ones in their place where available)."""

    open_time: Times
    close_time: Times
    open: Prices
    high: Prices
    low: Prices
    close: Prices

    def __len__(self) -> int:
        return len(self.open_time)


def fill_path(candles: Candles, fine: Candles | None = None) -> FillPath:
    """Replace each candle of `candles` fully covered by `fine` candles with those candles.

    `fine` may be any slice of the finer series; its interval must divide the coarse one.
    """
    length = candles.length_us
    coarse = FillPath(
        open_time=candles.open_time,
        close_time=candles.open_time + length,
        open=candles.open,
        high=candles.high,
        low=candles.low,
        close=candles.close,
    )
    if fine is None or len(fine) == 0 or len(candles) == 0:
        return coarse
    fine_length = fine.length_us
    if fine_length >= length or length % fine_length:
        msg = f"fine interval {fine.interval} must evenly divide {candles.interval}"
        raise ValueError(msg)
    ratio = length // fine_length
    starts = np.searchsorted(fine.open_time, candles.open_time, side="left")
    ends = np.searchsorted(fine.open_time, candles.open_time + length, side="left")
    # Fine open times are aligned and unique, so `ratio` of them means full coverage.
    full = (ends - starts) == ratio
    if not full.any():
        return coarse
    counts = np.where(full, ratio, 1)
    owner = np.repeat(np.arange(len(candles)), counts)
    offsets = np.cumsum(counts) - counts
    is_fine = full[owner]
    fine_idx = np.where(is_fine, starts[owner] + np.arange(len(owner)) - offsets[owner], 0)

    def pick(f: Prices, c: Prices) -> Prices:
        return np.where(is_fine, f[fine_idx], c[owner])

    open_time = np.where(is_fine, fine.open_time[fine_idx], candles.open_time[owner])
    return FillPath(
        open_time=open_time,
        close_time=open_time + np.where(is_fine, fine_length, length),
        open=pick(fine.open, candles.open),
        high=pick(fine.high, candles.high),
        low=pick(fine.low, candles.low),
        close=pick(fine.close, candles.close),
    )


@dataclass(frozen=True, slots=True)
class ExitFill:
    """How and when a position exits.

    `raw_price` is the exit price before costs (a level, an open or a close); `price` the fill
    price after slippage and latency. `entry_ref` is the raw entry open and `entry_price` the
    buy fill price; `net_return` is the round trip after both fees (see `Costs.net_return`).
    """

    time: datetime
    reason: ExitReason
    raw_price: float
    price: float
    entry_ref: float
    entry_price: float
    net_return: float


def _first(hit: np.ndarray) -> int:
    """Index of the first true element, or `len(hit)` if none."""
    if len(hit) == 0:
        return 0
    i = int(hit.argmax())
    return i if hit[i] else len(hit)


def resolve_exit(
    candles: Candles,
    plan: ExitPlan,
    costs: Costs | None = None,
    *,
    fine: Candles | None = None,
) -> ExitFill | None:
    """Simulate a position entered at the open of `candles[0]` until its first exit.

    `candles` must start at the entry candle and hold every candle up to the time exit (or to
    the end of the series). Returns `None` only when no exit happens within the window and the
    series has not ended (`candles.ended` is false), i.e. the window is too short to know. See
    the module docstring for the fill rules.
    """
    if len(candles) == 0:
        msg = "resolve_exit needs at least the entry candle"
        raise ValueError(msg)
    costs = costs or Costs()
    ref = float(candles.open[0])
    entry_us = int(candles.open_time[0])
    entry_price = costs.buy_price(ref)
    path = fill_path(candles, fine)
    n = len(path)

    deadline: int | None = None
    m = n
    if plan.max_hold is not None:
        deadline = entry_us + plan.max_hold // _US
        m = int(np.searchsorted(path.open_time, deadline, side="left"))
    o, h, lo, c = path.open[:m], path.high[:m], path.low[:m], path.close[:m]

    i_stop, stop_level, fixed_stop = m, np.empty(0), -inf
    if plan.stop is not None or plan.trailing is not None:
        if plan.stop is not None:
            fixed_stop = ref * (1 - plan.stop)
        stop_level = np.full(m, fixed_stop)
        if plan.trailing is not None:
            peak = np.maximum.accumulate(np.concatenate(([ref], h[:-1])))
            stop_level = np.maximum(stop_level, peak * (1 - plan.trailing))
        i_stop = _first(lo <= stop_level)

    i_tp, target = m, inf
    if plan.take_profit is not None:
        target = ref * (1 + plan.take_profit)
        tp_hit = h >= target if plan.tp_fill == "touch" else (c >= target) | (o >= target)
        i_tp = _first(tp_hit)

    def done(time_us: int, reason: ExitReason, raw: float, *, market: bool) -> ExitFill:
        price = costs.sell_price(raw, market=market)
        return ExitFill(
            time=from_us(time_us),
            reason=reason,
            raw_price=raw,
            price=price,
            entry_ref=ref,
            entry_price=entry_price,
            net_return=costs.net_return(entry_price, price),
        )

    # In one candle the stop wins, unless the candle opened beyond the target (and not through
    # the stop, which is possible only with `close` fills, where the stop can exceed the target).
    if i_tp < i_stop or (i_tp == i_stop < m and target <= o[i_tp] > stop_level[i_tp]):
        tp_gap = bool(o[i_tp] >= target)
        raw = float(o[i_tp]) if tp_gap else target
        when = int(path.open_time[i_tp] if tp_gap else path.close_time[i_tp])
        return done(when, ExitReason.TAKE_PROFIT, raw, market=False)
    if i_stop < m:
        i = i_stop
        level = float(stop_level[i])
        gap = o[i] <= level
        reason = ExitReason.TRAILING_STOP if level > fixed_stop else ExitReason.STOP
        raw = float(o[i]) if gap else level
        when = int(path.open_time[i] if gap else path.close_time[i])
        return done(when, reason, raw, market=True)

    if deadline is not None and (m < n or not candles.ended or path.close_time[n - 1] >= deadline):
        return done(deadline, ExitReason.TIME, float(c[m - 1]), market=True)
    if candles.ended:
        return done(
            int(path.close_time[n - 1]),
            ExitReason.SERIES_END,
            float(path.close[n - 1]),
            market=True,
        )
    return None


@dataclass(slots=True)
class Position:
    """An open long position. `cost` is the USDT spent, entry fee included."""

    symbol: str
    quantity: float
    entry_time: datetime
    entry_ref: float
    entry_price: float
    cost: float
    bracket_id: str | None = None


@dataclass(slots=True)
class _Bracket:
    order: Order
    exit: ExitFill | None


@dataclass(slots=True)
class _Market:
    candles: Candles
    fine: Candles | None = None


@dataclass
class SimulatedBroker:
    """A broker that fills orders against candles given to it with `set_candles`.

    - A market buy fills at the open of the first candle at or after the order time, plus costs.
      If it carries an `ExitPlan`, a bracket sell for the bought quantity is opened and its exit
      resolved at once with `resolve_exit`; the exit is applied (cash credited, holding removed)
      only when the broker is settled to its time, so balances never show the future.
    - A market sell fills at the open of the first candle at or after the order time; selling a
      whole holding cancels its bracket.
    - `settle(until)` applies every bracket exit due by `until`; every `place_order` first
      settles to the order's time.

    One position per symbol. The engine drives time; nothing here looks at a clock.
    """

    cash: float
    costs: Costs = field(default_factory=Costs)
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    _brackets: dict[str, _Bracket] = field(default_factory=dict)
    _markets: dict[str, _Market] = field(default_factory=dict)

    def set_candles(self, symbol: str, candles: Candles, fine: Candles | None = None) -> None:
        """Give the broker a symbol's candles (and optionally finer ones for exit fills).

        Replaces earlier candles for the symbol. A bracket of that symbol whose exit was still
        unknown (its window ran out before the series ended) is resolved again against them.
        """
        self._markets[symbol] = _Market(candles, fine)
        for bracket in self._brackets.values():
            if bracket.order.symbol == symbol and bracket.exit is None:
                bracket.exit = self._resolve(bracket.order)

    def place_order(self, order: Order) -> Fill | None:
        self.settle(order.time)
        if order.type is OrderType.BRACKET:
            msg = "bracket orders are opened by a buy with an exit plan"
            raise ValueError(msg)
        if order.side is Side.BUY:
            return self._buy(order)
        return self._sell(order)

    def cancel_order(self, order_id: str) -> None:
        bracket = self._brackets.pop(order_id)
        self.positions[bracket.order.symbol].bracket_id = None

    def balances(self) -> dict[str, float]:
        out = {QUOTE_ASSET: self.cash}
        for symbol, position in self.positions.items():
            out[base_asset(symbol)] = position.quantity
        return out

    def open_orders(self) -> list[Order]:
        return [b.order for b in self._brackets.values()]

    def settle(self, until: datetime) -> list[Fill]:
        """Apply every bracket exit due at or before `until`, in time order."""
        due = sorted(
            (b for b in self._brackets.values() if b.exit is not None and b.exit.time <= until),
            key=lambda b: b.exit.time if b.exit else until,
        )
        fills = []
        for bracket in due:
            assert bracket.exit is not None
            order = bracket.order
            del self._brackets[order.id]
            quantity = self.positions.pop(order.symbol).quantity
            fills.append(
                self._credit(
                    order, quantity, bracket.exit.price, bracket.exit.time, bracket.exit.reason
                )
            )
        return fills

    def _market(self, symbol: str) -> _Market:
        try:
            return self._markets[symbol]
        except KeyError:
            msg = f"no candles for {symbol}; call set_candles first"
            raise ValueError(msg) from None

    def _open_at(self, order: Order) -> tuple[datetime, float]:
        candles = self._market(order.symbol).candles
        i = candles.index_at_or_after(order.time)
        if i == len(candles):
            msg = f"no {order.symbol} candle at or after {order.time:%Y-%m-%d %H:%M}"
            raise ValueError(msg)
        return from_us(int(candles.open_time[i])), float(candles.open[i])

    def _buy(self, order: Order) -> Fill:
        assert order.quote_amount is not None
        if order.symbol in self.positions:
            msg = f"already holding {order.symbol}"
            raise ValueError(msg)
        if order.quote_amount > self.cash * (1 + 1e-9):
            msg = f"not enough cash: {order.quote_amount:.2f} > {self.cash:.2f}"
            raise ValueError(msg)
        time, ref = self._open_at(order)
        price = self.costs.buy_price(ref)
        quantity = order.quote_amount / (price * (1 + self.costs.fee))
        fill = Fill(
            order.id,
            order.symbol,
            Side.BUY,
            time,
            price,
            quantity,
            fee=order.quote_amount - quantity * price,
        )
        self.cash = max(self.cash - order.quote_amount, 0.0)
        self.fills.append(fill)
        position = Position(order.symbol, quantity, time, ref, price, order.quote_amount)
        self.positions[order.symbol] = position
        if order.exit_plan is not None:
            bracket = Order(
                order.symbol,
                Side.SELL,
                time,
                OrderType.BRACKET,
                quantity=quantity,
                exit_plan=order.exit_plan,
            )
            self._brackets[bracket.id] = _Bracket(bracket, self._resolve(bracket))
            position.bracket_id = bracket.id
        return fill

    def _resolve(self, bracket: Order) -> ExitFill | None:
        assert bracket.exit_plan is not None
        market = self._market(bracket.symbol)
        window = market.candles.window(bracket.time)
        if len(window) == 0 or int(window.open_time[0]) != to_us(bracket.time):
            return None
        return resolve_exit(window, bracket.exit_plan, self.costs, fine=market.fine)

    def _sell(self, order: Order) -> Fill:
        position = self.positions.get(order.symbol)
        if position is None:
            msg = f"not holding {order.symbol}"
            raise ValueError(msg)
        quantity = position.quantity if order.quantity is None else order.quantity
        whole = quantity >= position.quantity * (1 - 1e-9)
        if quantity > position.quantity * (1 + 1e-9):
            msg = f"cannot sell {quantity} {order.symbol}, holding {position.quantity}"
            raise ValueError(msg)
        if position.bracket_id is not None:
            if not whole:
                msg = "cancel the bracket before selling part of a position"
                raise ValueError(msg)
            del self._brackets[position.bracket_id]
        time, raw = self._open_at(order)
        if whole:
            quantity = position.quantity
            del self.positions[order.symbol]
        else:
            position.cost *= 1 - quantity / position.quantity
            position.quantity -= quantity
        return self._credit(order, quantity, self.costs.sell_price(raw, market=True), time, None)

    def _credit(
        self,
        order: Order,
        quantity: float,
        price: float,
        time: datetime,
        reason: ExitReason | None,
    ) -> Fill:
        fee = quantity * price * self.costs.fee
        self.cash += quantity * price - fee
        fill = Fill(order.id, order.symbol, Side.SELL, time, price, quantity, fee, reason)
        self.fills.append(fill)
        return fill
