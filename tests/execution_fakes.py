"""A fake exchange-style `ExecutionBroker` for engine tests.

`FakeExchangeBroker` behaves like an exchange broker would: orders are acknowledged at once but
their fills are reported later, through `events`; brackets are resting orders that fill on a
price path the test drives (`move`); it runs no time exits; and the test can make it reject an
order at submit or after accepting it, answer `unknown` (with the order placed or not), fail to
place a bracket, let a bracket fill while it is being cancelled, and report orphans or balances
that do not cover the positions.

The exchange's truth (orders, positions, balances) survives `restart()`, which drops what the
broker had not reported yet and forgets what it had (a new process): `query` and `reconcile`
bring the engine up to date again.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

from crypto_moonshot.broker.base import (
    ExitReason,
    Fill,
    Order,
    OrderType,
    Side,
    base_asset,
    child_order_id,
)
from crypto_moonshot.broker.candles import Candles
from crypto_moonshot.broker.execution import (
    BalanceMismatch,
    BracketFailed,
    BracketPlaced,
    BrokerEvent,
    CancelReason,
    CancelResult,
    CancelStatus,
    EntryFilled,
    ExitFilled,
    OrderCancelled,
    OrderRejection,
    OrderState,
    OrphanOrder,
    Position,
    QueryResult,
    Reconciliation,
    SubmitResult,
    WorkingOrder,
)
from crypto_moonshot.live.engine import needs_bracket
from crypto_moonshot.models import Interval

_STATES = {
    "open": OrderState.OPEN,
    "filled": OrderState.FILLED,
    "cancelled": OrderState.CANCELLED,
    "rejected": OrderState.REJECTED,
}


@dataclass
class ExchangeOrder:
    """An order the exchange holds: `status` open, filled, cancelled or rejected; `events` what
    happened to it, `reported` whether the broker told the engine."""

    order: Order
    interval: Interval
    status: str = "open"
    reason: str | None = None
    events: list[BrokerEvent] = field(default_factory=list)
    reported: bool = False


class FakeExchangeBroker:
    """See the module docstring. `clock` stamps fills and events; `prices` are the symbols'
    current prices (market orders fill there, brackets check them in `move`)."""

    needs_candles = False
    manages_time_exits = False
    sell_cancels_bracket = False

    def __init__(
        self, clock: Callable[[], datetime], prices: dict[str, float], *, fee: float = 0.001
    ) -> None:
        self.clock = clock
        self.prices = dict(prices)
        self.fee = fee
        self.orders: dict[str, ExchangeOrder] = {}
        self.positions: dict[str, Position] = {}
        self.balances: dict[str, float] = {}
        self.queue: list[tuple[str, BrokerEvent]] = []
        self.calls: list[tuple[str, str]] = []
        # faults the test sets
        self.reject_next: dict[str, str] = {}
        """symbol -> reason: the next order in it is rejected at submit."""
        self.reject_later: dict[str, str] = {}
        """symbol -> reason: the next order in it is accepted, then rejected (an event)."""
        self.unknown_next: dict[str, bool] = {}
        """symbol -> placed: the next submit answers unknown, having placed the order or not."""
        self.query_unknown: set[str] = set()
        """Order IDs whose query answers unknown."""
        self.cancel_unknown: set[str] = set()
        """Symbols whose cancel answers unknown (and cancels nothing)."""
        self.hold: set[str] = set()
        """Symbols whose market orders stay open until `release`."""
        self.fail_bracket: set[str] = set()
        """Symbols whose next bracket cannot be placed."""
        self.race_cancel: set[str] = set()
        """Symbols whose bracket fills (at the current price) while it is being cancelled."""
        self.immediate = False
        """Report a market order's fill in its `SubmitResult` instead of later."""

    # -- what the test drives ---------------------------------------------------------------

    def move(self, symbol: str, price: float) -> None:
        """The price moves: a bracket whose take profit or stop it reaches fills."""
        self.prices[symbol] = price
        position = self.positions.get(symbol)
        if position is None or position.bracket_id is None or position.exit_plan is None:
            return
        plan = position.exit_plan
        ref = position.entry_ref
        if plan.take_profit is not None and price >= ref * (1 + plan.take_profit):
            self._fill_bracket(position, ExitReason.TAKE_PROFIT, ref * (1 + plan.take_profit))
        elif plan.stop is not None and price <= ref * (1 - plan.stop):
            self._fill_bracket(position, ExitReason.STOP, price)

    def release(self, symbol: str) -> None:
        """Held market orders in `symbol` fill now (reported by the next `events`)."""
        self.hold.discard(symbol)
        for item in list(self.orders.values()):
            order = item.order
            if order.symbol == symbol and item.status == "open" and order.type is OrderType.MARKET:
                self._queue(item, self._fill(item))

    def place_brackets(self) -> None:
        """Retry the brackets that failed: each unprotected position gets one."""
        for symbol, position in list(self.positions.items()):
            if position.bracket_id is not None or not needs_bracket(position.exit_plan):
                continue
            placed = replace(
                position, bracket_id=child_order_id(position.entry_order_id, "bracket")
            )
            self.positions[symbol] = placed
            bracket = placed.bracket()
            assert bracket is not None
            item = ExchangeOrder(bracket, position.interval)
            self.orders[bracket.id] = item
            self._queue(item, [BracketPlaced(placed, self.clock())])

    def add_orphan(self, order: Order, interval: Interval = Interval.H1) -> None:
        """An open order on the exchange the engine never placed."""
        self.orders[order.id] = ExchangeOrder(order, interval)

    def restart(self) -> None:
        """A new broker process: unreported events are lost, nothing counts as reported."""
        self.queue.clear()
        for item in self.orders.values():
            item.reported = False

    def submits(self, order_id: str) -> int:
        return self.calls.count(("submit", order_id))

    # -- ExecutionBroker ----------------------------------------------------------------------

    def candle_needs(self, interval: Interval) -> dict[str, datetime]:
        return {}

    def add_candles(self, symbol: str, candles: Candles) -> None:
        pass

    async def submit(self, order: Order, *, interval: Interval) -> SubmitResult:
        self.calls.append(("submit", order.id))
        symbol = order.symbol
        if order.id in self.orders:
            return SubmitResult.rejected("duplicate_id")
        if symbol in self.reject_next:
            return SubmitResult.rejected(self.reject_next.pop(symbol))
        if symbol in self.unknown_next:
            if self.unknown_next.pop(symbol):
                self._queue_new(order, interval)
            return SubmitResult.unknown()
        if order.side is Side.SELL:
            position = self.positions.get(symbol)
            if position is None or position.bracket_id is not None:
                return SubmitResult.rejected("insufficient_balance")  # an OCO locks it
        item, events = self._accept(order, interval)
        if self.immediate and events:
            item.reported = True
            return SubmitResult.accepted(order.id, events)
        self._queue(item, events)
        return SubmitResult.accepted(order.id)

    async def cancel(self, order: Order) -> CancelResult:
        self.calls.append(("cancel", order.id))
        item = self.orders.get(order.id)
        if order.symbol in self.cancel_unknown:
            return CancelResult(CancelStatus.UNKNOWN)
        if item is None or item.status != "open":
            return CancelResult(CancelStatus.NOT_OPEN)
        position = self.positions.get(order.symbol)
        if order.type is OrderType.BRACKET and order.symbol in self.race_cancel:
            assert position is not None
            self.race_cancel.discard(order.symbol)
            self._fill_bracket(position, ExitReason.STOP, self.prices[order.symbol])
            events = self._take(item)
            return CancelResult(CancelStatus.NOT_OPEN, tuple(events))
        item.status = "cancelled"
        if order.type is OrderType.BRACKET and position is not None:
            self.positions[order.symbol] = replace(position, bracket_id=None)
        return CancelResult(CancelStatus.CANCELLED)

    async def query(self, order: Order) -> QueryResult:
        self.calls.append(("query", order.id))
        if order.id in self.query_unknown:
            return QueryResult(OrderState.UNKNOWN)
        item = self.orders.get(order.id)
        if item is None:
            return QueryResult(OrderState.NOT_FOUND)
        return QueryResult(_STATES[item.status], item.reason, tuple(self._take(item)))

    async def events(
        self, until: datetime, *, interval: Interval | None = None
    ) -> list[BrokerEvent]:
        self.calls.append(("events", until.isoformat()))
        out = [event for _, event in self.queue]
        for order_id, _ in self.queue:
            self.orders[order_id].reported = True
        self.queue.clear()
        out.sort(key=lambda e: e.time)
        return out

    async def reconcile(
        self,
        orders: Sequence[WorkingOrder],
        positions: Sequence[Position],
        *,
        now: datetime,
    ) -> Reconciliation:
        self.calls.append(("reconcile", now.isoformat()))
        known = {w.order.id for w in orders}
        for order_id in known:
            item = self.orders.get(order_id)
            if item is not None and not item.reported and item.events:
                self._queue(item, [])  # report what happened while nobody listened
        orphans = tuple(
            OrphanOrder(item.order.id, item.order.symbol, item.order.side, "unknown locally")
            for item in self.orders.values()
            if item.status == "open" and item.order.id not in known
        )
        need: dict[str, float] = {}
        for position in positions:
            if position.symbol in self.positions:  # not closed by an update reported now
                asset = base_asset(position.symbol)
                need[asset] = need.get(asset, 0.0) + position.quantity
        mismatches = tuple(
            BalanceMismatch(asset, quantity, self.balances.get(asset, 0.0))
            for asset, quantity in sorted(need.items())
            if self.balances.get(asset, 0.0) < quantity * (1 - 1e-9)
        )
        return Reconciliation(now, orphans, mismatches)

    # -- internals ----------------------------------------------------------------------------

    def _queue(self, item: ExchangeOrder, events: Sequence[BrokerEvent]) -> None:
        item.events.extend(events)
        item.reported = False
        queued = {id(e) for _, e in self.queue}
        for event in item.events:
            if id(event) not in queued:
                self.queue.append((item.order.id, event))

    def _take(self, item: ExchangeOrder) -> list[BrokerEvent]:
        """The order's events not reported yet (taken out of the queue)."""
        if item.reported:
            return []
        item.reported = True
        mine = {id(e) for e in item.events}
        self.queue = [(o, e) for o, e in self.queue if id(e) not in mine]
        return list(item.events)

    def _queue_new(self, order: Order, interval: Interval) -> None:
        item, events = self._accept(order, interval)
        self._queue(item, events)

    def _accept(self, order: Order, interval: Interval) -> tuple[ExchangeOrder, list[BrokerEvent]]:
        item = ExchangeOrder(order, interval)
        self.orders[order.id] = item
        if order.symbol in self.reject_later:
            item.status, item.reason = "rejected", self.reject_later.pop(order.symbol)
            return item, [
                OrderRejection(order.id, order.symbol, order.side, self.clock(), item.reason)
            ]
        if order.symbol in self.hold:
            return item, []
        return item, self._fill(item)

    def _fill(self, item: ExchangeOrder) -> list[BrokerEvent]:
        order, now = item.order, self.clock()
        symbol, asset = order.symbol, base_asset(order.symbol)
        price = self.prices[symbol]
        item.status = "filled"
        if order.side is Side.SELL:
            position = self.positions.pop(symbol)
            fill = Fill(
                order.id,
                symbol,
                Side.SELL,
                now,
                price,
                position.quantity,
                position.quantity * price * self.fee,
            )
            self.balances[asset] = self.balances.get(asset, 0.0) - position.quantity
            return [ExitFilled(fill, position)]
        assert order.quote_amount is not None
        quantity = order.quote_amount / (price * (1 + self.fee))
        fill = Fill(
            order.id, symbol, Side.BUY, now, price, quantity, order.quote_amount - quantity * price
        )
        plan = order.exit_plan
        failed = needs_bracket(plan) and symbol in self.fail_bracket
        self.fail_bracket.discard(symbol)
        bracket_id = (
            child_order_id(order.id, "bracket") if needs_bracket(plan) and not failed else None
        )
        position = Position(
            symbol=symbol,
            interval=item.interval,
            quantity=quantity,
            entry_time=now,
            entry_ref=price,
            entry_price=price,
            cost=order.quote_amount,
            entry_order_id=order.id,
            exit_plan=plan,
            bracket_id=bracket_id,
        )
        self.positions[symbol] = position
        self.balances[asset] = self.balances.get(asset, 0.0) + quantity
        events: list[BrokerEvent] = [EntryFilled(fill, position)]
        bracket = position.bracket()
        if bracket is not None:
            self.orders[bracket.id] = ExchangeOrder(bracket, item.interval)
        if failed:
            events.append(BracketFailed(symbol, order.id, now, "percent_price_by_side"))
        return events

    def _fill_bracket(self, position: Position, reason: ExitReason, price: float) -> None:
        assert position.bracket_id is not None
        item = self.orders[position.bracket_id]
        item.status = "filled"
        del self.positions[position.symbol]
        asset = base_asset(position.symbol)
        self.balances[asset] = self.balances.get(asset, 0.0) - position.quantity
        fill = Fill(
            position.bracket_id,
            position.symbol,
            Side.SELL,
            self.clock(),
            price,
            position.quantity,
            position.quantity * price * self.fee,
            reason,
        )
        self._queue(item, [ExitFilled(fill, position)])

    def cancel_event(self, order_id: str) -> None:
        """The exchange cancels an open order by itself (reported by the next `events`)."""
        item = self.orders[order_id]
        item.status = "cancelled"
        self._queue(item, [OrderCancelled(item.order, self.clock(), CancelReason.EXPIRED)])
