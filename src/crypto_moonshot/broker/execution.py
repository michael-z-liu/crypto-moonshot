"""The execution interface: what the live engine needs from any broker, paper or exchange.

The engine trades through an async `ExecutionBroker`. The paper broker is synchronous and
simulates fills from the candles the engine hands it; an exchange broker places real orders,
learns of fills later (a user data stream) and can fail to answer. One interface covers both,
and two properties tell the engine what a broker needs:

- `needs_candles`: the broker simulates fills from closed candles (`candle_needs`,
  `add_candles`). Only the paper broker does.
- `manages_time_exits`: the broker's bracket includes the time exit. The paper broker's does;
  on an exchange there is no order that expires after a holding time, so the engine sells at the
  first tick at or after `entry time + max_hold`.
- `sell_cancels_bracket`: a market sell of the whole position cancels its bracket by itself (the
  paper broker). Otherwise the engine cancels the bracket first (an OCO list holds the balance a
  sell needs) and sells only once the cancel is confirmed.

Order lifecycle (see `docs/architecture.md`, "Exchange broker")
-------------------------------------------------------------
The engine commits each order's intent (a `new` row with its deterministic client order ID)
before calling `submit`, and records the `SubmitResult` in a second transaction:

- `accepted`: the broker holds the order (`exchange_id`); anything that already happened to it
  (a market order's fill in the response) is in `events`. Later changes come from `events`.
- `rejected`: the order was refused (`reason`); nothing was placed.
- `unknown`: the request failed without an answer (a timeout, a 5xx): the order may or may not
  exist. The engine marks it `unknown` and resolves it with `query` at its next broker step,
  before anything else touches the symbol; a crash between intent and outcome leaves a `new`
  row that the next start resolves the same way. Nothing is retried blind.

Events
------
`events(until)` returns, in time order, everything that happened to the broker's orders since
the last call and was not reported yet (in a `SubmitResult`, `CancelResult` or `QueryResult`):

- `EntryFilled` (fill and the position, with its bracket ID if the bracket was placed with it),
  `ExitFilled` (a bracket exit: `fill.order_id` is the position's bracket ID and `fill.reason`
  says which leg; or a market sell: its own ID, `fill.reason` None), `OrderCancelled`
  (`CancelReason`) -- the paper broker's events, shared;
- `OrderRejection`: an order the broker accepted was rejected afterwards;
- `BracketFailed` / `BracketPlaced`: an entry filled but its bracket could not be placed (the
  position is unprotected; the broker retries), then it was;
- `Reconciliation`: a report from a reconciliation the broker ran by itself (after a
  reconnect);
- `BrokerHalted`: the broker stopped placing orders (the exchange refused its keys, or banned
  it); every later order is rejected with the same reason.

Every change of an order the engine knows must be reported exactly once; a duplicate is
ignored. `until` bounds a simulated broker (it decides only on candles closed by then); an
exchange broker returns everything it has received.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from crypto_moonshot.broker.base import Costs, Order, OrderRejected, Side
from crypto_moonshot.broker.candles import Candles
from crypto_moonshot.broker.paper import (
    CancelReason,
    EntryFilled,
    ExitFilled,
    OrderCancelled,
    PaperBroker,
    PaperPosition,
    PaperSnapshot,
)
from crypto_moonshot.models import Interval

__all__ = [
    "BalanceMismatch",
    "BracketFailed",
    "BracketPlaced",
    "BrokerEvent",
    "BrokerHalted",
    "CancelReason",
    "CancelResult",
    "CancelStatus",
    "EntryFilled",
    "ExecutionBroker",
    "ExitFilled",
    "OrderCancelled",
    "OrderRejection",
    "OrderState",
    "OrphanOrder",
    "PaperExecution",
    "Position",
    "QueryResult",
    "Reconciliation",
    "ReportsExchange",
    "RestoresState",
    "SubmitResult",
    "SubmitStatus",
    "WorkingOrder",
]

Position = PaperPosition
"""An open position as a broker reports it (the paper broker's type, shared by every broker)."""


# -- events -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderRejection:
    """An order the broker had accepted was rejected afterwards (`reason`: a short code such as
    `insufficient_balance` or `filter`). A rejected buy's position is cancelled and its stake
    released."""

    order_id: str
    symbol: str
    side: Side
    time: datetime
    reason: str


@dataclass(frozen=True, slots=True)
class BracketFailed:
    """The entry `entry_order_id` filled but its bracket could not be placed (`reason`): the
    position is unprotected until a `BracketPlaced` follows. The broker retries."""

    symbol: str
    entry_order_id: str
    time: datetime
    reason: str


@dataclass(frozen=True, slots=True)
class BracketPlaced:
    """A bracket was placed after its entry's `EntryFilled` (which had no bracket ID):
    `position.bracket_id` is its ID. Each bracket that reaches the exchange needs a new ID."""

    position: Position
    time: datetime


@dataclass(frozen=True, slots=True)
class BrokerHalted:
    """The broker stopped placing orders (`reason`: `auth` when the exchange refuses the keys,
    `banned` after an IP ban); every later order is rejected with that reason. Needs an
    operator."""

    time: datetime
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class OrphanOrder:
    """An open order on the exchange with one of our ID prefixes that the state file does not
    know. Reported and left alone."""

    order_id: str
    symbol: str
    side: Side
    detail: str = ""


@dataclass(frozen=True, slots=True)
class BalanceMismatch:
    """The exchange holds less of `asset` than the open positions in it need (free plus
    locked), after the updates the reconciliation reported."""

    asset: str
    expected: float
    actual: float


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """What a reconciliation found besides the order updates (which come from `events`).
    `clean` when there is nothing to report."""

    time: datetime
    orphans: tuple[OrphanOrder, ...] = ()
    mismatches: tuple[BalanceMismatch, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.orphans and not self.mismatches


BrokerEvent = (
    EntryFilled
    | ExitFilled
    | OrderCancelled
    | OrderRejection
    | BracketFailed
    | BracketPlaced
    | Reconciliation
    | BrokerHalted
)
"""Everything `ExecutionBroker.events` (and the `events` of a result) can report."""


# -- results ----------------------------------------------------------------------------------


class SubmitStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SubmitResult:
    """What `submit` did; see the module docstring."""

    status: SubmitStatus
    reason: str | None = None
    """Why it was rejected (a `RejectReason` value, or an exchange broker's short code)."""
    exchange_id: str | None = None
    events: tuple[BrokerEvent, ...] = ()
    """What already happened to the accepted order (a market order's fill)."""

    @classmethod
    def accepted(
        cls, exchange_id: str | None = None, events: Sequence[BrokerEvent] = ()
    ) -> SubmitResult:
        return cls(SubmitStatus.ACCEPTED, exchange_id=exchange_id, events=tuple(events))

    @classmethod
    def rejected(cls, reason: str) -> SubmitResult:
        return cls(SubmitStatus.REJECTED, reason=reason)

    @classmethod
    def unknown(cls) -> SubmitResult:
        return cls(SubmitStatus.UNKNOWN)


class CancelStatus(StrEnum):
    CANCELLED = "cancelled"
    """The order was open and is cancelled now."""
    NOT_OPEN = "not_open"
    """Nothing to cancel: the order is final (filled, cancelled, rejected) or was never placed.
    What happened to it is in the result's `events` or was reported before."""
    UNKNOWN = "unknown"
    """The request got no answer: the order may still be open."""


@dataclass(frozen=True, slots=True)
class CancelResult:
    status: CancelStatus
    events: tuple[BrokerEvent, ...] = ()
    """Changes the broker learned while cancelling (e.g. the fill that beat the cancel)."""


class OrderState(StrEnum):
    NOT_FOUND = "not_found"
    """The broker has no such order: it was never placed."""
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"
    """The broker still cannot tell (the query failed): ask again later."""


@dataclass(frozen=True, slots=True)
class QueryResult:
    """An order's state at the broker. For an order that exists, its fills and cancellations
    not reported yet are in `events` (or come from the next `events` call)."""

    state: OrderState
    reason: str | None = None
    """Why it was rejected."""
    events: tuple[BrokerEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkingOrder:
    """An order the state file holds as working (for `reconcile`: accepted and not final; for
    `RestoresState.load`: also `new` and `unknown` ones)."""

    order: Order
    interval: Interval
    strategy: str


# -- the interface ----------------------------------------------------------------------------


class ExecutionBroker(Protocol):
    """What the live engine trades through; see the module docstring.

    `cancel` and `query` take the engine's `Order` (an exchange needs the symbol, and a bracket's
    type tells it to cancel an order list). None of the methods may hold the engine's state file;
    they never raise for an order outcome (that is what the results are for); anything else
    they raise stops the engine, which recovers from the state file.
    """

    @property
    def needs_candles(self) -> bool:
        """Whether fills are simulated from candles (`candle_needs`, `add_candles`)."""
        ...

    @property
    def manages_time_exits(self) -> bool:
        """Whether the bracket includes the time exit; otherwise the engine runs it."""
        ...

    @property
    def sell_cancels_bracket(self) -> bool:
        """Whether a market sell of a whole position cancels its bracket by itself; otherwise
        the engine cancels the bracket first."""
        ...

    def candle_needs(self, interval: Interval) -> dict[str, datetime]:
        """`needs_candles` brokers: per symbol, the first candle on `interval` the broker
        needs (a position's entry candle, a pending order's time). Others return `{}`."""
        ...

    def add_candles(self, symbol: str, candles: Candles) -> None:
        """`needs_candles` brokers: closed candles of `symbol` (any order, duplicates
        ignored). Others ignore them."""
        ...

    async def submit(self, order: Order, *, interval: Interval) -> SubmitResult:
        """Place `order` (its `id` is the client order ID); `interval` is the strategy's."""
        ...

    async def cancel(self, order: Order) -> CancelResult:
        """Cancel an open order (a pending market order, or a position's bracket)."""
        ...

    async def query(self, order: Order) -> QueryResult:
        """The order's state at the broker, by its client order ID."""
        ...

    async def events(
        self, until: datetime, *, interval: Interval | None = None
    ) -> list[BrokerEvent]:
        """Everything that happened since the last call, in time order (see the module
        docstring). `interval`: a simulated broker decides only that interval's series."""
        ...

    async def reconcile(
        self,
        orders: Sequence[WorkingOrder],
        positions: Sequence[Position],
        *,
        now: datetime,
    ) -> Reconciliation:
        """Compare the broker with the state file at start-up (after every `new` and `unknown`
        order was resolved): bring each working order up to date (its changes come from the
        next `events` call), and report orphaned orders and balances that do not cover the
        open positions. Places and cancels nothing."""
        ...


@runtime_checkable
class RestoresState(Protocol):
    """An exchange broker that must know the state file's orders and positions before the
    engine resolves anything (a whole-position sell has no quantity: the broker sells what the
    position holds). The engine calls `load` at start, before `query` and `reconcile`."""

    def load(self, orders: Sequence[WorkingOrder], positions: Sequence[Position]) -> None:
        """Adopt the state file's working orders (new, pending or unknown) and open
        positions. Contacts nothing."""
        ...


@runtime_checkable
class ReportsExchange(Protocol):
    """An exchange broker that reports its own state for the status (never a secret)."""

    def exchange_status(self) -> dict[str, Any]:
        """JSON-ready facts: stream connection, halt reason, dust, odd commissions..."""
        ...


# -- the paper broker behind the interface ----------------------------------------------------


class PaperExecution:
    """`PaperBroker` as an `ExecutionBroker`: synchronous underneath, so `submit` and `cancel`
    answer at once and never return `unknown`; fills come from `events` (`advance`).

    It is rebuilt from the state file (`restore`) instead of reconciling: the paper broker is
    deterministic, so restoring an order the state file holds is placing it.
    """

    needs_candles = True
    manages_time_exits = True
    sell_cancels_bracket = True

    def __init__(self, broker: PaperBroker) -> None:
        self._broker = broker

    @classmethod
    def create(cls, cash: float, costs: Costs | None = None) -> PaperExecution:
        return cls(PaperBroker(cash, costs))

    @property
    def paper(self) -> PaperBroker:
        """The paper broker underneath."""
        return self._broker

    def restore(self, snapshot: PaperSnapshot) -> None:
        """Replace the paper broker with one in the state `snapshot` describes (no candles)."""
        self._broker = PaperBroker.restore(snapshot, self._broker.costs)

    def candle_needs(self, interval: Interval) -> dict[str, datetime]:
        out: dict[str, datetime] = {}
        for position in self._broker.positions.values():
            if position.interval is interval:
                start = position.entry_time
                out[position.symbol] = min(out.get(position.symbol, start), start)
        for pending in self._broker.pending:
            if pending.interval is interval:
                symbol, start = pending.order.symbol, pending.order.time
                out[symbol] = min(out.get(symbol, start), start)
        return out

    def add_candles(self, symbol: str, candles: Candles) -> None:
        self._broker.add_candles(symbol, candles)

    async def submit(self, order: Order, *, interval: Interval) -> SubmitResult:
        try:
            self._broker.place_order(order, interval=interval)
        except OrderRejected as exc:
            return SubmitResult.rejected(exc.reason.value)
        return SubmitResult.accepted()

    async def cancel(self, order: Order) -> CancelResult:
        try:
            self._broker.cancel_order(order.id)
        except KeyError:
            return CancelResult(CancelStatus.NOT_OPEN)
        return CancelResult(CancelStatus.CANCELLED)

    async def query(self, order: Order) -> QueryResult:
        held = any(p.order.id == order.id for p in self._broker.pending) or any(
            p.bracket_id == order.id for p in self._broker.positions.values()
        )
        return QueryResult(OrderState.OPEN if held else OrderState.NOT_FOUND)

    async def events(
        self, until: datetime, *, interval: Interval | None = None
    ) -> list[BrokerEvent]:
        return list(self._broker.advance(until, interval=interval))

    async def reconcile(
        self,
        orders: Sequence[WorkingOrder],
        positions: Sequence[Position],
        *,
        now: datetime,
    ) -> Reconciliation:
        """Nothing to compare: the paper broker is rebuilt from the state file."""
        return Reconciliation(now)
