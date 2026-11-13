"""The book an engine keeps (cash, open cost basis, the day's realised P&L) and position sizing.

Equity here is *realised* equity: cash plus the cost basis of open positions. Open positions are
not marked to market for sizing or risk decisions (as in the research simulators), so a decision
never depends on prices after the entries it already made. The mark-to-market equity curve is a
backtest output, computed separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from crypto_moonshot.models import EPOCH

_DAY = timedelta(days=1)


def utc_day(time: datetime) -> int:
    """Days since the Unix epoch of the UTC day containing `time`."""
    return (time - EPOCH) // _DAY


@dataclass
class Book:
    """Cash and open positions (cost basis per symbol, entry fee included) of a long-only book,
    plus the realised P&L of the current UTC day for the daily loss limit.

    Events must be fed in time order. The day rolls over on the first event of a later UTC day;
    `day_start_equity` is the realised equity before that event.
    """

    cash: float
    costs: dict[str, float] = field(default_factory=dict)
    open_cost: float = 0.0
    day: int | None = None
    day_start_equity: float = 0.0
    day_pnl: float = 0.0
    """Realised P&L since the start of the current UTC day."""
    day_worst_pnl: float = 0.0
    """The lowest `day_pnl` reached so far today (0 or below), so a halt lasts all day."""
    day_entries: int = 0
    """Entries opened since the start of the current UTC day."""

    @property
    def equity(self) -> float:
        """Realised equity: cash plus the cost basis of open positions."""
        return self.cash + self.open_cost

    @property
    def open_positions(self) -> int:
        return len(self.costs)

    def holds(self, symbol: str) -> bool:
        return symbol in self.costs

    def roll_day(self, time: datetime) -> None:
        """Start a new UTC day if `time` falls on a later day than the current one."""
        day = utc_day(time)
        if day != self.day:
            self.day = day
            self.day_start_equity = self.equity
            self.day_pnl = 0.0
            self.day_worst_pnl = 0.0
            self.day_entries = 0

    def open(self, symbol: str, cost: float, time: datetime) -> None:
        """Record an entry that spent `cost` USDT (fee included)."""
        if symbol in self.costs:
            msg = f"already holding {symbol}"
            raise ValueError(msg)
        self.roll_day(time)
        self.cash -= cost
        self.costs[symbol] = cost
        self.open_cost += cost
        self.day_entries += 1

    def close(self, symbol: str, proceeds: float, time: datetime) -> float:
        """Record an exit that returned `proceeds` USDT (fee deducted); returns its P&L."""
        self.roll_day(time)
        cost = self.costs.pop(symbol)
        self.open_cost = self.open_cost - cost if self.costs else 0.0
        self.cash += proceeds
        pnl = proceeds - cost
        self.day_pnl += pnl
        self.day_worst_pnl = min(self.day_worst_pnl, self.day_pnl)
        return pnl


class Sizing(Protocol):
    """How many USDT (entry fee included) the next entry spends."""

    def stake(self, book: Book) -> float: ...


@dataclass(frozen=True, slots=True)
class FixedStake:
    """The same USDT amount for every entry: no compounding, P&L simply adds up."""

    usdt: float = 50.0

    def __post_init__(self) -> None:
        if not self.usdt > 0:
            msg = f"stake must be positive, got {self.usdt}"
            raise ValueError(msg)

    def stake(self, book: Book) -> float:
        return self.usdt


@dataclass(frozen=True, slots=True)
class EqualSlots:
    """Compounding: each entry spends `realised equity / slots`."""

    slots: int

    def __post_init__(self) -> None:
        if self.slots < 1:
            msg = f"slots must be at least 1, got {self.slots}"
            raise ValueError(msg)

    def stake(self, book: Book) -> float:
        return book.equity / self.slots


@dataclass(frozen=True, slots=True)
class CashCapped:
    """Equal slots, but never more than the cash left.

    The engine skips an entry whose stake exceeds the cash (after a loss, `equity / slots` can).
    Capping the stake at the cash takes every entry the signals offer, which is what the research
    books did (they never checked cash). The bp per trade, an unweighted mean, does not depend on
    the stake.
    """

    slots: int

    def __post_init__(self) -> None:
        if self.slots < 1:
            msg = f"slots must be at least 1, got {self.slots}"
            raise ValueError(msg)

    def stake(self, book: Book) -> float:
        return min(book.equity / self.slots, book.cash)


@runtime_checkable
class WeightedSizing(Protocol):
    """A sizing that also reads a per-signal weight (the signal frame's `weight` column)."""

    def stake(self, book: Book) -> float: ...

    def stake_weighted(self, book: Book, weight: float) -> float: ...


@dataclass(frozen=True, slots=True)
class WeightedSlots:
    """Equal slots scaled by the signal's weight, capped at the cash left:
    `min(realised equity / slots x weight, cash)`. Without a weight the weight is 1."""

    slots: int

    def __post_init__(self) -> None:
        if self.slots < 1:
            msg = f"slots must be at least 1, got {self.slots}"
            raise ValueError(msg)

    def stake(self, book: Book) -> float:
        return self.stake_weighted(book, 1.0)

    def stake_weighted(self, book: Book, weight: float) -> float:
        return min(book.equity / self.slots * weight, book.cash)
