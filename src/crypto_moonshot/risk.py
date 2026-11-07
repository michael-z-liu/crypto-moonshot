"""Pre-trade risk checks, shared by the backtest and (later) the live engine.

Each check looks at the `Book` and a proposed entry (symbol and stake) and either allows it or
names why not. Checks are independent and composed as a sequence; `first_rejection` returns the
reason of the first that refuses. They read only the book, never prices, so the live engine can
run the same ones before placing an order.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, Protocol

from crypto_moonshot.portfolio import Book


class RiskCheck(Protocol):
    """A rule a new entry must pass. `reason` names the rule in skip counts."""

    @property
    def reason(self) -> str: ...

    def allows(self, book: Book, symbol: str, stake: float) -> bool: ...


@dataclass(frozen=True, slots=True)
class MaxPositions:
    """At most `limit` positions open at once."""

    limit: int
    reason: ClassVar[str] = "slots_full"

    def __post_init__(self) -> None:
        if self.limit < 1:
            msg = f"max positions must be at least 1, got {self.limit}"
            raise ValueError(msg)

    def allows(self, book: Book, symbol: str, stake: float) -> bool:
        return book.open_positions < self.limit


@dataclass(frozen=True, slots=True)
class OnePerSymbol:
    """No second position in a symbol already held."""

    reason: ClassVar[str] = "symbol_held"

    def allows(self, book: Book, symbol: str, stake: float) -> bool:
        return not book.holds(symbol)


@dataclass(frozen=True, slots=True)
class MaxExposure:
    """After the entry, the cost basis of open positions is at most `limit` x realised equity."""

    limit: float
    reason: ClassVar[str] = "exposure"

    def __post_init__(self) -> None:
        if not self.limit > 0:
            msg = f"max exposure must be positive, got {self.limit}"
            raise ValueError(msg)

    def allows(self, book: Book, symbol: str, stake: float) -> bool:
        equity = book.equity
        return equity > 0 and book.open_cost + stake <= self.limit * equity * (1 + 1e-12)


@dataclass(frozen=True, slots=True)
class DailyLossLimit:
    """No new entries for the rest of the UTC day once the day's *realised* P&L has fallen
    below `-limit x` the realised equity at the start of the day.

    Realised P&L only: open positions are not marked to market, so a position that is down but
    still open does not count until it closes. Once hit, the halt lasts until the next UTC day,
    even if later exits that day recover the loss. The book must be rolled to the entry time
    (`Book.roll_day`) before checking.
    """

    limit: float
    reason: ClassVar[str] = "daily_loss_limit"

    def __post_init__(self) -> None:
        if not self.limit > 0:
            msg = f"daily loss limit must be positive, got {self.limit}"
            raise ValueError(msg)

    def allows(self, book: Book, symbol: str, stake: float) -> bool:
        return book.day_worst_pnl >= -self.limit * book.day_start_equity


def first_rejection(
    checks: Iterable[RiskCheck], book: Book, symbol: str, stake: float
) -> str | None:
    """The reason of the first check that refuses the entry, or `None` if all allow it."""
    for check in checks:
        if not check.allows(book, symbol, stake):
            return check.reason
    return None
