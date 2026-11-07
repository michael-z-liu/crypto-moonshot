from datetime import UTC, datetime

import pytest

from crypto_moonshot.portfolio import Book
from crypto_moonshot.risk import (
    DailyLossLimit,
    MaxExposure,
    MaxPositions,
    OnePerSymbol,
    RiskCheck,
    first_rejection,
)

T = datetime(2024, 1, 1, tzinfo=UTC)


def book_with(*costs: float, cash: float = 1_000.0) -> Book:
    book = Book(cash=cash)
    for k, cost in enumerate(costs):
        book.open(f"S{k}USDT", cost, T)
    return book


def test_max_positions() -> None:
    check = MaxPositions(2)
    assert check.allows(book_with(10), "XUSDT", 10)
    assert not check.allows(book_with(10, 10), "XUSDT", 10)
    assert check.reason == "slots_full"
    with pytest.raises(ValueError, match="at least 1"):
        MaxPositions(0)


def test_one_per_symbol() -> None:
    book = book_with(10)
    assert not OnePerSymbol().allows(book, "S0USDT", 10)
    assert OnePerSymbol().allows(book, "S1USDT", 10)


def test_max_exposure_counts_the_new_stake() -> None:
    book = book_with(400)  # equity 1000, open cost 400
    assert MaxExposure(0.5).allows(book, "XUSDT", 100)
    assert not MaxExposure(0.5).allows(book, "XUSDT", 100.01)
    assert MaxExposure(1.0).allows(book, "XUSDT", 600)


def test_daily_loss_limit_uses_realised_pnl_and_latches() -> None:
    book = book_with(100, 100)
    check = DailyLossLimit(0.05)
    book.close("S0USDT", 60.0, T)  # -40: -4% of 1000
    assert check.allows(book, "XUSDT", 10)
    book.close("S1USDT", 89.0, T)  # -51 in total: below -5%
    assert not check.allows(book, "XUSDT", 10)
    book.open("S2USDT", 10.0, T)
    book.close("S2USDT", 100.0, T)  # back to +39, but the halt lasts the day
    assert not check.allows(book, "XUSDT", 10)
    book.roll_day(datetime(2024, 1, 2, tzinfo=UTC))
    assert check.allows(book, "XUSDT", 10)


def test_first_rejection_returns_the_first_failing_reason() -> None:
    checks: list[RiskCheck] = [MaxPositions(1), OnePerSymbol()]
    assert first_rejection(checks, book_with(), "S0USDT", 1) is None
    assert first_rejection(checks, book_with(1), "S0USDT", 1) == "slots_full"
    assert first_rejection(checks[1:], book_with(1), "S0USDT", 1) == "symbol_held"
