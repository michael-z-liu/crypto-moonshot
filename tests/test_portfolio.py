from datetime import UTC, datetime, timedelta

import pytest

from crypto_moonshot.portfolio import (
    Book,
    CashCapped,
    EqualSlots,
    FixedStake,
    WeightedSizing,
    WeightedSlots,
    utc_day,
)

DAY1 = datetime(2024, 1, 1, 12, tzinfo=UTC)
DAY2 = DAY1 + timedelta(days=1)


def test_book_tracks_cash_and_cost_basis() -> None:
    book = Book(cash=1_000.0)
    book.open("AUSDT", 100.0, DAY1)
    book.open("BUSDT", 200.0, DAY1)
    assert (book.cash, book.open_cost, book.equity, book.open_positions) == (700, 300, 1_000, 2)
    assert book.holds("AUSDT")
    assert book.close("AUSDT", 130.0, DAY1) == pytest.approx(30)
    assert (book.cash, book.open_cost, book.equity) == (830, 200, 1_030)
    with pytest.raises(ValueError, match="already holding"):
        book.open("BUSDT", 1.0, DAY1)


def test_book_open_cost_resets_to_zero_when_flat() -> None:
    book = Book(cash=1.0)
    book.open("AUSDT", 0.1, DAY1)
    book.open("BUSDT", 0.2, DAY1)
    book.close("AUSDT", 0.1, DAY1)
    book.close("BUSDT", 0.2, DAY1)
    assert book.open_cost == 0.0


def test_book_daily_pnl_rolls_over_at_utc_midnight() -> None:
    book = Book(cash=1_000.0)
    book.open("AUSDT", 100.0, DAY1)
    assert book.day_start_equity == 1_000
    book.open("BUSDT", 100.0, DAY1)
    book.close("AUSDT", 50.0, DAY1)
    book.close("BUSDT", 120.0, DAY1)
    assert (book.day_pnl, book.day_worst_pnl) == (-30, -50)
    midnight = datetime(2024, 1, 2, tzinfo=UTC)
    book.roll_day(midnight)
    assert utc_day(midnight) == utc_day(DAY2)
    assert (book.day_start_equity, book.day_pnl, book.day_worst_pnl) == (970, 0, 0)


def test_fixed_stake_ignores_equity() -> None:
    assert FixedStake(50).stake(Book(cash=10.0)) == 50
    with pytest.raises(ValueError, match="positive"):
        FixedStake(0)


def test_equal_slots_uses_realised_equity() -> None:
    book = Book(cash=1_000.0)
    book.open("AUSDT", 250.0, DAY1)
    assert EqualSlots(4).stake(book) == 250
    book.close("AUSDT", 450.0, DAY1)
    assert EqualSlots(4).stake(book) == 300
    with pytest.raises(ValueError, match="at least 1"):
        EqualSlots(0)


def test_cash_capped_never_exceeds_the_cash() -> None:
    book = Book(cash=1_000.0)
    book.open("AUSDT", 900.0, DAY1)
    assert EqualSlots(2).stake(book) == 500
    assert CashCapped(2).stake(book) == 100
    with pytest.raises(ValueError, match="at least 1"):
        CashCapped(0)


def test_weighted_slots_scale_by_weight_and_cap_at_cash() -> None:
    sizing = WeightedSlots(4)
    book = Book(cash=1_000.0)
    assert isinstance(sizing, WeightedSizing)
    assert sizing.stake(book) == 250
    assert sizing.stake_weighted(book, 0.5) == 125
    assert sizing.stake_weighted(book, 1.5) == 375
    book.open("AUSDT", 800.0, DAY1)
    assert sizing.stake_weighted(book, 1.5) == 200  # cash left
    with pytest.raises(ValueError, match="at least 1"):
        WeightedSlots(0)
    assert not isinstance(EqualSlots(4), WeightedSizing)


def test_book_counts_entries_per_utc_day() -> None:
    book = Book(cash=1_000.0)
    book.open("AUSDT", 10.0, DAY1)
    book.open("BUSDT", 10.0, DAY1)
    assert book.day_entries == 2
    book.roll_day(DAY1 + timedelta(days=1))
    assert book.day_entries == 0
    book.open("CUSDT", 10.0, DAY1 + timedelta(days=1))
    assert book.day_entries == 1
