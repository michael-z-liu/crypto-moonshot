from datetime import UTC, datetime, timedelta, timezone

import pytest

from crypto_moonshot.models import Interval


def utc(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


@pytest.mark.parametrize(
    ("interval", "dt", "floor", "next_"),
    [
        (Interval.M5, utc(2024, 3, 5, 10, 7, 30), utc(2024, 3, 5, 10, 5), utc(2024, 3, 5, 10, 10)),
        (Interval.M5, utc(2024, 3, 5, 10, 5), utc(2024, 3, 5, 10, 5), utc(2024, 3, 5, 10, 10)),
        (Interval.H1, utc(2023, 12, 31, 23, 59), utc(2023, 12, 31, 23), utc(2024, 1, 1)),
        (Interval.D1, utc(2024, 2, 28, 5), utc(2024, 2, 28), utc(2024, 2, 29)),
        (Interval.S1, utc(2024, 1, 1, 0, 0, 5), utc(2024, 1, 1, 0, 0, 5), utc(2024, 1, 1, 0, 0, 6)),
    ],
)
def test_fixed_interval_floor_and_next(
    interval: Interval, dt: datetime, floor: datetime, next_: datetime
) -> None:
    assert interval.floor(dt) == floor
    assert interval.next(dt) == next_
    assert interval.is_aligned(floor)
    assert interval.is_aligned(dt) == (dt == floor)


def test_three_day_candles_align_to_the_unix_epoch() -> None:
    # Binance's 3d candles open on multiples of 3 days since 1970-01-01.
    assert Interval.D3.floor(utc(1970, 1, 3, 23)) == utc(1970, 1, 1)
    assert Interval.D3.is_aligned(utc(1970, 1, 4))
    assert Interval.D3.floor(utc(2024, 1, 1)) == utc(2023, 12, 31)
    assert Interval.D3.next(utc(2023, 12, 31)) == utc(2024, 1, 3)
    # A year boundary falls mid-candle.
    assert Interval.D3.floor(utc(2024, 1, 2, 12)) == utc(2023, 12, 31)


def test_weekly_candles_open_on_monday() -> None:
    monday = utc(2024, 1, 1)
    assert monday.weekday() == 0
    assert Interval.W1.floor(utc(2024, 1, 7, 23, 59)) == monday
    assert Interval.W1.floor(monday) == monday
    assert Interval.W1.next(monday) == utc(2024, 1, 8)
    assert Interval.W1.is_aligned(monday)
    assert not Interval.W1.is_aligned(utc(2024, 1, 4))  # a Thursday
    # Across a year boundary.
    assert Interval.W1.floor(utc(2025, 1, 1)) == utc(2024, 12, 30)
    assert Interval.W1.next(utc(2024, 12, 30)) == utc(2025, 1, 6)


def test_monthly_candles_open_on_the_first() -> None:
    assert Interval.MO1.floor(utc(2024, 2, 29, 23, 59)) == utc(2024, 2, 1)
    assert Interval.MO1.next(utc(2024, 2, 1)) == utc(2024, 3, 1)
    assert Interval.MO1.next(utc(2024, 12, 1)) == utc(2025, 1, 1)
    assert Interval.MO1.floor(utc(2024, 12, 31, 23)) == utc(2024, 12, 1)
    assert Interval.MO1.is_aligned(utc(2025, 1, 1))
    assert not Interval.MO1.is_aligned(utc(2025, 1, 1, 0, 0, 1))


def test_non_utc_aware_datetimes_are_converted() -> None:
    plus_two = timezone(timedelta(hours=2))
    dt = datetime(2024, 1, 1, 1, 30, tzinfo=plus_two)  # 2023-12-31 23:30 UTC
    assert Interval.H1.floor(dt) == utc(2023, 12, 31, 23)
    assert Interval.H1.floor(dt).tzinfo is UTC


@pytest.mark.parametrize("interval", list(Interval))
def test_naive_datetimes_are_rejected(interval: Interval) -> None:
    naive = datetime(2024, 1, 1)  # noqa: DTZ001
    with pytest.raises(ValueError, match="timezone-aware"):
        interval.floor(naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        interval.next(naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        interval.is_aligned(naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        interval.count_open_times(naive, utc(2024, 2, 1))
    with pytest.raises(ValueError, match="timezone-aware"):
        list(interval.open_times(utc(2024, 1, 1), naive))


def test_all_binance_intervals_present() -> None:
    expected = ["1s", "1m", "3m", "5m", "15m", "30m"]
    expected += ["1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
    assert [i.value for i in Interval] == expected
    assert Interval("1M") is Interval.MO1
    assert Interval("1m") is Interval.M1


def test_count_and_open_times_for_fixed_intervals() -> None:
    start, end = utc(2024, 1, 1), utc(2024, 1, 2)
    assert Interval.H1.count_open_times(start, end) == 24
    assert Interval.M5.count_open_times(start, end) == 288
    assert list(Interval.H1.open_times(start, end))[-1] == utc(2024, 1, 1, 23)
    # Unaligned bounds: only open times inside [start, end) count.
    assert Interval.H1.count_open_times(utc(2024, 1, 1, 0, 30), utc(2024, 1, 1, 3, 30)) == 3
    assert list(Interval.H1.open_times(utc(2024, 1, 1, 0, 30), utc(2024, 1, 1, 3, 30))) == [
        utc(2024, 1, 1, 1),
        utc(2024, 1, 1, 2),
        utc(2024, 1, 1, 3),
    ]
    # End is exclusive; empty and inverted ranges are zero.
    assert Interval.H1.count_open_times(start, start) == 0
    assert Interval.H1.count_open_times(end, start) == 0
    assert Interval.H1.count_open_times(utc(2024, 1, 1, 0, 10), utc(2024, 1, 1, 0, 50)) == 0
    # 3d across a year boundary: 2023-12-31 and 2024-01-03.
    assert list(Interval.D3.open_times(utc(2023, 12, 31), utc(2024, 1, 4))) == [
        utc(2023, 12, 31),
        utc(2024, 1, 3),
    ]
    assert Interval.D3.count_open_times(utc(2023, 12, 31), utc(2024, 1, 4)) == 2


def test_count_and_open_times_for_weekly_and_monthly() -> None:
    assert Interval.W1.count_open_times(utc(2024, 12, 23), utc(2025, 1, 13)) == 3
    assert list(Interval.W1.open_times(utc(2024, 12, 24), utc(2025, 1, 7))) == [
        utc(2024, 12, 30),
        utc(2025, 1, 6),
    ]
    assert Interval.MO1.count_open_times(utc(2023, 11, 1), utc(2024, 3, 1)) == 4
    assert Interval.MO1.count_open_times(utc(2023, 11, 1), utc(2024, 3, 1, 0, 0, 1)) == 5
    assert (
        Interval.MO1.count_open_times(utc(2023, 11, 15), utc(2024, 3, 15)) == 4
    )  # Dec, Jan, Feb, Mar
    assert Interval.MO1.count_open_times(utc(2024, 1, 1), utc(2024, 1, 31)) == 1
    assert Interval.MO1.count_open_times(utc(2024, 1, 2), utc(2024, 1, 31)) == 0
    assert list(Interval.MO1.open_times(utc(2023, 11, 20), utc(2024, 2, 2))) == [
        utc(2023, 12, 1),
        utc(2024, 1, 1),
        utc(2024, 2, 1),
    ]


@pytest.mark.parametrize("interval", list(Interval))
def test_count_matches_enumeration(interval: Interval) -> None:
    start = utc(2023, 12, 17, 3, 21, 9)
    ends = [utc(2024, 1, 1), utc(2024, 3, 5, 7, 11), utc(2025, 1, 1, 0, 0, 1)]
    if interval is Interval.S1:
        ends = [start + timedelta(minutes=5, milliseconds=300)]
    for end in ends:
        assert interval.count_open_times(start, end) == len(list(interval.open_times(start, end)))


def test_length() -> None:
    assert Interval.D3.length == timedelta(days=3)
    with pytest.raises(ValueError, match="no fixed length"):
        _ = Interval.MO1.length
