"""Core domain types shared across the toolkit."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import polars as pl

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
# The Unix epoch is a Thursday; this is the first Monday after it.
_FIRST_MONDAY = datetime(1970, 1, 5, tzinfo=UTC)

SCHEMA_VERSION = 1
"""Version of the stored kline schema (see `KLINE_SCHEMA`)."""

KLINE_SCHEMA: dict[str, pl.DataType] = {
    "open_time": pl.Datetime("us", "UTC"),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "quote_volume": pl.Float64(),
    "trade_count": pl.Int64(),
    "taker_buy_base_volume": pl.Float64(),
    "taker_buy_quote_volume": pl.Float64(),
}
"""Column names and dtypes of a kline frame, in order (schema v1)."""


def as_utc(dt: datetime) -> datetime:
    """Return `dt` in UTC; naive datetimes are rejected."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        msg = f"datetime must be timezone-aware, got naive {dt!r}"
        raise ValueError(msg)
    return dt.astimezone(UTC)


def month_start(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1, tzinfo=UTC)


def next_month(dt: datetime) -> datetime:
    return datetime(dt.year + dt.month // 12, dt.month % 12 + 1, 1, tzinfo=UTC)


class Interval(StrEnum):
    """Binance spot kline intervals, with candle open-time alignment.

    Candles open at multiples of their length since the Unix epoch, except `1w` (opens on
    Monday 00:00 UTC) and `1M` (opens on the 1st of the month 00:00 UTC).
    """

    S1 = "1s"
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"
    MO1 = "1M"

    @property
    def is_monthly(self) -> bool:
        return self is Interval.MO1

    @property
    def length(self) -> timedelta:
        """Candle length; raises for the calendar-based `1M`."""
        if self.is_monthly:
            msg = "1M candles have no fixed length"
            raise ValueError(msg)
        return _FIXED_LENGTHS[self]

    def floor(self, dt: datetime) -> datetime:
        """Open time of the latest candle that opens at or before `dt`."""
        dt = as_utc(dt)
        if self.is_monthly:
            return month_start(dt)
        length = self.length
        anchor = _FIRST_MONDAY if self is Interval.W1 else EPOCH
        return anchor + (dt - anchor) // length * length

    def next(self, dt: datetime) -> datetime:
        """Open time of the candle after the one containing `dt` (after `dt` if aligned)."""
        open_time = self.floor(dt)
        if self.is_monthly:
            return next_month(open_time)
        return open_time + self.length

    def ceil(self, dt: datetime) -> datetime:
        """Earliest candle open time at or after `dt`."""
        dt = as_utc(dt)
        floored = self.floor(dt)
        return floored if floored == dt else self.next(floored)

    def is_aligned(self, dt: datetime) -> bool:
        """Whether `dt` is exactly a candle open time."""
        dt = as_utc(dt)
        return self.floor(dt) == dt

    def open_times(self, start: datetime, end: datetime) -> Iterator[datetime]:
        """Yield every candle open time `t` with `start <= t < end`, in order."""
        end = as_utc(end)
        current = self.ceil(start)
        while current < end:
            yield current
            current = self.next(current)

    def count_open_times(self, start: datetime, end: datetime) -> int:
        """Number of candle open times in the half-open range `[start, end)`."""
        start, end = as_utc(start), as_utc(end)
        if end <= start:
            return 0
        first = self.ceil(start)
        if first >= end:
            return 0
        if self.is_monthly:
            return (
                (end.year - first.year) * 12 + end.month - first.month + int(end > self.floor(end))
            )
        # Number of aligned times in [first, end): ceil((end - first) / length).
        return -((first - end) // self.length)


_FIXED_LENGTHS: dict[Interval, timedelta] = {
    Interval.S1: timedelta(seconds=1),
    Interval.M1: timedelta(minutes=1),
    Interval.M3: timedelta(minutes=3),
    Interval.M5: timedelta(minutes=5),
    Interval.M15: timedelta(minutes=15),
    Interval.M30: timedelta(minutes=30),
    Interval.H1: timedelta(hours=1),
    Interval.H2: timedelta(hours=2),
    Interval.H4: timedelta(hours=4),
    Interval.H6: timedelta(hours=6),
    Interval.H8: timedelta(hours=8),
    Interval.H12: timedelta(hours=12),
    Interval.D1: timedelta(days=1),
    Interval.D3: timedelta(days=3),
    Interval.W1: timedelta(weeks=1),
}
