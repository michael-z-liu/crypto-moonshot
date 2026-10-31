"""What is stored for each series, checked against what should be there (read-only).

`moonshot data status` is a thin layer over `store_report`. It never touches the network and never
writes: it reads month files, `_meta.json` and the folder listing only.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from crypto_moonshot.data.store import KlineStore, Range, missing_ranges
from crypto_moonshot.models import Interval, as_utc


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else as_utc(dt).isoformat()


@dataclass(frozen=True)
class SeriesReport:
    """Stored-data health of one (symbol, interval) series."""

    symbol: str
    interval: Interval
    listing: datetime | None
    """First open time Binance has for the symbol, if sync looked it up."""
    first_open_time: datetime | None
    last_open_time: datetime | None
    candles_stored: int
    """Candles in valid month files."""
    expected_candles: int
    """Open times from `max(listing, first stored)` up to and including the last stored candle."""
    missing_candles: int
    """Candles inside that span that are neither stored nor recorded as gaps."""
    gaps: tuple[Range, ...]
    """Recorded gaps (`[start, end)` open-time ranges Binance has no candles for)."""
    gap_candles: int
    month_files: int
    """Month files present, valid or not."""
    invalid_files: tuple[tuple[datetime, tuple[str, ...]], ...]
    """Month files that fail the format checks, with their problems."""
    last_candle_age_s: float | None
    """Seconds since the last stored candle opened."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict: datetimes as ISO-8601 strings, enums as their values."""
        return {
            "symbol": self.symbol,
            "interval": self.interval.value,
            "listing": _iso(self.listing),
            "first_open_time": _iso(self.first_open_time),
            "last_open_time": _iso(self.last_open_time),
            "candles_stored": self.candles_stored,
            "expected_candles": self.expected_candles,
            "missing_candles": self.missing_candles,
            "gap_ranges": len(self.gaps),
            "gap_candles": self.gap_candles,
            "gaps": [
                {
                    "start": _iso(start),
                    "end": _iso(end),
                    "candles": self.interval.count_open_times(start, end),
                }
                for start, end in self.gaps
            ],
            "month_files": self.month_files,
            "invalid_files": [
                {"month": f"{month:%Y-%m}", "problems": list(problems)}
                for month, problems in self.invalid_files
            ],
            "last_candle_age_s": (
                None if self.last_candle_age_s is None else round(self.last_candle_age_s, 1)
            ),
        }


def series_report(
    store: KlineStore, symbol: str, interval: Interval, *, now: datetime | None = None
) -> SeriesReport:
    """Inspect one series. Invalid month files are reported, not counted as stored candles."""
    now = as_utc(now) if now is not None else datetime.now(UTC)
    months = store.list_months(symbol, interval)
    invalid = store.invalid_months(symbol, interval, now=now)
    valid = [month for month in months if month not in invalid]
    stored = store.stored_open_times(symbol, interval, months=valid)
    meta = store.read_meta(symbol, interval)
    gaps = tuple(meta.gaps)

    first = last = None
    expected = missing = 0
    if stored.len():
        first_value, last_value = stored.min(), stored.max()
        assert isinstance(first_value, datetime)
        assert isinstance(last_value, datetime)
        first, last = as_utc(first_value), as_utc(last_value)
        lo = max(meta.listing, first) if meta.listing is not None else first
        hi = interval.next(last)
        expected = interval.count_open_times(lo, hi)
        holes = missing_ranges(interval, lo, hi, stored, gaps)
        missing = sum(interval.count_open_times(start, end) for start, end in holes)

    return SeriesReport(
        symbol=symbol,
        interval=interval,
        listing=meta.listing,
        first_open_time=first,
        last_open_time=last,
        candles_stored=stored.len(),
        expected_candles=expected,
        missing_candles=missing,
        gaps=gaps,
        gap_candles=sum(interval.count_open_times(start, end) for start, end in gaps),
        month_files=len(months),
        invalid_files=tuple((month, tuple(problems)) for month, problems in invalid.items()),
        last_candle_age_s=None if last is None else (now - last).total_seconds(),
    )


def store_report(
    store: KlineStore,
    symbols: Collection[str] = (),
    intervals: Collection[Interval] = (),
    *,
    now: datetime | None = None,
) -> list[SeriesReport]:
    """Reports for every stored series, optionally limited to some symbols and intervals."""
    now = as_utc(now) if now is not None else datetime.now(UTC)
    return [
        series_report(store, symbol, interval, now=now)
        for symbol, interval in store.list_series()
        if (not symbols or symbol in symbols) and (not intervals or interval in intervals)
    ]
