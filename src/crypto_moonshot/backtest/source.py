"""Where the backtest engine gets candles: one window per position, loaded on demand."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl

from crypto_moonshot.broker.candles import Candles, Prices, Times, from_us, to_us
from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.models import Interval, as_utc, next_month

_US = timedelta(microseconds=1)


class CandleSource(Protocol):
    """Candle windows by symbol, interval and time span.

    The engine asks for a window per position: at the signal interval, and at the fine (1m)
    interval when fills on finer candles are wanted. Implementations should read lazily (1m
    history is far too large to hold), and must set `Candles.ended` on a window that reaches
    the end of a series that has ended, so a position still open then closes at the last close.
    """

    def window(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> Candles | None:
        """`symbol`'s candles at `interval` opening in `[start, end)`.

        `None` if there are no candles for the symbol at that interval at all (for the fine
        interval: fills then use the coarse candles).
        """
        ...


class InMemoryCandleSource:
    """A `CandleSource` over whole series held in memory (tests and small universes)."""

    def __init__(self, series: Mapping[tuple[str, Interval], Candles]) -> None:
        self._series = dict(series)

    @classmethod
    def from_frames(
        cls,
        frames: Mapping[tuple[str, Interval], pl.DataFrame],
        *,
        ended: bool = True,
    ) -> InMemoryCandleSource:
        """From kline frames (open_time and OHLC columns) keyed by (symbol, interval).

        `ended`: whether each series has ended at its last candle (a delisting, or the end of
        the data). True by default: a backtest cannot see past the data it has.
        """
        return cls(
            {
                (symbol, interval): Candles.from_frame(frame, interval, ended=ended)
                for (symbol, interval), frame in frames.items()
            }
        )

    def window(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> Candles | None:
        series = self._series.get((symbol, interval))
        return None if series is None else series.window(start, end)


_PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True, slots=True)
class _Month:
    """One month file's candles (before the source's `until`) as numpy arrays."""

    open_time: Times
    open: Prices
    high: Prices
    low: Prices
    close: Prices

    @property
    def nbytes(self) -> int:
        return 5 * self.open_time.nbytes


class ParquetCandleSource:
    """A `CandleSource` over the Parquet kline store (`docs/data-format.md`), read lazily.

    A window reads only the month files it touches. Months are kept as numpy arrays in an LRU
    cache bounded by `cache_bytes`, since a backtest's positions cluster in time.

    - `until`: the data end (exclusive). Candles opening at or after it are invisible, so a run
      can reproduce a fixed data range. Without it the data end is the latest candle close of
      any symbol in the store at that interval.
    - A window has `ended` set when it reaches past the series' last visible candle and that
      candle closes before the data end (a delisting). A series still trading at the data end
      has not ended.
    - `None` when the window touches no stored month file of that symbol and interval (no data:
      for the fine interval, fills then use the coarse candles). Months with only some candles
      give partial windows.

    Month files are read as stored, without the format checks of `KlineStore.read_month`.
    """

    def __init__(
        self,
        root: Path,
        *,
        until: datetime | None = None,
        cache_bytes: int = 2 * 1024**3,
    ) -> None:
        self.store = KlineStore(root)
        self.until = None if until is None else as_utc(until)
        self.cache_bytes = cache_bytes
        self._cache: OrderedDict[tuple[str, Interval, datetime], _Month] = OrderedDict()
        self._cached_bytes = 0
        self._months: dict[tuple[str, Interval], list[datetime]] = {}
        self._last: dict[tuple[str, Interval], int | None] = {}
        self._data_end: dict[Interval, int] = {}
        self.reads = 0
        """Month files read so far (cache misses)."""

    def window(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> Candles | None:
        start, end = as_utc(start), as_utc(end)
        if self.until is not None:
            end = min(end, self.until)
        wanted = [
            m for m in self._month_list(symbol, interval) if m < end and next_month(m) > start
        ]
        if not wanted:
            return None
        parts = [self._month(symbol, interval, m) for m in wanted]
        open_time = np.concatenate([p.open_time for p in parts])
        lo = int(np.searchsorted(open_time, to_us(start), side="left"))
        hi = int(np.searchsorted(open_time, to_us(end), side="left"))
        last = self._last_open(symbol, interval)
        ended = False
        if last is not None and to_us(end) > last:
            ended = last + interval.length // _US < self._end_us(interval)
        return Candles(
            interval=interval,
            open_time=open_time[lo:hi],
            open=np.concatenate([p.open for p in parts])[lo:hi],
            high=np.concatenate([p.high for p in parts])[lo:hi],
            low=np.concatenate([p.low for p in parts])[lo:hi],
            close=np.concatenate([p.close for p in parts])[lo:hi],
            ended=ended,
        )

    def data_end(self, interval: Interval) -> datetime:
        """The data end at `interval`: `until`, or the latest candle close in the store if that
        is earlier (or there is no `until`)."""
        return from_us(self._end_us(interval))

    def _month_list(self, symbol: str, interval: Interval) -> list[datetime]:
        key = (symbol, interval)
        months = self._months.get(key)
        if months is None:
            months = self.store.list_months(symbol, interval)
            if self.until is not None:
                months = [m for m in months if m < self.until]
            self._months[key] = months
        return months

    def _month(self, symbol: str, interval: Interval, month: datetime) -> _Month:
        key = (symbol, interval, month)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        path = self.store.month_path(symbol, interval, month)
        frame = pl.read_parquet(path, columns=["open_time", *_PRICE_COLUMNS])
        self.reads += 1
        if self.until is not None and next_month(month) > self.until:
            frame = frame.filter(pl.col("open_time") < self.until)
        frame = frame.sort("open_time")
        loaded = _Month(
            open_time=frame["open_time"].dt.epoch("us").to_numpy().astype(np.int64),
            open=frame["open"].to_numpy().astype(np.float64),
            high=frame["high"].to_numpy().astype(np.float64),
            low=frame["low"].to_numpy().astype(np.float64),
            close=frame["close"].to_numpy().astype(np.float64),
        )
        self._cache[key] = loaded
        self._cached_bytes += loaded.nbytes
        while self._cached_bytes > self.cache_bytes and len(self._cache) > 1:
            _, evicted = self._cache.popitem(last=False)
            self._cached_bytes -= evicted.nbytes
        return loaded

    def _last_open(self, symbol: str, interval: Interval) -> int | None:
        """Open time (µs) of the series' last visible candle (reads only `open_time`)."""
        key = (symbol, interval)
        if key not in self._last:
            last: int | None = None
            for month in reversed(self._month_list(symbol, interval)):
                path = self.store.month_path(symbol, interval, month)
                times = pl.read_parquet(path, columns=["open_time"])["open_time"]
                if self.until is not None:
                    times = times.filter(times < self.until)
                if len(times):
                    latest = times.max()
                    assert isinstance(latest, datetime)
                    last = to_us(latest)
                    break
            self._last[key] = last
        return self._last[key]

    def _end_us(self, interval: Interval) -> int:
        end = self._data_end.get(interval)
        if end is None:
            length = interval.length // _US
            closes = [
                last + length
                for symbol, iv in self.store.list_series()
                if iv == interval and (last := self._last_open(symbol, iv)) is not None
            ]
            end = max(closes, default=0)
            if self.until is not None:
                end = min(end, to_us(self.until)) if closes else to_us(self.until)
            self._data_end[interval] = end
        return end
