"""Candle series as numpy arrays, for fast fill simulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt
import polars as pl

from crypto_moonshot.models import EPOCH, Interval, as_utc

type Times = npt.NDArray[np.int64]
type Prices = npt.NDArray[np.float64]

_US = timedelta(microseconds=1)


def to_us(dt: datetime) -> int:
    """Microseconds since the Unix epoch of an aware datetime."""
    return (as_utc(dt) - EPOCH) // _US


def from_us(us: int) -> datetime:
    """Aware UTC datetime from microseconds since the Unix epoch."""
    return EPOCH + timedelta(microseconds=us)


@dataclass(frozen=True, slots=True)
class Candles:
    """One symbol's candles at one interval: open times (µs since epoch, UTC) and OHLC prices.

    Open times are strictly increasing and aligned to `interval`; candles may be missing (holes).
    `ended` says whether the series has no candles after the last one held here: true for a
    whole series loaded to its end (a delisting, or the end of the data), false for a window cut
    out of a longer series. A fill simulation that runs out of candles closes the position at
    the last close only when the series has ended.
    """

    interval: Interval
    open_time: Times
    open: Prices
    high: Prices
    low: Prices
    close: Prices
    ended: bool = True

    def __post_init__(self) -> None:
        n = len(self.open_time)
        if any(len(a) != n for a in (self.open, self.high, self.low, self.close)):
            msg = "candle arrays must have equal lengths"
            raise ValueError(msg)

    @classmethod
    def from_frame(cls, frame: pl.DataFrame, interval: Interval, *, ended: bool = True) -> Candles:
        """Build from a kline frame (`KLINE_SCHEMA` columns; only open_time and OHLC are used)."""
        frame = frame.sort("open_time")
        return cls(
            interval=interval,
            open_time=frame["open_time"].dt.epoch("us").to_numpy().astype(np.int64),
            open=frame["open"].to_numpy().astype(np.float64),
            high=frame["high"].to_numpy().astype(np.float64),
            low=frame["low"].to_numpy().astype(np.float64),
            close=frame["close"].to_numpy().astype(np.float64),
            ended=ended,
        )

    def __len__(self) -> int:
        return len(self.open_time)

    @property
    def length_us(self) -> int:
        """Candle length in microseconds."""
        return self.interval.length // _US

    def index_at_or_after(self, dt: datetime) -> int:
        """Index of the first candle opening at or after `dt` (`len(self)` if none)."""
        return int(np.searchsorted(self.open_time, to_us(dt), side="left"))

    def window(self, start: datetime, end: datetime | None = None) -> Candles:
        """Candles opening in `[start, end)` (to the end of the series if `end` is None).

        The window has `ended` set only if it reaches the end of an ended series.
        """
        lo = self.index_at_or_after(start)
        hi = len(self) if end is None else self.index_at_or_after(end)
        hi = max(hi, lo)
        return Candles(
            interval=self.interval,
            open_time=self.open_time[lo:hi],
            open=self.open[lo:hi],
            high=self.high[lo:hi],
            low=self.low[lo:hi],
            close=self.close[lo:hi],
            ended=self.ended and hi == len(self),
        )
