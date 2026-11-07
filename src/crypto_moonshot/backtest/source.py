"""Where the backtest engine gets candles: one window per position, loaded on demand."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

import polars as pl

from crypto_moonshot.broker.candles import Candles
from crypto_moonshot.models import Interval


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
