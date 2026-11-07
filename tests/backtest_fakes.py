"""Synthetic signals and candle sources for backtest engine tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

import polars as pl
from broker_fakes import T0, Bar, candles

from crypto_moonshot.backtest.source import InMemoryCandleSource
from crypto_moonshot.broker.base import Costs
from crypto_moonshot.models import Interval

NO_COSTS = Costs(fee=0.0, slippage=0.0)


def source(
    series: Mapping[str, Sequence[Bar | None]],
    *,
    interval: Interval = Interval.M5,
    start: datetime = T0,
    ended: bool = True,
    fine: Mapping[str, Sequence[Bar | None]] | None = None,
    data_end: datetime | None = None,
) -> InMemoryCandleSource:
    """One series per symbol from (open, high, low, close) bars starting at `start`; `fine`
    adds 1m series for the same symbols; `data_end` is the data end the source reports."""
    out = {
        (symbol, interval): candles(bars, interval=interval, start=start, ended=ended)
        for symbol, bars in series.items()
    }
    for symbol, bars in (fine or {}).items():
        out[symbol, Interval.M1] = candles(bars, interval=Interval.M1, start=start, ended=ended)
    return InMemoryCandleSource(out, data_end=data_end)


def signals(
    *rows: tuple[str, int, float],
    interval: Interval = Interval.M5,
    start: datetime = T0,
    **extra: Sequence[object],
) -> pl.DataFrame:
    """Signals from (symbol, signal candle index, priority) rows, plus extra columns."""
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "open_time": [start + r[1] * interval.length for r in rows],
            "priority": [r[2] for r in rows],
            **extra,
        },
        schema_overrides={"open_time": pl.Datetime("us", "UTC")},
    )
