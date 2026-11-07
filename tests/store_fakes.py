"""Synthetic kline series written through the repo's `KlineStore`."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.models import KLINE_SCHEMA, Interval


def kline_frame(
    times: Sequence[datetime],
    close: Sequence[float] | np.ndarray,
    volume: Sequence[float] | np.ndarray | None = None,
    *,
    open_: Sequence[float] | np.ndarray | None = None,
    high: Sequence[float] | np.ndarray | None = None,
    low: Sequence[float] | np.ndarray | None = None,
    quote_volume: Sequence[float] | np.ndarray | None = None,
) -> pl.DataFrame:
    """A `KLINE_SCHEMA` frame. Defaults: open = close, high/low = close +- 1%, volume 1,
    quote volume = volume x close."""
    c = np.asarray(close, dtype=np.float64)
    v = np.ones(len(c)) if volume is None else np.asarray(volume, dtype=np.float64)
    o = c if open_ is None else np.asarray(open_, dtype=np.float64)
    h = c * 1.01 if high is None else np.asarray(high, dtype=np.float64)
    lo = c * 0.99 if low is None else np.asarray(low, dtype=np.float64)
    qv = v * c if quote_volume is None else np.asarray(quote_volume, dtype=np.float64)
    return pl.DataFrame(
        {
            "open_time": list(times),
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": v,
            "quote_volume": qv,
            "trade_count": np.ones(len(c), dtype=np.int64),
            "taker_buy_base_volume": v / 2,
            "taker_buy_quote_volume": qv / 2,
        },
        schema=KLINE_SCHEMA,
    )


def write_series(root: Path, symbol: str, interval: Interval, frame: pl.DataFrame) -> None:
    """Write `frame` into month files under the data root `root`."""
    store = KlineStore(root)
    months = frame.with_columns(_month=pl.col("open_time").dt.truncate("1mo"))
    for (month,), part in months.group_by("_month", maintain_order=True):
        assert isinstance(month, datetime)
        store.write_month(symbol, interval, month, part.drop("_month"))
