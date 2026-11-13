"""M1: per coin, 1h/4h/1d/1w/1M bars with every study and placebo SMA level, `atr_d` and `liq`.

    uv run --group research python build_bars.py [--symbols BTCUSDT ...] [--force]

Bars are aggregated from the stored 5m candles on UTC boundaries; before 2020 the 1d bars come
from `cache/daily_pre2020` and the 1w/1M bars from those. Column `L<n>` is the level in force
during the bar (the mean close of the n bars before it); the 1d table also has `atr_d` and `liq`
in force during the day. Writes `cache/parts/bars_<tf>/<SYMBOL>.parquet` (not combined).
"""

from __future__ import annotations

import polars as pl
from events import TF_NAMES, build_bars, read_5m, read_daily_pre2020
from runner import END_US, run

TABLES = tuple(f"bars_{tf}" for tf in TF_NAMES)


def step(symbol: str) -> dict[str, pl.DataFrame]:
    bars = build_bars(read_5m(symbol, END_US), read_daily_pre2020(symbol, END_US))
    return {f"bars_{tf}": df for tf, df in bars.items()}


if __name__ == "__main__":
    run(step, TABLES, __doc__ or "", combine=False)
