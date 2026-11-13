"""M1: entry fills: limit fills per buffer on 1m candles (base and stress), touch-market entries.

    uv run --group research python build_fills.py [--symbols BTCUSDT ...] [--force]

A limit order is placed at the trigger candle's close at L(1 + b u), follows L as bars complete
(u stays the trigger's), and is valid for T/2. It fills at the limit price in the first 1m candle
(5m where the 1m candles are incomplete) whose low reaches it; under the stress only when the low
is 5 bp through it. If the trigger close is already at or below the limit, it fills at the next
open as a taker. Touch-market entries are market buys at the next 5m open. Writes
`cache/parts/fills/<SYMBOL>.parquet` and `cache/fills.parquet`.
"""

from __future__ import annotations

import polars as pl
from events import build_fills, make_series, read_1m, read_5m, read_bars
from runner import END_US, read_part, run


def step(symbol: str) -> dict[str, pl.DataFrame]:
    series = make_series(symbol, read_5m(symbol, END_US), read_1m(symbol, END_US), END_US)
    return {"fills": build_fills(read_part("candidates", symbol), read_bars(symbol), series)}


if __name__ == "__main__":
    run(step, ("fills",), __doc__ or "", show_counts=False)
