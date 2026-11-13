"""M1: exit labels E1..E6 of every filled entry through the production `resolve_exit`, and the
forward paths.

    uv run --group research python build_labels.py [--symbols BTCUSDT ...] [--force]

Combinations: study MAs, limit family: 3 buffers x E1..E6; touch-market: E1..E6; placebo MAs:
limit buffer 0.05 with E1 and E3, touch-market E1. Each in the base scenario (the base fill) and
the stress (the stress fill, 0.1% more cost per side). Forward path (study MAs, base, primary
buffer and touch-market): 32 closes over T and the MFE / MAE, in units of u. Writes
`cache/parts/{labels,fwd}/<SYMBOL>.parquet` and `cache/labels.parquet`, `cache/fwd.parquet`.

2025-26 outcomes are in these tables: never print or aggregate them before the M6 reveal.
"""

from __future__ import annotations

import polars as pl
from events import build_labels, make_series, read_1m, read_5m
from runner import END_US, read_part, run


def step(symbol: str) -> dict[str, pl.DataFrame]:
    series = make_series(symbol, read_5m(symbol, END_US), read_1m(symbol, END_US), END_US)
    labels, fwd = build_labels(read_part("candidates", symbol), read_part("fills", symbol), series)
    return {"labels": labels, "fwd": fwd}


if __name__ == "__main__":
    run(step, ("labels", "fwd"), __doc__ or "", show_counts=False)
