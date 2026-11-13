"""M1: the candidates (setups) of every study and placebo MA, for both entry families.

    uv run --group research python build_candidates.py [--symbols BTCUSDT ...] [--force]

Limit family: the trigger in the approach zone (armed at L(1 + 0.45u), trigger at the first 5m
close at or below L(1 + 0.30u) while above). Touch-market family: the first 5m candle whose low
reaches L while above. Both need `atr_d` defined and `liq` >= $1M at the trigger. Reads the bars
of `build_bars.py`; writes `cache/parts/candidates/<SYMBOL>.parquet` and
`cache/candidates.parquet`.
"""

from __future__ import annotations

import polars as pl
from events import build_candidates, read_5m, read_bars
from runner import END_US, run


def step(symbol: str) -> dict[str, pl.DataFrame]:
    return {"candidates": build_candidates(symbol, read_5m(symbol, END_US), read_bars(symbol))}


if __name__ == "__main__":
    run(step, ("candidates",), __doc__ or "")
