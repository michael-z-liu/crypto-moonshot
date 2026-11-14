"""M1: the coin-day panel of eligible entries (no outcomes).

    uv run --group research python build_panel.py [--workers 8]

Writes `cache/panel.parquet`: one row per eligible entry, for every year up to the last `D` whose
30-day window is complete by the data end. The protocol is in `docs/plans/rebound-factors.md`:

- `D` is every UTC midnight from 2020-01-31 to `LAST_D` (2026-10-03, since the window
  `[D, D + 30 days)` then ends exactly at the data end, 2026-11-02 00:00 UTC exclusive).
- `liq`: total quote volume of the 5m candles in the 30 complete UTC days before `D`, over 30
  (days without candles count as zero). Eligible when `liq >= 1,000,000`.
- `days_listed`: UTC days from the coin's first candle to `D` (the first daily candle before
  2020 for coins that traded then, from the pre-2020 daily candles `ma_rebound` downloaded,
  else the first 5m candle). Eligible when at least 30.
- `P`: the close of the last 5m candle before `D`. Added here, as the sensible reading of the
  protocol for halts and delistings: that candle must have opened within the day before `D`
  (a coin with no trade in the last 24 hours, e.g. one delisted days ago whose 30-day mean
  volume still clears the floor, has no current price and is not an entry).

Every column uses only candles that closed by `D`; `check_panel.py` rebuilds coins with the
candles cut at `D` and checks the rows are identical.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from common import (
    DAY,
    F64,
    FIRST_D_US,
    I64,
    LAST_D_US,
    LIQUIDITY_DAYS,
    MIN_DAYS_LISTED,
    MIN_LIQUIDITY,
    PANEL,
    STALE,
    first_day_pre2020,
    read_5m,
    split_expr,
    universe,
)

PANEL_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String(),
    "d_us": pl.Int64(),
    "D": pl.Datetime("us", "UTC"),
    "P": pl.Float64(),
    "liq": pl.Float64(),
    "days_listed": pl.Int32(),
    "year": pl.Int32(),
    "split": pl.String(),
}


def panel_rows(
    symbol: str, t: I64, close: F64, quote_volume: F64, pre_first_day: int | None
) -> pl.DataFrame:
    """The eligible entries of one coin from its 5m candles (sorted by open time `t`)."""
    if len(t) == 0:
        return pl.DataFrame(schema=PANEL_SCHEMA)
    day = t // DAY
    d0 = int(day[0])
    first_day = d0 if pre_first_day is None else min(d0, pre_first_day)
    lo_day = max(FIRST_D_US // DAY, first_day + MIN_DAYS_LISTED)
    hi_day = min(LAST_D_US // DAY, int(day[-1]) + 1)  # P needs a candle in the day before D
    if hi_day < lo_day:
        return pl.DataFrame(schema=PANEL_SCHEMA)
    daily = np.bincount(day - d0, weights=quote_volume)
    cs = np.concatenate(([0.0], np.cumsum(daily)))
    days = np.arange(lo_day, hi_day + 1, dtype=np.int64)
    # Sum of the days [D - 30, D): prefix sums at the offsets, clipped to the series' range.
    a = np.clip(days - LIQUIDITY_DAYS - d0, 0, len(daily))
    b = np.clip(days - d0, 0, len(daily))
    liq = (cs[b] - cs[a]) / LIQUIDITY_DAYS
    d_us = days * DAY
    last = np.searchsorted(t, d_us, side="left") - 1
    has_p = last >= 0
    lastc = np.maximum(last, 0)
    fresh = has_p & (t[lastc] >= d_us - STALE)
    keep = fresh & (liq >= MIN_LIQUIDITY)
    df = pl.DataFrame(
        {
            "symbol": symbol,
            "d_us": d_us[keep],
            "P": close[lastc][keep],
            "liq": liq[keep],
            "days_listed": (days - first_day)[keep].astype(np.int32),
        }
    )
    ts = pl.from_epoch(pl.col("d_us"), time_unit="us").dt.replace_time_zone("UTC")
    df = df.with_columns(D=ts, year=ts.dt.year().cast(pl.Int32), split=split_expr())
    return df.select(list(PANEL_SCHEMA)).cast(pl.Schema(PANEL_SCHEMA))


def coin_panel(symbol: str) -> pl.DataFrame:
    c5 = read_5m(symbol)
    return panel_rows(
        symbol,
        c5["t"].to_numpy().astype(np.int64),
        c5["close"].to_numpy().astype(np.float64),
        c5["quote_volume"].to_numpy().astype(np.float64),
        first_day_pre2020(symbol),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    symbols = universe()
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        parts = list(pool.map(coin_panel, symbols, chunksize=4))
    panel = pl.concat(parts).sort("d_us", "symbol")
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    tmp = PANEL.with_name(f".{PANEL.name}.tmp")
    panel.write_parquet(tmp)
    tmp.replace(PANEL)
    secs = time.perf_counter() - t0
    print(f"{len(symbols)} coins, {panel['symbol'].n_unique()} with entries")
    print(f"panel.parquet: {panel.height} rows in {secs:.0f}s")
    print(panel.group_by("split").len().sort("split"))


if __name__ == "__main__":
    main()
