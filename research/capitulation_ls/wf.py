"""Walk-forward rule selection: for each test year, pick the configuration with the best record on
all earlier years (from 2021), then trade it in the test year. The realised walk-forward book is
the chain of test years.

Selection score (fixed 2026-11-27, before any walk-forward result): the mean over earlier years of
log(1 + yearly return), minus 0.5 x |worst max drawdown| over those years. Ties: more trades.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import polars as pl

YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)


def table(run: Callable[[dict[str, Any]], pl.DataFrame], grid: Iterable[dict[str, Any]], workers: int = 8) -> pl.DataFrame:
    """run(config) -> per-year frame (year, trades, bp, ret, dd); returns one row per config-year."""
    grid = list(grid)
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(run, grid))
    rows = []
    for i, (cfg, yrs) in enumerate(zip(grid, parts, strict=True)):
        for r in yrs.iter_rows(named=True):
            rows.append({"cfg": i, "label": str(cfg), **r})
    return pl.DataFrame(rows)


def select(tab: pl.DataFrame, test_year: int, first_year: int = 2021) -> int:
    past = tab.filter((pl.col("year") >= first_year) & (pl.col("year") < test_year))
    g = past.group_by("cfg").agg(
        score=(pl.col("ret").clip(-0.99, None).log1p()).mean() - 0.5 * pl.col("dd").min().abs(),
        n=pl.col("trades").sum(), years=pl.len(),
    )  # fmt: skip
    need = test_year - first_year
    g = g.filter(pl.col("years") == need).sort(["score", "n"], descending=True)
    return int(g["cfg"][0])


def walk_forward(tab: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for y in TEST_YEARS:
        c = select(tab, y)
        r = tab.filter((pl.col("cfg") == c) & (pl.col("year") == y))
        if r.height:
            rows.append(r.row(0, named=True))
    out = pl.DataFrame(rows)
    return out


def summarize(wf: pl.DataFrame) -> str:
    eq = float(np.prod(1 + wf["ret"].to_numpy()))
    n = int(wf["trades"].sum())
    bp = float((wf["bp"] * wf["trades"]).sum() / max(n, 1))
    cagr = eq ** (1 / (len(wf) - 2 / 12)) - 1 if len(wf) else 0.0  # 2026 runs to Nov
    lines = [f"  {r['year']}: {r['trades']:>4}t {r['bp']:+6.0f}bp ret {r['ret'] * 100:+6.0f}% dd {r['dd'] * 100:4.0f}%  <- {r['label']}" for r in wf.iter_rows(named=True)]
    return "\n".join(lines) + f"\n  WF total x{eq:.2f}, CAGR {cagr * 100:+.0f}%, {n} trades, {bp:+.0f} bp/trade"
