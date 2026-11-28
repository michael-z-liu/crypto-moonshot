"""E23: walk-forward selection of the capitulation-long rule (threshold, take profit, hold, slots)
for each score set: each test year uses the configuration with the best record on earlier years
(from 2022; 2023's pick sees 2022 only). Selection score as in wf.py."""

from __future__ import annotations

import itertools
import sys
from typing import Any

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, simulate_yearly
from wf import summarize, table

NAME = sys.argv[1] if len(sys.argv) > 1 else "day5"
Y = (2022, 2023, 2024, 2025, 2026)


def run(cfg: dict[str, Any]) -> pl.DataFrame:
    m = meta()
    yr = m["year"].to_numpy()
    pct = scores(NAME)
    idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= cfg["q"]))
    c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
    return simulate_yearly(
        c, Rule(hold_min=cfg["hold"], tp=cfg["tp"], slots=cfg["slots"], cost=0.004), years=Y
    ).years


def walk(tab: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for y in (2023, 2024, 2025, 2026):
        past = tab.filter(pl.col("year") < y)
        g = past.group_by("cfg").agg(
            score=pl.col("ret").clip(-0.99, None).log1p().mean() - 0.5 * pl.col("dd").min().abs(),
            n=pl.col("trades").sum(),
            k=pl.len(),
        )
        c = int(g.filter(pl.col("k") == y - 2022).sort(["score", "n"], descending=True)["cfg"][0])
        rows.append(tab.filter((pl.col("cfg") == c) & (pl.col("year") == y)).row(0, named=True))
    return pl.DataFrame(rows)


if __name__ == "__main__":
    grid = [dict(q=q, tp=tp, hold=h, slots=s) for q, tp, h, s in itertools.product(
        (0.98, 0.99, 0.995, 0.997, 0.998), (None, 0.1, 0.15, 0.2, 0.3), (120, 1440, 2880, 4320), (5, 10))]  # fmt: skip  # noqa: E501
    tab = table(run, grid)
    wf = walk(tab)
    print(NAME, "walk-forward 2023-26:")
    print(summarize(wf))
    g = tab.group_by("cfg").agg(
        minret=pl.col("ret").min(),
        bp=(pl.col("bp") * pl.col("trades")).sum() / pl.col("trades").sum(),
    )
    print(
        "configs:",
        g.height,
        "positive every year 2022-26:",
        int((g["minret"] > 0).sum()),
        "; with >= +10%/trade:",
        int((g["bp"] >= 1000).sum()),
    )
