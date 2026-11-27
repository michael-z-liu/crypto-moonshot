"""E22: the capitulation long rule (pct >= q, TP 20%, hold 72h, 10 slots) on every walk-forward
score set available, to see whether it depends on one model."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, fmt_years, simulate_yearly

m = meta()
yr = m["year"].to_numpy()
Y = (2022, 2023, 2024, 2025, 2026)
for name in (
    "path_all",
    "path_all5",
    "base3",
    "day",
    "day5",
    "volday5",
    "volscale5",
    "foldnorm",
    "recency",
    "roll3y",
    "tail",
    "es_top",
    "fixed",
):
    try:
        pct = scores(name)
    except (FileNotFoundError, KeyError):
        continue
    for q in (0.99, 0.995):
        idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= q))
        c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
        b = simulate_yearly(c, Rule(hold_min=4320, tp=0.2, slots=10, cost=0.004), years=Y)
        print(
            f"{name:10s} q {q}: {b.summary['bp']:+5.0f}bp n={b.summary['trades']:>3} {fmt_years(b, Y)}"  # noqa: E501
        )
