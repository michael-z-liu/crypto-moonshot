"""E50: the long sleeve (pct >= q, TP 20%, 72h, 10 slots) and the LMC book on day-weighted path
models with other networks (train_arch_day.py) vs the CNN (day5, 5 seeds; day10b, 10 seeds)."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from common_ls import meta, scores
from e25_combined import book, sleeve_crash_short, sleeve_listing_weak, sleeve_long
from sim import Rule, fmt_years, simulate_yearly

m = meta()
yr = m["year"].to_numpy()
Y = (2022, 2023, 2024, 2025, 2026)
for name in ["day5", *sys.argv[1:]]:
    pct = scores(name)
    for q in (0.99, 0.995):
        idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= q))
        c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
        b = simulate_yearly(c, Rule(hold_min=4320, tp=0.2, slots=10, cost=0.004), years=Y)
        print(
            f"{name:16s} q {q}: {b.summary['bp']:+5.0f}bp n={b.summary['trades']:>3} {fmt_years(b, Y)}"  # noqa: E501
        )
    parts = [sleeve_long(score=name), sleeve_listing_weak(), sleeve_crash_short()]
    bk = book(pl.concat(parts, how="vertical_relaxed"), 10)
    n = int(bk["trades"].sum())
    bp = float((bk["bp"] * bk["trades"]).sum() / n)
    print(f"{name:16s} LMC: {bp:+5.0f}bp, years {[round(r * 100) for r in bk['ret'].to_list()]}")
