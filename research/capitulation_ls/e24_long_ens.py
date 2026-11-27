"""E24: ensemble the walk-forward percentiles of several score sets for the capitulation long."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, fmt_years, simulate_yearly

m = meta()
yr = m["year"].to_numpy()
Y = (2022, 2023, 2024, 2025, 2026)
sets = {
    "day-family": ["day", "day5", "volday5"],
    "all13": [
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
    ],
    "diverse4": ["day5", "volscale5", "path_all5", "foldnorm"],
}
for name, members in sets.items():
    P = np.stack([scores(s) for s in members])
    ens = np.nanmean(P, 0)  # mean percentile
    mn = np.nanmin(P, 0)  # all members agree
    for lab, sc in (("mean", ens), ("min", mn)):
        for q, tp in itertools.product((0.99, 0.995, 0.997), (0.15, 0.2, 0.3)):
            idx = np.flatnonzero((yr >= 2022) & np.isfinite(sc) & (sc >= q))
            c = pl.DataFrame({"row": idx, "side": 1, "prio": sc[idx]})
            b = simulate_yearly(c, Rule(hold_min=4320, tp=tp, slots=10, cost=0.004), years=Y)
            print(
                f"{name:10s} {lab:4s} q {q} tp {tp}: {b.summary['bp']:+5.0f}bp n={b.summary['trades']:>3} {fmt_years(b, Y)}"  # noqa: E501
            )
