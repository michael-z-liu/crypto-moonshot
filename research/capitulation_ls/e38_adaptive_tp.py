"""E38: per-trade take profit from the day5 CNN's own 8h path forecast (q90 peak or q50 peak)."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, fmt_years, simulate_yearly

m = meta()
pct = scores("day5")
path = scores("day5", "path")  # (n, 32 steps, 3 quantiles), percent
yr = m["year"].to_numpy()
Y = (2022, 2023, 2024, 2025, 2026)
idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.995))
pk90 = np.asarray(path[idx, :, 2], dtype=np.float64).max(1) / 100
pk50 = np.asarray(path[idx, :, 1], dtype=np.float64).max(1) / 100
print("q90 peak quantiles", np.round(np.nanpercentile(pk90, [10, 50, 90]), 3), "q50 peak", np.round(np.nanpercentile(pk50, [10, 50, 90]), 3))
for lab, tp in (("fixed 20%", np.full(len(idx), 0.2)),) + tuple((f"{k}x q90 peak", np.clip(k * pk90, 0.08, 0.5)) for k in (1.0, 1.5, 2.0, 3.0)) + tuple((f"{k}x q50 peak", np.clip(k * pk50, 0.08, 0.5)) for k in (2.0, 4.0, 6.0)):
    c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx], "tp": tp})
    b = simulate_yearly(c, Rule(hold_min=4320, slots=10, cost=0.004), years=Y)
    print(f"{lab:14s} {b.summary['bp']:+5.0f}bp n={b.summary['trades']:>3} {fmt_years(b, Y)}")
