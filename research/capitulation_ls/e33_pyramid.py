"""E33: allow a second position in a coin already held (a later, deeper capitulation spike)."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, fmt_years, simulate_yearly

m = meta()
pct = scores("day5")
yr = m["year"].to_numpy()
Y = (2022, 2023, 2024, 2025, 2026)
for q in (0.99, 0.995):
    idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= q))
    c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
    for mpc, slots in ((1, 10), (2, 10), (2, 15), (3, 15)):
        b = simulate_yearly(
            c, Rule(hold_min=4320, tp=0.2, slots=slots, cost=0.004, max_per_coin=mpc), years=Y
        )
        print(
            f"q {q} per-coin {mpc} slots {slots}: {b.summary['bp']:+5.0f}bp n={b.summary['trades']:>3} {fmt_years(b, Y)}"  # noqa: E501
        )
