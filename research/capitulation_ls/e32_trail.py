"""E32: capitulation longs that let winners run: trailing stop armed at a gain, hold up to 7 days
(perp paths), versus the fixed +20% take profit."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, fmt_years, simulate_yearly

m = meta()
pct = scores("day5")
yr = m["year"].to_numpy()
Y = (2022, 2023, 2024, 2025, 2026)
idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.995))
c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})


def show(lab, r):
    return print(
        f"{lab:44s}",
        (lambda b: f"{b.summary['bp']:+5.0f}bp n={b.summary['trades']:>3} {fmt_years(b, Y)}")(
            simulate_yearly(c, r, years=Y, source="perp")
        ),
    )


show("TP 20% / 72h (reference)", Rule(hold_min=4320, tp=0.2, slots=10, cost=0.003))
show("TP 20% / 168h", Rule(hold_min=10080, tp=0.2, slots=10, cost=0.003))
for arm, trail, hold in itertools.product((0.1, 0.15, 0.2, 0.3), (0.05, 0.1, 0.15), (4320, 10080)):
    show(
        f"arm {arm} trail {trail} hold {hold // 60}h",
        Rule(hold_min=hold, trail=trail, trail_arm=arm, slots=10, cost=0.003),
    )
for arm, trail in itertools.product((0.2,), (0.1,)):
    show(
        f"arm {arm} trail {trail} + TP 50% 168h",
        Rule(hold_min=10080, trail=trail, trail_arm=arm, tp=0.5, slots=10, cost=0.003),
    )
