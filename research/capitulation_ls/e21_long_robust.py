"""E21: robustness of the capitulation long (day5 pct >= 0.995, TP, hold <= 72h): neighbours,
costs, one candle late, perp paths, leverage with liquidation, and a stop."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, fmt_years, perp_paths, simulate_yearly

m = meta()
pct = scores("day5")
yr = m["year"].to_numpy()
Y = (2022, 2023, 2024, 2025, 2026)


def cands(q: float) -> pl.DataFrame:
    idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= q))
    return pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})


def show(label: str, c: pl.DataFrame, rule: Rule, src: str = "spot") -> None:
    b = simulate_yearly(c, rule, years=Y, source=src)
    print(f"{label:48s} {b.summary['bp']:+5.0f}bp n={b.summary['trades']:>3} {fmt_years(b, Y)}")


c995 = cands(0.995)
_, ok = perp_paths(c995["row"].to_numpy())
print("perp path coverage of q>=0.995 candidates:", round(float(ok.mean()) * 100), "%")
print("-- neighbours (spot, 0.4%)")
for q, tp, hold, slots in itertools.product((0.99, 0.995, 0.997), (0.1, 0.15, 0.2, 0.25, 0.3), (1440, 2880, 4320), (5, 10)):
    if (tp in (0.1, 0.3) or hold == 2880) and slots == 5:
        continue
    show(f"q {q} tp {tp} hold {hold // 60}h slots {slots}", cands(q), Rule(hold_min=hold, tp=tp, slots=slots, cost=0.004))
base = Rule(hold_min=4320, tp=0.2, slots=10, cost=0.004)
print("-- stresses on q 0.995 tp 0.2 72h 10 slots")
show("base", c995, base)
show("cost 0.8%", c995, Rule(hold_min=4320, tp=0.2, slots=10, cost=0.008))
show("one candle late", c995, Rule(hold_min=4320, tp=0.2, slots=10, cost=0.004, delay=1))
show("three candles late", c995, Rule(hold_min=4320, tp=0.2, slots=10, cost=0.004, delay=3))
show("perp paths, 0.3%", c995, Rule(hold_min=4320, tp=0.2, slots=10, cost=0.003), "perp")
print("-- leverage on perp-cost spot paths (liquidation at 1/lev - 1%)")
for lev, slots, sl in itertools.product((1, 2, 3), (5, 10), (None, 0.25)):
    show(f"lev {lev} slots {slots} sl {sl}", c995, Rule(hold_min=4320, tp=0.2, sl=sl, slots=slots, cost=0.003, lev=lev, stop_slip=0.01))
