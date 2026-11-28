"""E03: simulator sanity check against the #10 study, then longer holds for longs and shorts."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, fmt_years, simulate

m = meta()
pct = scores("day5")
r4 = m["ret_4h"].to_numpy()
qv = m["qv_base"].to_numpy()
yr = m["year"].to_numpy()


def cands(mask: np.ndarray, side: int, prio: np.ndarray) -> pl.DataFrame:
    idx = np.flatnonzero(mask)
    return pl.DataFrame({"row": idx, "side": np.full(len(idx), side), "prio": prio[idx]})


wf = (yr >= 2022) & np.isfinite(pct)
L = cands(wf & (pct >= 0.98), 1, pct)
for hold in (60, 120, 240, 480, 1440, 4320):
    b = simulate(L, Rule(hold_min=hold, slots=5))
    print(f"long day5>=.98 hold {hold:>4}m  {fmt_years(b, (2022, 2023, 2024, 2025, 2026))}")
L2 = cands(wf & (pct >= 0.995), 1, pct)
for hold in (120, 1440, 4320):
    b = simulate(L2, Rule(hold_min=hold, slots=5))
    print(f"long day5>=.995 hold {hold:>4}m {fmt_years(b, (2022, 2023, 2024, 2025, 2026))}")
S = cands((yr >= 2021) & (r4 >= 0.40) & (qv >= 20_000), -1, r4)
for hold in (240, 1440, 4320):
    b = simulate(S, Rule(hold_min=hold, slots=5, cost=0.003))
    print(f"short pump40 liq hold {hold:>4}m {fmt_years(b)}")
