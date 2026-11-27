"""E05: pump-short grid: pump size, hold, take profit, stop (no funding yet, spot paths)."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common_ls import meta
from sim import Rule, simulate

m = meta()
r4, qv, yr = m["ret_4h"].to_numpy(), m["qv_base"].to_numpy(), m["year"].to_numpy()
rows = []
for thr, liq in itertools.product((0.2, 0.3, 0.4, 0.6), (20_000,)):
    idx = np.flatnonzero((yr >= 2021) & (r4 >= thr) & (qv >= liq))
    S = pl.DataFrame({"row": idx, "side": -1, "prio": r4[idx]})
    for hold, tp, sl in itertools.product((240, 1440, 4320), (None, 0.2, 0.3), (None, 0.2, 0.4)):
        b = simulate(S, Rule(hold_min=hold, tp=tp, sl=sl, slots=5, cost=0.003, stop_slip=0.01))
        y = {r["year"]: r for r in b.years.iter_rows(named=True)}
        rows.append({"thr": thr, "hold": hold, "tp": tp, "sl": sl, "n": b.summary["trades"],
                     "bp": round(b.summary["bp"]), **{f"bp{k}": round(y[k]["bp"]) for k in range(2021, 2027) if k in y},
                     "ret25": round(y[2025]["ret"] * 100), "ret26": round(y[2026]["ret"] * 100),
                     "worst_yr_ret": round(min(r["ret"] for r in y.values()) * 100),
                     "maxdd": round(b.summary["maxdd"] * 100)})  # fmt: skip
out = pl.DataFrame(rows).sort("bp", descending=True)
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(30)
print(out.head(40))
out.write_csv("results/e05_short_grid.csv")
