"""E20: capitulation longs with longer holds: day5 CNN percentile thresholds, holds 1h-72h, slots,
take profits, a BTC-flush gate; spot paths (0.4%) and perp-cost (0.3%) variants."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, fmt_years, simulate_yearly

m = meta()
pct = scores("day5")
yr = m["year"].to_numpy()
btc4 = m["btc_ret_4h"].to_numpy()
Y = (2022, 2023, 2024, 2025, 2026)
rows = []
for q, gate, hold, slots, tp in itertools.product((0.98, 0.99, 0.995, 0.998), (None, -0.03), (120, 480, 1440, 4320), (3, 5, 10), (None, 0.2)):
    mask = (yr >= 2022) & np.isfinite(pct) & (pct >= q)
    if gate is not None:
        mask &= btc4 <= gate
    idx = np.flatnonzero(mask)
    c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
    b = simulate_yearly(c, Rule(hold_min=hold, slots=slots, tp=tp, cost=0.004), years=Y)
    y = {r["year"]: r for r in b.years.iter_rows(named=True)}
    rows.append({"q": q, "gate": gate, "hold": hold, "slots": slots, "tp": tp, "n": b.summary["trades"], "bp": round(b.summary["bp"]),
                 "min_ret": round(min(r["ret"] for r in y.values()) * 100), "med_ret": round(float(np.median([r["ret"] for r in y.values()])) * 100),
                 "dd": round(b.summary["maxdd"] * 100), **{f"r{k}": round(v["ret"] * 100) for k, v in y.items()}, **{f"bp{k}": round(v["bp"]) for k, v in y.items()}})  # fmt: skip
out = pl.DataFrame(rows)
out.write_csv("results/e20_long.csv")
pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_rows(25)
print(out.sort("min_ret", descending=True).head(20))
print(out.filter(pl.col("bp") >= 1000).sort("min_ret", descending=True).head(10))
