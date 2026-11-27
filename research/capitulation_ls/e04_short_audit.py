"""E04: audit the pump-short result: who, when, listing age, tails, data glitches."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import meta
from sim import Rule, simulate

m = meta()
r4, qv, yr = m["ret_4h"].to_numpy(), m["qv_base"].to_numpy(), m["year"].to_numpy()
idx = np.flatnonzero((yr >= 2021) & (r4 >= 0.40) & (qv >= 20_000))
S = pl.DataFrame({"row": idx, "side": -1, "prio": r4[idx]})
b = simulate(S, Rule(hold_min=1440, slots=5, cost=0.003))
t = b.trades.join(m.select(pl.int_range(pl.len()).alias("row"), "listed_days", "ret_24h", "btc_ret_4h", "vol_mult"), on="row")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print(t.select("net").describe())
print("by listing age:")
print(t.with_columns(age=pl.col("listed_days").cut([7, 30, 90, 365])).group_by("age").agg(n=pl.len(), bp=pl.col("net").mean() * 1e4, med=pl.col("net").median() * 1e4).sort("age"))
print("by symbol (top contributors):")
print(t.group_by("symbol").agg(n=pl.len(), pnl_bp=pl.col("net").sum() * 1e4).sort("pnl_bp", descending=True).head(12))
print("worst trades:")
print(t.sort("net").select("symbol", "open_time", "listed_days", "ret_4h" if "ret_4h" in t.columns else "prio", "net", "kind").head(10))
print("best trades:")
print(t.sort("net", descending=True).select("symbol", "open_time", "listed_days", "prio", "net").head(10))
print("share of P&L from top 10 trades:", float(t.sort("net", descending=True)["net"].head(10).sum() / t["net"].sum()))
