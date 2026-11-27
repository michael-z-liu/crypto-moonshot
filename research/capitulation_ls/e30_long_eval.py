"""E30: evaluate long rankers on crash rows (realistic TP 20% / 72h target): IC, within-day IC,
books at score thresholds, alone and blended with the day5 CNN percentile."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from common_ls import OWN, scores
from sim import Rule, fmt_years, simulate_yearly

Y = (2022, 2023, 2024, 2025, 2026)
d = pl.read_parquet(OWN / "feats.parquet", columns=["row", "open_time", "ret_4h", "qv_base"]).filter((pl.col("ret_4h") <= -0.10) & (pl.col("qv_base") >= 20_000))
d = d.join(pl.read_parquet(OWN / "target_long_tp20.parquet"), on="row").with_columns(year=pl.col("open_time").dt.year(), day=pl.col("open_time").dt.date())
pct = scores("day5")
d = d.with_columns(day5=pl.Series(pct[d["row"].to_numpy()]))
for f in sys.argv[1:]:
    p = pl.read_parquet(OWN / f"{f}.parquet").select("row", pred="pred", pq="pq")
    x = d.join(p, on="row").filter(pl.col("year") >= 2022).drop_nulls("pq")
    x = x.with_columns(blend=(pl.col("pq") + pl.col("day5").fill_nan(0)) / 2)
    for col in ("pq", "day5", "blend"):
        ic = x.group_by("year").agg(ic=pl.corr(pl.col(col).rank(), pl.col("target").rank())).sort("year")["ic"].to_list()
        wd = x.group_by("day").agg(ic=pl.corr(pl.col(col).rank(), pl.col("target").rank()), n=pl.len()).filter(pl.col("n") >= 5)["ic"].drop_nans().mean()
        print(f"{f} [{col}] IC {[round(v, 3) for v in ic]} within-day {wd:+.3f}")
    for col, qs in (("pq", (0.9, 0.95, 0.98)), ("blend", (0.9, 0.95, 0.97)), ("day5", (0.995,))):
        for q in qs:
            c = x.filter(pl.col(col) >= q).select("row", side=pl.lit(1), prio=col)
            b = simulate_yearly(c, Rule(hold_min=4320, tp=0.2, slots=10, cost=0.004), years=Y)
            print(f"   {col}>={q}: {b.summary['bp']:+5.0f}bp n={b.summary['trades']:>3} {fmt_years(b, Y)}")
