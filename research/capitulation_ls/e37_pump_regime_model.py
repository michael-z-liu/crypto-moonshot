"""E37: regime-filtered pump shorts (BTC 7d > -3%) with the realistic-target GBM ranks (E19)."""

from __future__ import annotations

import polars as pl
from common_ls import OWN
from e34_regime import btc_trend
from sim import Rule, fmt_years, simulate_yearly

Y = (2022, 2023, 2024, 2025, 2026)
d = pl.read_parquet(OWN / "feats.parquet").filter(pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("open_time").dt.year() >= 2022))
d = d.with_columns(b7=pl.Series(btc_trend(d["t"].to_numpy(), 7)))
for f in ("e19_metrics_72h", "e19_base_72h"):
    x = d.join(pl.read_parquet(OWN / f"{f}.parquet"), on="row", how="inner")
    for lab, filt in (("r4>=0.4", pl.col("ret_4h") >= 0.4), ("r4>=0.4 & q>=0.5", (pl.col("ret_4h") >= 0.4) & (pl.col("pq") >= 0.5)),
                      ("r4>=0.3 & q>=0.7", (pl.col("ret_4h") >= 0.3) & (pl.col("pq") >= 0.7)), ("q>=0.9", pl.col("pq") >= 0.9)):
        c = x.filter(filt & (pl.col("b7") > -0.03)).select("row", side=pl.lit(-1), prio="pred")
        b = simulate_yearly(c, Rule(hold_min=4320, slots=10, cost=0.003, sl=1.0), perp_funding=True, source="perp", years=Y)
        print(f"{f:16s} {lab:18s} b7>-3%: {b.summary['bp']:+5.0f}bp n={b.summary['trades']:>3} {fmt_years(b, Y)}")
