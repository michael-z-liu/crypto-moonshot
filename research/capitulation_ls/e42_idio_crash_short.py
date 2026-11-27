"""E42: short idiosyncratic crashes (coin down hard while BTC is calm) for continuation, on perp
paths, isolated 1x. Optional: only when the CNN does not expect a bounce (low day5 percentile)."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common_ls import OWN, scores
from sim import Rule, fmt_years, simulate_yearly

Y = (2022, 2023, 2024, 2025, 2026)
d = pl.read_parquet(OWN / "feats.parquet").filter(pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("open_time").dt.year() >= 2022) & (pl.col("ret_4h") <= -0.10))
pct = scores("day5")
d = d.with_columns(day5=pl.Series(pct[d["row"].to_numpy()]))
for r4, calm, cnn, hold in itertools.product((-0.10, -0.15, -0.20), (0.01, 0.02), (None, 0.5), (1440, 4320)):
    f = (pl.col("ret_4h") <= r4) & (pl.col("btc_ret_4h").abs() <= calm)
    if cnn is not None:
        f &= pl.col("day5") <= cnn
    c = d.filter(f).select("row", side=pl.lit(-1), prio=-pl.col("ret_4h"))
    if c.height < 30:
        continue
    b = simulate_yearly(c, Rule(hold_min=hold, slots=10, cost=0.003, sl=1.0), perp_funding=True, source="perp", years=Y)
    print(f"r4<={r4} |btc4|<={calm} cnn<={cnn} hold {hold // 60}h: {b.summary['bp']:+5.0f}bp n={b.summary['trades']:>3} {fmt_years(b, Y)}")
