"""E16: pump shorts on perps with honest tails: cross margin (losses beyond the stake), more and
smaller slots, catastrophe stops far away, and isolated 1x (liquidation at +100%)."""

from __future__ import annotations

import itertools

import polars as pl
from common_ls import OWN
from sim import Rule, fmt_years, simulate_yearly

d = pl.read_parquet(OWN / "feats.parquet").filter(pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("open_time").dt.year() >= 2021))
Y = (2021, 2022, 2023, 2024, 2025, 2026)
for thr in (0.4, 0.5, 0.6):
    c = d.filter(pl.col("ret_4h") >= thr).select("row", side=pl.lit(-1), prio="ret_4h")
    for slots, sl, hold, tp in itertools.product((5, 10, 20), (None, 1.0, 2.0, 3.0), (4320, 10080), (None, 0.5)):
        r = Rule(hold_min=hold, tp=tp, sl=sl, slots=slots, cost=0.003, stop_slip=0.05, loss_floor=-20.0)
        b = simulate_yearly(c, r, perp_funding=True, source="perp")
        yr = b.years
        print(f"thr {thr} slots {slots:>2} sl {str(sl):4s} hold {hold // 60:>3}h tp {str(tp):4s}: {b.summary['bp']:+5.0f}bp worst yr {yr['ret'].min() * 100:+4.0f}% | {fmt_years(b, Y)}")
