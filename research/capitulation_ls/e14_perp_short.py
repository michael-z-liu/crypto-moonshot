"""E14: pump shorts on perp paths (5m to 8h, hourly to 7 days), spot paths for comparison."""

from __future__ import annotations

import itertools

import polars as pl
from common_ls import OWN
from sim import Rule, fmt_years, simulate_yearly

d = pl.read_parquet(OWN / "feats.parquet").filter(
    pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("open_time").dt.year() >= 2021)
)
Y = (2021, 2022, 2023, 2024, 2025, 2026)
for thr in (0.3, 0.4, 0.6):
    c = d.filter(pl.col("ret_4h") >= thr).select("row", side=pl.lit(-1), prio="ret_4h")
    for src in ("spot", "perp"):
        b = simulate_yearly(
            c, Rule(hold_min=4320, slots=5, cost=0.003), perp_funding=True, source=src
        )
        print(f"thr {thr} 72h {src:4s}: {b.summary['bp']:+5.0f}bp {fmt_years(b, Y)}")
    for hold, tp, sl in itertools.product(
        (1440, 4320, 7200, 10080), (None, 0.3, 0.5), (None, 0.5, 1.0)
    ):
        sl_v = None if sl is None else sl
        b = simulate_yearly(
            c,
            Rule(hold_min=hold, tp=tp, sl=sl_v, slots=5, cost=0.003, stop_slip=0.01),
            perp_funding=True,
            source="perp",
        )
        print(
            f"thr {thr} perp hold {hold // 60:>3}h tp {tp} sl {sl}: {b.summary['bp']:+5.0f}bp {fmt_years(b, Y)}"  # noqa: E501
        )
