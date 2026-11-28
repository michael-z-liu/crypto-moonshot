"""E10: short entry timing: enter 0-4h after the spike, optionally only once the price turned."""

from __future__ import annotations

import polars as pl
from common_ls import OWN
from sim import Rule, fmt_years, simulate_yearly

d = pl.read_parquet(OWN / "feats.parquet").filter(
    pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("open_time").dt.year() >= 2022)
)
g = pl.read_parquet(OWN / "e08_short_72h.parquet")
d = d.join(g, on="row", how="left")
sets = {
    "rule r4>=0.4": d.filter(pl.col("ret_4h") >= 0.4).select("row", side=pl.lit(-1), prio="ret_4h"),
    "gbm q>=0.8": d.filter((pl.col("pq") >= 0.8) & (pl.col("ret_4h") >= 0.15)).select(
        "row", side=pl.lit(-1), prio="pred"
    ),
}
for name, c in sets.items():
    for delay, cond in (
        (0, None),
        (1, None),
        (3, None),
        (6, None),
        (12, None),
        (24, None),
        (48, None),
        (3, "confirm"),
        (12, "confirm"),
        (24, "confirm"),
    ):
        b = simulate_yearly(
            c,
            Rule(hold_min=4320, slots=5, cost=0.003, delay=delay, delay_cond=cond),
            perp_funding=True,
        )
        print(
            f"{name:13s} delay {delay * 5:>3}m {cond!s:8s} {b.summary['bp']:+5.0f}bp {fmt_years(b, (2022, 2023, 2024, 2025, 2026))}"  # noqa: E501
        )
