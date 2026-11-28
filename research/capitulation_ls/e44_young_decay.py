"""E44: weekly 'young coin decay' shorts: every Monday 00:00 UTC, short each spot-listed coin aged
7-180 days that trades below its listing price (perp liquid), hold 7 days (liquidation at +100%,
funding, 0.3% cost). Per-trade stats by year and by age bucket."""

from __future__ import annotations

import numpy as np
import polars as pl
from e13_listing_book import COST, L, perp
from futures_ls import funding_paid

rows = []
mondays = pl.datetime_range(
    pl.datetime(2021, 1, 4, time_zone="UTC"),
    pl.datetime(2026, 10, 19, time_zone="UTC"),
    "7d",
    eager=True,
)
for r in L.iter_rows(named=True):
    a = perp.get(r["symbol"])
    if a is None:
        continue
    h, o, hi, lo, c, qv = a
    t0 = int(r["listed"].timestamp())
    k0 = np.searchsorted(h, t0)
    if k0 >= len(h):
        continue
    p_list = o[k0]
    for mon in mondays:
        tm = int(mon.timestamp())
        age = (tm - t0) / 86400
        if age < 7 or age > 180:
            continue
        i = np.searchsorted(h, tm)
        if i >= len(h) - 1 or i < 1 or h[i] != tm or qv[i - 1] < 50_000:
            continue
        rel = o[i] / p_list - 1
        j = min(np.searchsorted(h, tm + 7 * 86400), len(h) - 1)
        up = hi[i : j + 1].max() / o[i] - 1
        ret = -1.0 if up >= 1.0 else -(c[j] / o[i] - 1)
        rows.append(
            {
                "symbol": r["symbol"],
                "t_in": tm,
                "t_out": int(h[j]) + 3600,
                "ret": ret,
                "age": age,
                "rel": rel,
                "year": mon.year,
            }
        )
d = pl.DataFrame(rows)
f = funding_paid(
    d["symbol"].to_numpy(), d["t_in"].to_numpy(), d["t_out"].to_numpy(), -np.ones(d.height)
)
d = d.with_columns(net=pl.col("ret") - COST - pl.Series(f))
for lab, flt in (
    ("all young", pl.lit(True)),
    ("below listing", pl.col("rel") <= 0),
    ("below -30%", pl.col("rel") <= -0.3),
    ("above listing", pl.col("rel") > 0),
):
    x = d.filter(flt)
    print(
        f"{lab:14s} n={x.height:>5} {x['net'].mean() * 1e4:+5.0f}bp med {x['net'].median() * 1e4:+5.0f} |",  # noqa: E501
        x.group_by("year")
        .agg(n=pl.len(), bp=(pl.col("net").mean() * 1e4).round())
        .sort("year")
        .to_dicts(),
    )
x = d.filter(pl.col("rel") <= 0)
print(
    x.with_columns(b=pl.col("age").cut([30, 60, 120]))
    .group_by("b")
    .agg(n=pl.len(), bp=(pl.col("net").mean() * 1e4).round())
    .sort("b")
    .to_dicts()
)
d.write_parquet("/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/e44_young.parquet")
