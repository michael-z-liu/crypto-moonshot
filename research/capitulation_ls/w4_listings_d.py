"""w4 D: run-up before spot listing (for coins whose perp pre-exists), 2025 cadence by month."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN
from e13_listing_book import L, perp

pl.Config.set_tbl_width_chars(200)
pl.Config.set_tbl_rows(60)
D = pl.read_parquet(OWN / "w4_spot_trades.parquet")
rows = []
for r in D.iter_rows(named=True):
    h, o, hi, lo, c, qv = perp[r["symbol"]]
    t0 = int(L.filter(pl.col("symbol") == r["symbol"])["listed"][0].timestamp())
    k0 = np.searchsorted(h, t0)
    rec = {
        "symbol": r["symbol"],
        "perp_age_d": (t0 - h[0]) / 86400,
        "net": r["net"],
        "weak": r["weak"],
        "since": r["since_list"],
        "year": r["year"],
    }
    for lab, back in (("ru24", 24), ("ru72", 72), ("ru168", 168), ("ru720", 720)):
        rec[lab] = (
            o[k0] / o[k0 - back] - 1
            if k0 - back >= 0 and h[k0] - h[k0 - back] == back * 3600
            else np.nan
        )
    # perp volume before listing vs day1 after
    rec["qv_pre24"] = qv[max(0, k0 - 24) : k0].sum() if k0 >= 24 else np.nan
    rec["qv_post24"] = qv[k0 : k0 + 24].sum()
    rows.append(rec)
x = pl.DataFrame(rows)
pre = x.filter(pl.col("perp_age_d") > 3)
print(
    "coins whose perp existed >3d before spot listing: n =",
    pre.height,
    "weak share",
    pre["weak"].mean(),
    "mean net",
    pre["net"].mean(),
)
print(
    "perp age at spot listing (days): quartiles",
    np.percentile(pre["perp_age_d"], [25, 50, 75]).round(1),
)
for col in ("ru24", "ru72", "ru168", "ru720"):
    y = pre.drop_nulls(col)
    print(
        f"{col}: median {y[col].median():+.3f}; corr with since_list {np.corrcoef(y[col], y['since'])[0, 1]:+.2f}; corr with net {np.corrcoef(y[col], y['net'])[0, 1]:+.2f}"  # noqa: E501
    )
    print(
        "   ",
        y.with_columns(b=pl.col(col).qcut(3, allow_duplicates=True))
        .group_by("b")
        .agg(
            n=pl.len(),
            net=pl.col("net").mean(),
            weak=pl.col("weak").mean(),
            since=pl.col("since").median(),
        )
        .sort("b")
        .rows(),
    )
print(
    "\nvolume: day-1-after / 24h-before for pre coins, median",
    np.nanmedian((pre["qv_post24"] / pre["qv_pre24"]).to_numpy()),
)
# by month
m = L.with_columns(ym=pl.col("listed").dt.strftime("%Y-%m")).filter(
    pl.col("listed").dt.year() >= 2024
)
w = D.join(L.select("symbol", "listed"), on="symbol").with_columns(
    ym=pl.col("listed").dt.strftime("%Y-%m")
)
tab = (
    m.group_by("ym")
    .agg(listings=pl.len(), perp_at_list=pl.col("perp_at_list").sum())
    .join(
        w.group_by("ym").agg(traded=pl.len(), weak=pl.col("weak").sum(), net=pl.col("net").mean()),
        on="ym",
        how="left",
    )
    .sort("ym")
)
print(tab)
# 2025 quarters: weak share, median since, qv
q = w.with_columns(
    q=pl.col("listed").dt.year().cast(pl.Utf8) + "Q" + pl.col("listed").dt.quarter().cast(pl.Utf8)
)
print(
    q.group_by("q")
    .agg(
        n=pl.len(),
        weak=pl.col("weak").sum(),
        since=pl.col("since_list").median(),
        net=pl.col("net").mean(),
        net_w=pl.col("net").filter(pl.col("weak")).mean(),
        qv=pl.col("qv_day1").median(),
    )
    .sort("q")
)
# listings per year total spot listings without perp (not tradable)
print(
    L.group_by("year")
    .agg(n=pl.len(), perp=pl.col("perp_at_list").sum(), share=pl.col("perp_at_list").mean())
    .sort("year")
)
# 2021-23 spot listings had no perp: how did they do (spot short gain 3d..)
print(
    L.group_by("year", "perp_at_list")
    .agg(
        n=pl.len(),
        s3d_7d=pl.col("s3d_7d").median(),
        s3d_30d=pl.col("s3d_30d").median(),
        s3d_30m=pl.col("s3d_30d").mean(),
    )
    .sort("year", "perp_at_list")
)
