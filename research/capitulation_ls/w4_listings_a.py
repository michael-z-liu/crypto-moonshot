"""w4 A: spot-listing vs perp-only listings; weak/strong; outcome decomposition; cadence by year."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN
from e13_listing_book import L, trades
from e18_perp_listings import perp as perp18
from e18_perp_listings import trades as trades18
from futures_ls import spot_name

pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
X = pl.read_parquet(OWN / "e27_listings.parquet")
spot_syms = set(L["symbol"].to_list())

# ---- spot-listing trades (72h, 30d, 50/50)
d = trades(72, 30, 0.5, 0.5).join(
    X.select("symbol", "since_list", "off_high", "perp_first_h", "qv_day1", "fund_right", "qv_3d"),
    on="symbol",
)
d = d.with_columns(weak=pl.col("since_list") <= 0)
print("spot-listing trades", d.height, "mean net", d["net"].mean(), "weak n", d["weak"].sum())
exitk = d.with_columns(
    kind=pl.when(pl.col("ret") >= 0.5)
    .then(pl.lit("TP"))
    .when(pl.col("ret") <= -0.5)
    .then(pl.lit("SL"))
    .otherwise(pl.lit("time"))
)
print(
    exitk.group_by("weak", "kind")
    .agg(n=pl.len(), net=pl.col("net").mean(), fund=pl.col("fund").mean())
    .sort("weak", "kind")
)
print("by weak:")
print(
    d.group_by("weak").agg(
        n=pl.len(),
        net=pl.col("net").mean(),
        med=pl.col("net").median(),
        since=pl.col("since_list").median(),
        offhigh=pl.col("off_high").median(),
        qv1=pl.col("qv_day1").median(),
        qv3=pl.col("qv_3d").median(),
        fund=pl.col("fund_right").mean(),
        pf=pl.col("perp_first_h").median(),
        adv=pl.col("adv").mean(),
    )
)

# continuous bins of since_list
print(
    d.with_columns(b=pl.col("since_list").cut([-0.5, -0.3, -0.15, 0, 0.15, 0.4, 1.0]))
    .group_by("b")
    .agg(
        n=pl.len(),
        net=pl.col("net").mean(),
        med=pl.col("net").median(),
        win=(pl.col("net") > 0).mean(),
    )
    .sort("b")
)

# ---- perp-only vs spot-listed in e18 universe
e = trades18(72, 30, 0.5, 0.5)
e = e.with_columns(sp=pl.col("symbol").map_elements(spot_name, return_dtype=pl.Utf8))
e = e.with_columns(spot_listed=pl.col("sp").is_in(list(spot_syms)))
since = []
for r in e.iter_rows(named=True):
    h, o = perp18[r["symbol"]][0], perp18[r["symbol"]][1]
    i = np.searchsorted(h, r["t_in"])
    since.append(o[i] / o[0] - 1)
e = e.with_columns(since=pl.Series(since)).with_columns(weak=pl.col("since") <= 0)
print("\nE18 universe by spot_listed x weak (perp-first-candle clock)")
print(
    e.group_by("spot_listed", "weak")
    .agg(
        n=pl.len(),
        net=pl.col("net").mean(),
        med=pl.col("net").median(),
        win=(pl.col("net") > 0).mean(),
        qv24=pl.col("qv").median(),
        fund=pl.col("fund").mean(),
    )
    .sort("spot_listed", "weak")
)
print(
    e.group_by("spot_listed", "year")
    .agg(n=pl.len(), net=pl.col("net").mean(), weakshare=pl.col("weak").mean())
    .sort("spot_listed", "year")
)

# day-1 perp quote volume per group (perp 24h after first candle) and first 72h volatility/return
rows = []
for sym, a in perp18.items():
    h, o, hi, lo, c, qv = a
    if h[0] < 1609459200 or len(h) < 800:
        continue
    sp = spot_name(sym)
    r72 = o[72] / o[0] - 1
    rows.append({"sym": sp, "spot": sp in spot_syms, "year": int(np.datetime64(int(h[0]), "s").astype("datetime64[Y]").astype(int) + 1970),  # noqa: E501
                 "qv1": qv[:24].sum(), "qv3": qv[:72].sum(), "r72": r72, "hi72": hi[:72].max() / o[0] - 1,  # noqa: E501
                 "lo72": lo[:72].min() / o[0] - 1, "r30": c[72 + 720] / o[72] - 1 if len(c) > 800 else np.nan,  # noqa: E501
                 "vol72": float(np.std(np.diff(np.log(c[:72]))))})  # fmt: skip
g = pl.DataFrame(rows)
print("\nperp-first-candle clock, all perps listed 2021+ with >=800h:")
print(
    g.group_by("spot").agg(
        n=pl.len(),
        qv1=pl.col("qv1").median(),
        qv3=pl.col("qv3").median(),
        r72=pl.col("r72").median(),
        r72m=pl.col("r72").mean(),
        hi72=pl.col("hi72").median(),
        lo72=pl.col("lo72").median(),
        vol72=pl.col("vol72").median(),
        r30=pl.col("r30").median(),
        r30m=pl.col("r30").mean(),
        pdown=(pl.col("r72") <= 0).mean(),
    )
)
print(
    g.group_by("year", "spot")
    .agg(
        n=pl.len(),
        r72=pl.col("r72").median(),
        r30=pl.col("r30").median(),
        qv1=pl.col("qv1").median(),
    )
    .sort("year", "spot")
)

# spot listings: perp existed before spot?
print("\nspot listings: perp age at spot listing (h), share perp existed at spot listing:")
print(X.group_by("year" if "year" in X.columns else pl.lit(0)).agg(n=pl.len()) if False else "")
x2 = d.with_columns(pre=pl.col("perp_first_h") > 1)
print(x2.group_by("pre").agg(n=pl.len(), net=pl.col("net").mean(), weak=pl.col("weak").mean()))

# ---- cadence by year
Ly = L.with_columns(has_trade=pl.col("symbol").is_in(d["symbol"].to_list()))
print("\nlistings per year (L) / with qualifying perp trade / weak share")
print(
    Ly.group_by("year")
    .agg(
        n=pl.len(),
        perp_list=pl.col("perp_at_list").sum(),
        traded=pl.col("has_trade").sum(),
        qv1=pl.col("qv_1d").median(),
        s3d_7d=pl.col("s3d_7d").median(),
    )
    .sort("year")
    .join(
        d.group_by("year").agg(
            weak=pl.col("weak").sum(),
            weak_share=pl.col("weak").mean(),
            net_w=pl.col("net").filter(pl.col("weak")).mean(),
            net_s=pl.col("net").filter(~pl.col("weak")).mean(),
            n_s=(~pl.col("weak")).sum(),
        ),
        on="year",
    )
)
d.write_parquet(OWN / "w4_spot_trades.parquet")
