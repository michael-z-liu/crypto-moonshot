"""E27: which spot listings fall most? Outcome = realistic listing short (entry at listing + 72h,
hold 30d, TP 50%, stop 50%, perp prices, funding). Features known at entry."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN
from e13_listing_book import L, perp, trades
from futures_ls import funding_cum

d = trades(72, 30, 0.5, 0.5)
feats = []
fc = funding_cum()
for r in d.iter_rows(named=True):
    h, o, hi, lo, c, qv = perp[r["symbol"]]
    lst = L.filter(pl.col("symbol") == r["symbol"]).row(0, named=True)
    t0 = int(lst["listed"].timestamp())
    i = np.searchsorted(h, r["t_in"])
    k0 = np.searchsorted(h, t0)
    first_perp = int(h[0])
    seg_hi = hi[k0:i].max() if i > k0 else np.nan
    since = o[i] / o[k0] - 1 if i > k0 else np.nan
    f = fc.get(r["symbol"])
    fr = np.nan
    if f is not None:
        j = np.searchsorted(f[0], r["t_in"]) - 1
        if j >= 3:
            fr = (f[1][j] - f[1][j - 3]) / 3
    feats.append({"symbol": r["symbol"], "perp_first_h": (t0 - first_perp) / 3600, "since_list": since,
                  "off_high": o[i] / seg_hi - 1 if seg_hi == seg_hi else np.nan, "qv_3d": float(qv[k0:i].sum()),
                  "fund": fr, "qv_day1": float(lst.get("qv_1d") or np.nan)})  # fmt: skip
x = d.join(pl.DataFrame(feats), on="symbol")
pl.Config.set_tbl_width_chars(200)
for col in ("perp_first_h", "since_list", "off_high", "qv_3d", "fund", "qv_day1"):
    g = x.with_columns(b=pl.col(col).qcut(4, allow_duplicates=True)).group_by("b").agg(
        n=pl.len(), bp=pl.col("net").mean() * 1e4, med=pl.col("net").median() * 1e4, win=(pl.col("net") > 0).mean()).sort("b")
    print(col); print(g)
x.write_parquet(OWN / "e27_listings.parquet")
