"""E49: spot listings whose perp arrives later (no liquid perp at listing + 72h): short once the
perp has traded a day (perp listing + 24h) if the price is at or below the spot listing price,
within 60 days of the spot listing; hold 30 days, TP 50%, stop 50%, funding."""

from __future__ import annotations

import numpy as np
import polars as pl
from e13_listing_book import COST, L, perp
from futures_ls import funding_paid, spot_name

by_spot: dict[str, str] = {}
for p in perp:
    by_spot.setdefault(spot_name(p), p)
rows = []
for r in L.iter_rows(named=True):
    p = by_spot.get(r["symbol"])
    if p is None:
        continue
    h, o, hi, lo, c, qv = perp[p]
    t0 = int(r["listed"].timestamp())
    i72 = np.searchsorted(h, (t0 // 3600 + 72) * 3600)
    if i72 < len(h) and h[i72] - t0 < 80 * 3600 and qv[max(i72 - 1, 0)] >= 50_000:
        continue  # had a liquid perp at +72h: already in sleeve M
    i = np.searchsorted(h, h[0] + 24 * 3600)
    if i >= len(h) - 1 or h[i] - t0 > 60 * 86400 or h[i] < t0 or qv[i - 1] < 50_000:
        continue
    # spot listing price: first spot open from the store
    import os, sys  # noqa: E401

    os.environ["MOONSHOT_UNIVERSE"] = "all"
    from common_ls import CACHE_ALL

    sys.path.insert(0, str(CACHE_ALL.parent))
    from common import load

    p_list = float(load(r["symbol"])["open"][0])
    if o[i] > p_list:
        continue
    j_end = min(np.searchsorted(h, h[i] + 30 * 86400), len(h) - 1)
    up, dn = hi[i : j_end + 1] / o[i] - 1, lo[i : j_end + 1] / o[i] - 1
    hs, ht = np.flatnonzero(up >= 0.5), np.flatnonzero(-dn >= 0.5)
    ks, kt = (hs[0] if len(hs) else 10**9), (ht[0] if len(ht) else 10**9)
    ret, k = (-0.52, i + ks) if ks <= kt and ks < 10**9 else ((0.5, i + kt) if kt < 10**9 else (-(c[j_end] / o[i] - 1), j_end))
    rows.append({"symbol": r["symbol"], "t_in": int(h[i]), "t_out": int(h[k]) + 3600, "ret": ret, "year": r["listed"].year})
d = pl.DataFrame(rows)
if d.height:
    f = funding_paid(d["symbol"].to_numpy(), d["t_in"].to_numpy(), d["t_out"].to_numpy(), -np.ones(d.height))
    d = d.with_columns(net=pl.col("ret") - COST - pl.Series(f))
    print(d.height, "late-perp weak listings;", round(d["net"].mean() * 1e4), "bp; by year", d.group_by("year").agg(n=pl.len(), bp=(pl.col("net").mean() * 1e4).round()).sort("year").to_dicts())
else:
    print("none")
