"""E36: listing shorts with a trigger: from listing + 72h, short the first time the perp's hourly
open is at or below the listing price (within `window` days), then hold 30 days (TP 50%,
stop 50%, liquidation at +100%, funding)."""

from __future__ import annotations

import numpy as np
import polars as pl
from e13_listing_book import COST, L, perp
from e18_perp_listings import book
from futures_ls import funding_paid

rows = []
for window in (0, 7, 14, 30):
    out = []
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
        i0 = np.searchsorted(h, (t0 // 3600 + 72) * 3600)
        i1 = np.searchsorted(h, h[min(i0, len(h) - 1)] + window * 86400) if window else i0 + 1
        cand = np.flatnonzero(o[i0:i1] <= p_list)
        if i0 >= len(h) or not len(cand):
            continue
        i = i0 + cand[0]
        if i < 1 or qv[i - 1] < 50_000:
            continue
        j_end = min(np.searchsorted(h, h[i] + 30 * 86400), len(h) - 1)
        e = o[i]
        up, dn = hi[i : j_end + 1] / e - 1, lo[i : j_end + 1] / e - 1
        ret, k_exit = -(c[j_end] / e - 1), j_end
        hs, ht = np.flatnonzero(up >= 0.5), np.flatnonzero(-dn >= 0.5)
        ks, kt = (hs[0] if len(hs) else 10**9), (ht[0] if len(ht) else 10**9)
        if ks <= kt and ks < 10**9:
            ret, k_exit = -0.52, i + ks
        elif kt < 10**9:
            ret, k_exit = 0.5, i + kt
        out.append({"symbol": r["symbol"], "t_in": int(h[i]), "t_out": int(h[k_exit]) + 3600, "ret": float(ret),  # noqa: E501
                    "year": r["listed"].year, "qv": float(qv[max(0, i - 24):i].sum()), "wait_d": (h[i] - h[i0]) / 86400})  # fmt: skip  # noqa: E501
    d = pl.DataFrame(out)
    f = funding_paid(
        d["symbol"].to_numpy(), d["t_in"].to_numpy(), d["t_out"].to_numpy(), -np.ones(d.height)
    )
    d = d.with_columns(net=pl.col("ret") - COST - pl.Series(f))
    b = book(d, 10)
    yrs = " | ".join(
        f"{r['year']}: {r['trades']:>3}t {r['bp']:+5.0f}bp {r['ret'] * 100:+5.0f}%"
        for r in b.iter_rows(named=True)
    )
    print(f"window {window:>2}d: n={d.height} {d['net'].mean() * 1e4:+5.0f}bp | {yrs}")
