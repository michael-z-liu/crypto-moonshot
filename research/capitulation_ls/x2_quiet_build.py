"""X2 quiet breakouts: build candidates (hourly bars) and path outcomes for a config grid."""

from __future__ import annotations

import os
import sys
from itertools import product
from multiprocessing import Pool

import numpy as np
import polars as pl

os.environ["MOONSHOT_UNIVERSE"] = "all"
sys.path.insert(0, "../volume_spike")
import common

OWN = "/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/explore/"
W1, W0, BRK = 504, 2160, 480  # 21d quiet window, 90d reference, 20d breakout lookback
MAXH = 336
HOLDS = [24, 72, 168, 336]
TPS = [0.2, 0.3, 0.5, None]
STOPS = [None, 0.15, 0.25]
CONFIGS = list(product(HOLDS, TPS, STOPS))
FUND_DAY = 0.0003  # short funding charged per day (assumption)


def per_coin(sym: str):
    d = common.load(sym).filter(pl.col("quote_volume").is_not_null())
    h = (
        d.group_by_dynamic("open_time", every="1h")
        .agg(
            o=pl.col("open").first(),
            h=pl.col("high").max(),
            l=pl.col("low").min(),
            c=pl.col("close").last(),
            qv=pl.col("quote_volume").sum(),
        )
        .sort("open_time")
    )
    n = h.height
    if n < W0 + MAXH + 10:
        return None
    t = h["open_time"].dt.epoch("s").to_numpy()
    o, hi, lo, c, qv = (h[k].to_numpy() for k in ("o", "h", "l", "c", "qv"))
    r = np.zeros(n)
    r[1:] = np.log(c[1:] / c[:-1])
    cs, cs2 = np.concatenate([[0], np.cumsum(r)]), np.concatenate([[0], np.cumsum(r * r)])

    def std(a: int, b: int, idx: np.ndarray) -> np.ndarray:  # rows [idx-a, idx-b)
        lo_, hi_ = idx - a, idx - b
        m = (cs[hi_] - cs[lo_]) / (a - b)
        return np.sqrt(np.maximum((cs2[hi_] - cs2[lo_]) / (a - b) - m * m, 1e-18))

    s = h["qv"]
    med1 = s.rolling_median(W1).shift(1).to_numpy()
    med0 = s.rolling_median(W0 - W1).shift(W1 + 1).to_numpy()
    pmax = h["h"].rolling_max(BRK).shift(1).to_numpy()
    pmin = h["l"].rolling_min(BRK).shift(1).to_numpy()
    idx = np.arange(n)
    ok = (idx >= W0 + 1) & (idx < n - 2)
    surge = qv / np.where(med1 > 0, med1, np.nan)
    up = ok & (c > pmax) & (c > o) & (surge >= 3)
    dn = ok & (c < pmin) & (c < o) & (surge >= 3)
    cand = np.where((up | dn) & (med1 * 24 >= 50_000))[0]
    cand = cand[(t[np.minimum(cand + 1, n - 1)] - t[cand] == 3600)]
    if cand.size == 0:
        return None
    rv = std(W1, 1, cand) / std(W0, W1, cand)
    rq = med1[cand] / med0[cand]
    # cooldown: one candidate per coin per 3 days per side
    keep, last = [], {1: -(10**9), -1: -(10**9)}
    side = np.where(up[cand], 1, -1)
    for k, i in enumerate(cand):
        if i - last[side[k]] >= 72:
            keep.append(k)
            last[side[k]] = i
    cand, rv, rq, side = cand[keep], rv[keep], rq[keep], side[keep]
    mag = np.where(side == 1, c[cand] / pmax[cand] - 1, pmin[cand] / c[cand] - 1)
    rows, outs = [], []
    for k, i in enumerate(cand):
        e = o[i + 1]
        hs, ls, cl = hi[i + 1 : i + 1 + MAXH], lo[i + 1 : i + 1 + MAXH], c[i + 1 : i + 1 + MAXH]
        if len(hs) < MAXH:
            continue
        sd = side[k]
        fav = (hs / e - 1) if sd == 1 else (1 - ls / e)
        adv = (1 - ls / e) if sd == 1 else (hs / e - 1)  # for short adverse = price up
        cmf, cma = np.maximum.accumulate(fav), np.maximum.accumulate(adv)
        res = []
        for H, tp, st in CONFIGS:
            end = H - 1
            ex, kind = end, 0
            ft = np.searchsorted(cmf[: end + 1], tp, side="left") if tp else H
            fs = np.searchsorted(cma[: end + 1], st, side="left") if st else H
            fl = H
            if sd == -1:
                fl = np.searchsorted(cma[: end + 1], 1.0, side="left")
            m = min(ft, fs, fl)
            if m < H:
                ex = m
                if fl == m:
                    kind = 3
                elif fs == m:
                    kind = 2  # stop first when same bar
                else:
                    kind = 1
            if kind == 3:
                g = -1.0
                net = -1.0
            else:
                if kind == 1:
                    g = tp
                elif kind == 2:
                    g = -st
                else:
                    g = (cl[ex] / e - 1) * sd
                days = (ex + 1) / 24
                net = g - (0.004 if sd == 1 else 0.003 + FUND_DAY * days)
                net = max(net, -1.0)
            res.append((net, kind, t[i + 1 + ex] + 3600, cmf[ex], cma[ex]))
        outs.append(res)
        rows.append(
            (sym, sd, t[i + 1], rv[k], rq[k], mag[k], surge[i], qv[i - 23 : i + 1].sum(), i)
        )
    if not rows:
        return None
    return rows, np.array(outs, dtype=np.float64)


if __name__ == "__main__":
    syms = common.universe()
    with Pool(10) as p:
        res = [r for r in p.map(per_coin, syms, chunksize=4) if r]
    rows = [x for r in res for x in r[0]]
    outs = np.concatenate([r[1] for r in res])
    df = pl.DataFrame(
        rows,
        schema=["symbol", "side", "t_in", "rv", "rq", "mag", "surge", "qv24", "i"],
        orient="row",
    )
    df = df.with_columns(year=pl.from_epoch("t_in").dt.year())
    df.write_parquet(OWN + "x2_cand.parquet")
    np.save(OWN + "x2_outs.npy", outs)
    print(df.height, outs.shape)
    print(df.group_by("year", "side").len().sort("year", "side"))
