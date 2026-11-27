"""E31: flip short when a capitulation bounce reaches the take profit (+20%): after the long's TP
fills, short at that price and hold N hours (perp paths, isolated 1x, funding ignored here)."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common_ls import meta, scores, first_per_event
from sim import PERP_STEP_END_MIN, perp_paths

m = meta()
pct = scores("day5")
yr = m["year"].to_numpy()
for q, tp in itertools.product((0.98, 0.995), (0.15, 0.2, 0.3)):
    idx = first_per_event((yr >= 2022) & np.isfinite(pct) & (pct >= q))
    pp, ok = perp_paths(idx)
    idx = idx[ok]
    hi, lo, cl = pp[:, 0], pp[:, 1], pp[:, 2]
    last72 = int(np.searchsorted(PERP_STEP_END_MIN, 4320))
    hit = np.nan_to_num(hi[:, : last72 + 1], nan=0) >= tp
    k = np.where(hit.any(1), hit.argmax(1), -1)
    rows = []
    for i in np.flatnonzero(k >= 0):
        e = 1 + tp
        for h in (24, 72, 120):
            j = min(int(np.searchsorted(PERP_STEP_END_MIN, PERP_STEP_END_MIN[k[i]] + h * 60)), pp.shape[2] - 1)
            up = np.nan_to_num(hi[i, k[i] + 1 : j + 1], nan=0).max(initial=-1) if j > k[i] else -1
            r = -((1 + np.nan_to_num(cl[i, j])) / e - 1)
            r = -1.0 if (1 + up) / e - 1 >= 0.99 else r
            rows.append({"h": h, "ret": r - 0.003, "year": int(yr[idx[i]])})
    d = pl.DataFrame(rows)
    g = d.group_by("h").agg(n=pl.len(), bp=pl.col("ret").mean() * 1e4, med=pl.col("ret").median() * 1e4, win=(pl.col("ret") > 0).mean()).sort("h")
    print(f"q {q} tp {tp}: {int((k >= 0).sum())} of {len(k)} bounces hit TP")
    print(g)
    print(d.filter(pl.col("h") == 72).group_by("year").agg(n=pl.len(), bp=pl.col("ret").mean() * 1e4).sort("year").to_dicts())
