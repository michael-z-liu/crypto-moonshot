"""E40: does the CNN tell coins' volatility apart? Within each UTC day, rank-correlate the
day5 forecast's 8h band width (q90 - q10) with the realised |8h return|, and compare with naive
per-coin predictors (trailing 24h volatility from the input sequence, crash depth)."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import CACHE_ALL, fwd2, meta, scores

m = meta()
pct = scores("day5")
path = scores("day5", "path")
yr = m["year"].to_numpy()
idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.9))
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(idx, size=min(len(idx), 120_000), replace=False))
band = np.asarray(path[idx, -1, 2], dtype=np.float64) - np.asarray(path[idx, -1, 0], dtype=np.float64)
med = np.asarray(path[idx, -1, 1], dtype=np.float64)
real = np.asarray(fwd2()[idx, 2, 95], dtype=np.float64)
seq = np.load(CACHE_ALL / "seq2.npy", mmap_mode="r")
closes = np.asarray(seq[idx, 3, :], dtype=np.float32)
trail_vol = np.nanstd(np.diff(closes, axis=1), axis=1)
d = m[idx].select("open_time", "ret_4h").with_columns(
    day=pl.col("open_time").dt.date(), band=pl.Series(band), med=pl.Series(med), absr=pl.Series(np.abs(real)), r=pl.Series(real),
    tvol=pl.Series(trail_vol.astype(np.float64)), depth=-pl.col("ret_4h"), year=pl.col("open_time").dt.year(),
).drop_nans()
g = d.group_by("day").agg(
    n=pl.len(),
    ic_band=pl.corr(pl.col("band").rank(), pl.col("absr").rank()),
    ic_tvol=pl.corr(pl.col("tvol").rank(), pl.col("absr").rank()),
    ic_depth=pl.corr(pl.col("depth").rank(), pl.col("absr").rank()),
    ic_med=pl.corr(pl.col("med").rank(), pl.col("r").rank()),
).filter(pl.col("n") >= 10)
print(f"{g.height} days with >= 10 candidates (pct >= 0.9)")
print(g.select([pl.col(c).drop_nans().mean().round(3).alias(c) for c in ("ic_band", "ic_tvol", "ic_depth", "ic_med")]))
# Does the band add to trailing vol? Partial: residualise band on tvol within day.
d2 = d.with_columns(rb=pl.col("band").rank().over("day"), rt=pl.col("tvol").rank().over("day"), ra=pl.col("absr").rank().over("day"))
x = d2.select("rb", "rt", "ra").to_numpy()
beta = np.linalg.lstsq(np.c_[x[:, 1], np.ones(len(x))], x[:, 0], rcond=None)[0]
res = x[:, 0] - (beta[0] * x[:, 1] + beta[1])
print("partial corr of band (net of trailing vol) with |8h return|:", round(float(np.corrcoef(res, x[:, 2])[0, 1]), 3))
print("calibration: share of realised 8h returns inside the q10-q90 band:",
      round(float(np.mean((real >= np.asarray(path[idx, -1, 0]) / 100) & (real <= np.asarray(path[idx, -1, 2]) / 100))), 3))
d3 = d2.with_columns(blend=(pl.col("rb") + pl.col("rt")) / 2)
g3 = d3.group_by("day").agg(n=pl.len(), ic=pl.corr(pl.col("blend"), pl.col("ra"))).filter(pl.col("n") >= 10)
print("blend of band and trailing-vol ranks, within-day IC with |8h return|:", round(float(g3["ic"].drop_nans().mean()), 3))
for k in (1.1, 1.2, 1.3, 1.4):
    lo, hi_ = np.asarray(path[idx, -1, 0]) / 100, np.asarray(path[idx, -1, 2]) / 100
    mid = np.asarray(path[idx, -1, 1]) / 100
    print(f"band widened x{k}: coverage", round(float(np.mean((real >= mid - k * (mid - lo)) & (real <= mid + k * (hi_ - mid)))), 3))
