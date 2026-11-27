"""E19: rank pump shorts with a target that matches the realistic trade: the perp short return at
the horizon, or -100% if the perp went +100% first (isolated 1x liquidation). Walk-forward GBM
(base and +metric features). Books: iso1x, 10 slots."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from common_ls import OWN
from e08_gbm import FEATS
from e17_squeeze import MF
from sim import Rule, fmt_years, perp_paths, simulate_yearly
from sklearn.ensemble import HistGradientBoostingRegressor

HOLD_H = int(sys.argv[1]) if len(sys.argv) > 1 else 72
d = pl.read_parquet(OWN / "feats.parquet").filter(pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("ret_4h") >= 0.15))
d = d.join(pl.read_parquet(OWN / "metric_feats.parquet"), on="row", how="left").sort("row")
pp, ok = perp_paths(d["row"].to_numpy())
d = d.filter(pl.Series(ok))
k = 96 + HOLD_H - 9 if HOLD_H > 8 else HOLD_H * 12 - 1
adv = np.nan_to_num(pp[:, 0, : k + 1], nan=0).max(1)
s = -np.nan_to_num(pp[:, 2, k], nan=0)
iso = np.where(adv >= 0.99, -1.0, s) - 0.003
d = d.with_columns(iso=pl.Series(iso), year=pl.col("open_time").dt.year())
print(d.height, "rows; mean iso target by year:", d.group_by("year").agg(pl.col("iso").mean() * 1e4).sort("year").to_dict(as_series=False))
Y = (2022, 2023, 2024, 2025, 2026)
for name, feats in (("base", FEATS), ("metrics", FEATS + MF)):
    X = d.select(feats).to_numpy().astype(np.float32)
    yr = d["year"].to_numpy()
    pred, pq = np.full(d.height, np.nan), np.full(d.height, np.nan)
    for y in Y:
        tr, te = yr < y, yr == y
        reg = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.04, min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        reg.fit(X[tr], d["iso"].to_numpy()[tr])
        pt, ptr = reg.predict(X[te]), reg.predict(X[tr])
        pred[te], pq[te] = pt, np.searchsorted(np.sort(ptr), pt) / len(ptr)
    x = d.with_columns(pred=pl.Series(pred), pq=pl.Series(pq)).filter(pl.col("year") >= 2022)
    ic = x.group_by("year").agg(ic=pl.corr(pl.col("pred").rank(), pl.col("iso").rank())).sort("year")["ic"].to_list()
    print(name, "IC by year", [round(v, 3) for v in ic])
    x.select("row", "pred", "pq").write_parquet(OWN / f"e19_{name}_{HOLD_H}h.parquet")
    for q in (0.5, 0.7, 0.8, 0.9):
        c = x.filter(pl.col("pq") >= q).select("row", side=pl.lit(-1), prio="pred")
        for slots in (10, 20):
            b = simulate_yearly(c, Rule(hold_min=HOLD_H * 60, slots=slots, cost=0.003, sl=1.0), perp_funding=True, source="perp", years=Y)
            print(f"  {name} q>={q:.1f} slots {slots}: {b.summary['bp']:+5.0f}bp {fmt_years(b, Y)}")
    for thr in (0.3, 0.4):
        c = x.filter((pl.col("ret_4h") >= thr) & (pl.col("pq") >= 0.5)).select("row", side=pl.lit(-1), prio="pred")
        b = simulate_yearly(c, Rule(hold_min=HOLD_H * 60, slots=10, cost=0.003, sl=1.0), perp_funding=True, source="perp", years=Y)
        print(f"  {name} r4>={thr} & q>=0.5: {b.summary['bp']:+5.0f}bp {fmt_years(b, Y)}")
