"""E26: a dedicated ranker for capitulation longs, trained on the realistic outcome (take profit
+20%, exit 72h, spot paths, 0.4% cost) of crash events (r4 <= -10%, liquid). Inputs: the tabular
features plus the walk-forward day5 CNN percentile (stacking; each row's day5 score is out of
sample). Walk-forward GBM; books compared with the plain day5 rule."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN, meta, scores
from e08_gbm import FEATS
from sim import Rule, exits, fmt_years, paths, simulate_yearly
from sklearn.ensemble import HistGradientBoostingRegressor

Y = (2022, 2023, 2024, 2025, 2026)
d = pl.read_parquet(OWN / "feats.parquet").filter((pl.col("ret_4h") <= -0.10) & (pl.col("qv_base") >= 20_000)).sort("row")
rows = d["row"].to_numpy()
p = paths(rows)
ret, _, _ = exits(p, np.ones(len(rows)), Rule(hold_min=4320, tp=0.2))
pct = scores("day5")
d = d.with_columns(target=pl.Series(ret - 0.004), pct_day5=pl.Series(pct[rows]), year=pl.col("open_time").dt.year())
d.select("row", "target").write_parquet(OWN / "target_long_tp20.parquet")
print(d.height, "crash rows; mean target by year:", d.group_by("year").agg((pl.col("target").mean() * 1e4).round()).sort("year").to_dict(as_series=False))

out = {}
for name, feats in (("gbm", FEATS), ("gbm+cnn", FEATS + ["pct_day5"])):
    X = d.select(feats).to_numpy().astype(np.float32)
    yr = d["year"].to_numpy()
    pred, pq = np.full(d.height, np.nan), np.full(d.height, np.nan)
    for y in Y:
        tr, te = (yr < y) & (yr >= (2022 if name == "gbm+cnn" else 2020)), yr == y
        if name == "gbm+cnn" and y == 2022:
            tr = yr < y  # no day5 scores before 2022: train without them (NaN-aware trees)
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, min_samples_leaf=150, l2_regularization=1.0, random_state=0)
        reg.fit(X[tr], d["target"].to_numpy()[tr])
        pt, ptr = reg.predict(X[te]), reg.predict(X[tr])
        pred[te], pq[te] = pt, np.searchsorted(np.sort(ptr), pt) / len(ptr)
    x = d.with_columns(pred=pl.Series(pred), pq=pl.Series(pq)).filter(pl.col("year") >= 2022)
    ic = x.group_by("year").agg(ic=pl.corr(pl.col("pred").rank(), pl.col("target").rank())).sort("year")["ic"].to_list()
    wd = x.group_by(pl.col("open_time").dt.date()).agg(ic=pl.corr(pl.col("pred").rank(), pl.col("target").rank()), n=pl.len()).filter(pl.col("n") >= 5)["ic"].drop_nans().mean()
    print(f"{name}: IC by year {[round(v, 3) for v in ic]}, within-day IC {wd:+.3f}")
    x.select("row", "pred", "pq").write_parquet(OWN / f"e26_{name.replace('+', '_')}.parquet")
    for q in (0.9, 0.95, 0.98, 0.99):
        c = x.filter(pl.col("pq") >= q).select("row", side=pl.lit(1), prio="pred")
        b = simulate_yearly(c, Rule(hold_min=4320, tp=0.2, slots=15, cost=0.004), years=Y)
        print(f"   q>={q}: {b.summary['bp']:+5.0f}bp n={b.summary['trades']:>3} {fmt_years(b, Y)}")
xd = d.filter((pl.col("year") >= 2022) & (pl.col("pct_day5") >= 0.995)).select("row", side=pl.lit(1), prio="pct_day5")
b = simulate_yearly(xd, Rule(hold_min=4320, tp=0.2, slots=15, cost=0.004), years=Y)
print(f"day5>=0.995 on crash rows: {b.summary['bp']:+5.0f}bp n={b.summary['trades']:>3} {fmt_years(b, Y)}")
