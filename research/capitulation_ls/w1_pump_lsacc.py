"""W1: is the low long/short-account-ratio (crowded shorts) signal for squeezes real or noise?
Univariate
terciles, by year sign, and permutation p-value (deduped, ret_4h>=0.25, with metrics)."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN, first_per_event, meta
from futures_ls import funding_paid
from sim import perp_paths
from sklearn.metrics import roc_auc_score

K = 96 + 63
bp = 1e4
rng = np.random.default_rng(2)
f = pl.read_parquet(OWN / "feats.parquet").filter(
    pl.col("perp")
    & (pl.col("qv_base") >= 20_000)
    & (pl.col("year") >= 2021)
    & (pl.col("ret_4h") >= 0.25)
)
f = f.filter(
    pl.col("row").is_in(
        first_per_event(np.isin(np.arange(meta().height), f["row"].to_numpy())).tolist()
    )
).sort("row")
pp, ok = perp_paths(f["row"].to_numpy())
f = f.filter(pl.Series(ok))
pc = pp[:, 2, : K + 1]
last = np.array([c[~np.isnan(c)][-1] if (~np.isnan(c)).any() else 0.0 for c in pc])
adv = np.nan_to_num(pp[:, 0, : K + 1], nan=0).max(1)
t0 = f["t"].to_numpy() + 300
fund = funding_paid(f["symbol"].to_numpy(), t0, t0 + 72 * 3600, -np.ones(len(f)))
f = f.with_columns(
    net=pl.Series(np.where(adv >= 0.99, -1.0, -last) - 0.003 - fund),
    sq100=pl.Series(adv >= 1),
    sq50=pl.Series(adv >= 0.5),
    fund=pl.Series(fund),
)
m = pl.read_parquet(OWN / "metric_feats.parquet")
d = f.join(m, on="row", how="inner")
print(
    "rows",
    d.height,
    "sq100",
    int(d["sq100"].sum()),
    "years",
    d.group_by("year").len().sort("year").to_dicts(),
)
for c in ("ls_acc", "top_ls_acc", "top_ls_pos", "taker_ls"):
    v = d[c].to_numpy().astype(float)
    ok = ~np.isnan(v)
    y = d["sq100"].to_numpy()[ok]
    auc = roc_auc_score(y, v[ok])
    perm = [roc_auc_score(rng.permutation(y), v[ok]) for _ in range(2000)]
    p = np.mean(np.abs(np.array(perm) - 0.5) >= abs(auc - 0.5))
    print(f"{c}: AUC sq100 {auc:.3f} perm p={p:.4f} (n={ok.sum()}, pos={int(y.sum())})")
d = d.drop_nulls("ls_acc")
t = d.with_columns(q=pl.col("ls_acc").qcut(3, labels=["low", "mid", "high"]))
print(
    t.group_by("q")
    .agg(
        n=pl.len(),
        ls=pl.col("ls_acc").mean(),
        sq50=pl.col("sq50").mean(),
        sq100=pl.col("sq100").mean(),
        net=pl.col("net").mean() * bp,
        fund=pl.col("fund").mean() * bp,
        rate=pl.col("rate").median(),
    )
    .sort("q")
)
print("by year, AUC sq50 of ls_acc (<0.5 means low ls_acc -> squeeze):")
for y in sorted(set(d["year"].to_list())):
    s = d.filter(pl.col("year") == y)
    if s["sq50"].sum() >= 3 and s["sq50"].sum() < s.height:
        print(
            " ",
            y,
            s.height,
            int(s["sq50"].sum()),
            round(roc_auc_score(s["sq50"].to_numpy(), s["ls_acc"].to_numpy()), 3),
        )
