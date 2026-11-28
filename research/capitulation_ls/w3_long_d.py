"""W3-D: crash-event rankers trained on the TP20 target: year vs within-day skill, pick clustering."""  # noqa: E501

import numpy as np
import polars as pl
from common_ls import OWN, meta, scores

m = meta()
pct5 = scores("day5")
T = pl.read_parquet(OWN / "target_long_tp20.parquet")
for f in ("nn_long_transformer_72h_tp20", "e26_gbm"):
    R = pl.read_parquet(OWN / f"{f}.parquet")
    print(f, R.columns, R.height)
G = pl.read_parquet(OWN / "e26_gbm.parquet")
N = pl.read_parquet(OWN / "nn_long_transformer_72h_tp20.parquet")
base = T.join(G.rename({"pred": "gbm", "pq": "gbm_q"}), on="row")
ncols = [c for c in N.columns if c != "row"]
print(ncols)
base = base.join(N, on="row", how="left")
mm = m.select(pl.int_range(pl.len()).alias("row"), "day", "year", "symbol").with_columns(
    pl.col("day").cast(pl.Date)
)
d = (
    base.join(mm, on="row")
    .with_columns(
        day5=pl.Series(pct5)[base["row"].to_numpy()]
        if False
        else pl.Series(pct5[base["row"].to_numpy()])
    )
    .filter(pl.col("year") >= 2022)
)
sc = ["gbm", *[c for c in ncols if c in d.columns][:1], "day5"]
print("scores:", sc, "rows", d.height)


def wic(s, a, b="target"):
    s = s.drop_nulls([a, b])
    return s.select(
        pl.corr(pl.col(a) - pl.col(a).mean().over("day"), pl.col(b) - pl.col(b).mean().over("day"))
    ).item()


for a in sc:
    s = d.drop_nulls(a)
    yrs = [
        round(
            s.filter(pl.col("year") == y)
            .select(pl.corr(pl.col(a).rank(), pl.col("target").rank()))
            .item(),
            3,
        )
        for y in (2022, 2023, 2024, 2025, 2026)
    ]
    # day-mean score vs day-mean target (between-day), across days
    g = (
        s.group_by("day")
        .agg(a=pl.col(a).mean(), t=pl.col("target").mean(), n=pl.len())
        .filter(pl.col("n") >= 5)
    )
    bd = np.corrcoef(g["a"].rank().to_numpy(), g["t"].rank().to_numpy())[0, 1]
    print(
        f" {a:10s} yearly spearman {yrs}; within-day pooled {wic(s, a):+.3f}; between-day spearman of day means {bd:+.3f} ({g.height} days); share of score variance that is day-level {1 - s.select((pl.col(a) - pl.col(a).mean().over('day')).var()).item() / s[a].var():.2f}"  # noqa: E501
    )
print(
    f" target: day-level variance share {1 - d.select((pl.col('target') - pl.col('target').mean().over('day')).var()).item() / d['target'].var():.2f} (unadjusted)"  # noqa: E501
)
print("== top 1% picks (by each ranker, own percentile of rows with score) on crash rows 2022+")
dm = d.with_columns(dmean=pl.col("target").mean().over("day"))
bm = d["target"].mean()
for a in sc:
    s = dm.drop_nulls(a)
    thr = s[a].quantile(0.99)
    p = s.filter(pl.col(a) >= thr)
    days = p.group_by("day").agg(n=pl.len()).sort("n", descending=True)
    yc = p.group_by("year").agg(n=pl.len()).sort("year")
    print(
        f" {a:8s} n={p.height} days={days.height} top3-day share {days['n'].head(3).sum() / p.height:.2f} by year {dict(zip(yc['year'].to_list(), yc['n'].to_list(), strict=False))} mean target {p['target'].mean() * 1e4:+.0f}bp = base {s['target'].mean() * 1e4:+.0f} + day-part {(p['dmean'].mean() - s['target'].mean()) * 1e4:+.0f} + within-day {(p['target'] - p['dmean']).mean() * 1e4:+.0f}"  # noqa: E501
    )
print("== per-day mean Spearman (days with n>=5) and by-day-size")
for a in sc:
    s = d.drop_nulls(a).with_columns(n=pl.len().over("day"))
    for lo, hi, lab in ((5, 10**9, "n>=5"), (5, 29, "5-29"), (30, 10**9, "n>=30")):
        z = (
            s.filter((pl.col("n") >= lo) & (pl.col("n") <= hi))
            .group_by("day")
            .agg(ic=pl.corr(pl.col(a).rank(), pl.col("target").rank()), n=pl.len())
            .filter(pl.col("ic").is_finite())
        )
        print(
            f" {a:6s} {lab:6s}: mean per-day IC {z['ic'].mean():+.3f} over {z.height} days (nan-dropped)"  # noqa: E501
        )
