"""E11: evaluate prediction files (row, pred, pq) for a side: IC per year, within-day IC, books.

Run: uv run --group research python e11_eval.py short gbm=e08_short_72h mlp=nn_short_mlp_72h ...
A name "ens" averages the pq of all given files (rows present in all).
"""

from __future__ import annotations

import sys

import polars as pl
from common_ls import OWN
from sim import Rule, fmt_years, simulate_yearly

SIDE = sys.argv[1]
S = -1 if SIDE == "short" else 1
files = dict(a.split("=") for a in sys.argv[2:])
feats = pl.read_parquet(OWN / "feats.parquet").filter(pl.col("qv_base") >= 20_000)
feats = feats.filter(pl.col("ret_4h") >= 0.15) if S < 0 else feats.filter(pl.col("ret_4h") <= -0.10)
feats = feats.with_columns(
    target=(S * pl.col("y72h")).clip(-0.5, 1.0),
    year=pl.col("open_time").dt.year(),
    day=pl.col("open_time").dt.date(),
)
preds = {
    k: pl.read_parquet(OWN / f"{v}.parquet").select("row", pred=pl.col("pred"), pq=pl.col("pq"))
    for k, v in files.items()
}
if len(preds) > 1:
    j = None
    for k, p in preds.items():
        q = p.select("row", pl.col("pq").alias(k))
        j = q if j is None else j.join(q, on="row")
    assert j is not None
    j = j.with_columns(pq=pl.mean_horizontal(list(preds)), pred=pl.mean_horizontal(list(preds)))
    preds["ens"] = j.select("row", "pred", "pq")

for name, p in preds.items():
    d = feats.join(p, on="row").filter(pl.col("year") >= 2022).drop_nulls("pred")
    ic = (
        d.group_by("year")
        .agg(ic=pl.corr(pl.col("pred").rank(), pl.col("target").rank()))
        .sort("year")
    )
    wd = (
        d.group_by("day")
        .agg(ic=pl.corr(pl.col("pred").rank(), pl.col("target").rank()), n=pl.len())
        .filter(pl.col("n") >= 5)
    )
    print(
        f"== {name}: IC by year {[round(v, 3) for v in ic['ic'].to_list()]}, within-day IC {wd['ic'].drop_nans().drop_nulls().mean():+.3f} over {wd.height} days"  # noqa: E501
    )
    base = d.filter(pl.col("perp")) if S < 0 else d
    for q in (0.5, 0.8, 0.9):
        c = base.filter(pl.col("pq") >= q).select("row", side=pl.lit(S), prio="pred")
        rule = (
            Rule(hold_min=4320, slots=10, cost=0.003, sl=1.0)
            if S < 0
            else Rule(hold_min=4320, slots=5, cost=0.003)
        )
        b = simulate_yearly(c, rule, perp_funding=S < 0, source="perp")
        print(
            f"   q>={q:.1f}: {b.summary['bp']:+5.0f}bp {fmt_years(b, (2022, 2023, 2024, 2025, 2026))}"  # noqa: E501
        )
    if S < 0:  # the rule's set (pump >= 40%) with the model as a veto on its bottom part
        for q in (0.0, 0.3, 0.5):
            c = base.filter((pl.col("ret_4h") >= 0.4) & (pl.col("pq") >= q)).select(
                "row", side=pl.lit(S), prio="pred"
            )
            b = simulate_yearly(
                c,
                Rule(hold_min=4320, slots=10, cost=0.003, sl=1.0),
                perp_funding=True,
                source="perp",
            )
            print(
                f"   r4>=0.4 & q>={q:.1f}: {b.summary['bp']:+5.0f}bp {fmt_years(b, (2022, 2023, 2024, 2025, 2026))}"  # noqa: E501
            )
