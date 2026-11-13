"""M13: how much of each edge survives when delisted and fallen coins are added?

    MOONSHOT_UNIVERSE=poolA uv run python research/volume_spike/m16_survivorship.py rows
    uv run python research/volume_spike/m16_survivorship.py combine   # all universes, one table

`rows` writes the per-year results of the main strategies for the current universe, split by
whether the coin traded is in the original pool, delisted (group A) or listed outside the pool
(group B). `combine` stacks the universes into results/m16_survivorship.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from common import DATA_DIR, HERE, RESULTS, UNIVERSE, archive_groups
from m13_policy import Ctx, Rule

UNIVERSES = ("pool", "poolA", "all")


def group_of() -> pl.Expr:
    g = archive_groups()
    pool = (DATA_DIR / "top200.txt").read_text().split()
    return (
        pl.when(pl.col("symbol").is_in(pool)).then(pl.lit("pool"))
        .when(pl.col("symbol").is_in(g["A_delisted"])).then(pl.lit("A delisted"))
        .when(pl.col("symbol").is_in(g["B_listed_outside_pool"])).then(pl.lit("B outside pool"))
        .otherwise(pl.lit("other"))
    )  # fmt: skip


def summarise(trades: pl.DataFrame, strategy: str) -> pl.DataFrame:
    """trades: year, symbol, ret, pnl (pnl in units of equity at the start of the year)."""
    t = trades.with_columns(group=group_of())
    total = t.group_by("year").agg(
        trades=pl.len(), mean_bp=pl.col("ret").mean() * 1e4, ret_sum=pl.col("pnl").sum(),
        added_trades=(pl.col("group") != "pool").sum(),
        added_mean_bp=pl.col("ret").filter(pl.col("group") != "pool").mean() * 1e4,
        added_pnl=pl.col("pnl").filter(pl.col("group") != "pool").sum(),
        delisted_trades=(pl.col("group") == "A delisted").sum(),
        delisted_mean_bp=pl.col("ret").filter(pl.col("group") == "A delisted").mean() * 1e4,
    )  # fmt: skip
    return total.with_columns(strategy=pl.lit(strategy), universe=pl.lit(UNIVERSE)).sort("year")


def cnn_rows(variant: str, rule: Rule, name: str) -> pl.DataFrame | None:
    wf = HERE / ("cache" if UNIVERSE == "pool" else f"cache_{UNIVERSE}") / "wf" / f"{variant}.npz"
    if not wf.exists():
        return None
    yrs = Ctx(variant).evaluate(rule)
    rows = [(y["year"], r[3], r[2], r[1]) for y in yrs for r in y["_rows"]]
    t = pl.DataFrame(rows, schema=["year", "symbol", "ret", "pnl"], orient="row")
    out = summarise(t, name)
    cagr = pl.DataFrame(
        [{"year": y["year"], "return": y["cagr"], "max_dd": y["max_dd"]} for y in yrs]
    )
    return out.join(cagr, on="year", how="left")


def rows() -> None:
    parts = []
    for f in ("F1", "F2", "F3"):
        path = RESULTS / f"trades_{f}.csv"
        t = pl.read_csv(path).with_columns(
            year=pl.col("entry_time").str.slice(0, 4).cast(pl.Int32),
            pnl=pl.col("stake") * pl.col("ret"),
        )
        parts.append(summarise(t.select("year", "symbol", "ret", "pnl"), f"{f} rule (5 slots)"))
    for variant, rule, name in (
        ("base", Rule(), "Base CNN, 1h, 5 slots"),
        ("market", Rule(), "Market CNN, 1h, 5 slots"),
        ("path_all", Rule(), "Path forecaster, 1h, 5 slots"),
    ):
        r = cnn_rows(variant, rule, name)
        if r is not None:
            parts.append(r)
    ens = HERE / ("cache" if UNIVERSE == "pool" else f"cache_{UNIVERSE}") / "wf" / "ens_all.npz"
    if ens.exists():
        q = float(np.load(ens)["q"])
        r = cnn_rows("ens_all", Rule(q=q), "CNN ensemble, 1h, 5 slots")
        if r is not None:
            parts.append(r)
    df = pl.concat(parts, how="diagonal_relaxed")
    wf = RESULTS / "m7_walkforward_t_30.csv"
    if wf.exists():
        w = pl.read_csv(wf).with_columns(year=pl.col("start").str.slice(0, 4).cast(pl.Int32))
        w = w.group_by("year").agg(
            day_bp=(pl.col("oos_bp") * pl.col("oos_days")).sum() / pl.col("oos_days").sum(),
            days=pl.col("oos_days").sum(),
        )
        w = w.with_columns(
            strategy=pl.lit("Walk-forward rules (per signal day)"), universe=pl.lit(UNIVERSE)
        )
        df = pl.concat([df, w], how="diagonal_relaxed")
    df.write_csv(RESULTS / "m16_rows.csv")
    with pl.Config(tbl_rows=60, tbl_cols=14, tbl_width_chars=240, float_precision=0):
        print(df.select("strategy", "year", "trades", "mean_bp", "added_trades", "added_mean_bp",
                        "delisted_trades", "delisted_mean_bp", "day_bp"))  # fmt: skip


def combine() -> None:
    parts = []
    for u in UNIVERSES:
        p = (
            HERE / "results" if u == "pool" else HERE / "results" / f"universe_{u}"
        ) / "m16_rows.csv"
        if p.exists():
            parts.append(pl.read_csv(p))
    df = pl.concat(parts, how="diagonal_relaxed")
    out = Path(HERE / "results" / "m16_survivorship.csv")
    df.write_csv(out)
    piv = df.pivot(on="universe", index=["strategy", "year"], values="mean_bp").sort(
        "strategy", "year"
    )
    with pl.Config(tbl_rows=80, tbl_cols=8, tbl_width_chars=200, float_precision=0):
        print(piv)


if __name__ == "__main__":
    {"rows": rows, "combine": combine}[sys.argv[1]]()
