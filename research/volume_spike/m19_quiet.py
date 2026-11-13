"""Quiet coins before a spike (B3). Protocol: docs/plans/cnn-replication.md, milestone B3.

Step 1 (`step1`): how quiet snapshots resolve, by quiet-score decile (cut on 2020-21) and year.
Step 2 (`rules`): quiet entry rules through the $50 no-stop book of m18_longshot.

Run: MOONSHOT_UNIVERSE=all uv run python research/volume_spike/m19_quiet.py step1|rules
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import numpy as np
import polars as pl
from common import BARS_PER_DAY, CACHE, RESULTS

LAST_ENTRY = datetime(2026, 10, 1, tzinfo=UTC)
MIN_QV = 1_000.0
D3, D7 = 3 * BARS_PER_DAY, 7 * BARS_PER_DAY
# snapshots whose 7-day window covers the 2025-10-10/11 cascade
CASCADE = (datetime(2025, 10, 3, tzinfo=UTC), datetime(2025, 10, 12, tzinfo=UTC))


def load() -> pl.DataFrame:
    q = pl.read_parquet(CACHE / "quiet.parquet")
    q = q.filter(
        (pl.col("open_time") < LAST_ENTRY)
        & (pl.col("qv_med") >= MIN_QV)
        & pl.col("q1").is_not_null()
        & pl.col("q7").is_not_null()
        & pl.col("q1").is_finite()
        & pl.col("q7").is_finite()
    )
    cut = q.filter(pl.col("year") <= 2021)
    out = q
    for s in ("q1", "q7"):
        edges = np.quantile(cut[s].to_numpy(), np.linspace(0.1, 0.9, 9))
        out = out.with_columns(
            pl.Series(f"{s}_dec", np.searchsorted(edges, out[s].to_numpy()) + 1).cast(pl.Int8)
        )
    return out.with_columns(
        cascade=pl.col("open_time").is_between(*CASCADE, closed="left"),
        period=pl.when(pl.col("year") <= 2021)
        .then(pl.lit("2020-21"))
        .when(pl.col("year") <= 2024)
        .then(pl.lit("2022-24"))
        .otherwise(pl.lit("2025-26")),
    )


def outcome_aggs() -> list[pl.Expr]:
    up = lambda c, n: ((pl.col(c) >= 0) & (pl.col(c) < n)).mean() * 100  # noqa: E731
    first = (pl.col("up_20") >= 0) & ((pl.col("dn_20") < 0) | (pl.col("up_20") < pl.col("dn_20")))
    return [
        pl.len().alias("n"),
        up("up_20", D3).alias("up20_3d"),
        up("dn_20", D3).alias("dn20_3d"),
        up("up_10", D3).alias("up10_3d"),
        up("dn_10", D3).alias("dn10_3d"),
        up("up_30", D7).alias("up30_7d"),
        up("dn_30", D7).alias("dn30_7d"),
        (first.mean() * 100).alias("up20_first"),
        (pl.col("c_3d").mean() * 100).alias("ret3d"),
        (pl.col("c_7d").mean() * 100).alias("ret7d"),
        (pl.col("c_7d").median() * 100).alias("med7d"),
    ]


def step1() -> None:
    q = load()
    print(q.height, "snapshots used,", q["symbol"].n_unique(), "coins")
    rows = []
    for s in ("q1", "q7"):
        for keep_cascade in (True, False):
            d = q if keep_cascade else q.filter(~pl.col("cascade"))
            tag = "" if keep_cascade else " ex-cascade"
            for by in ("period", "year"):
                g = d.group_by(f"{s}_dec", by).agg(outcome_aggs())
                a = (
                    d.group_by(by)
                    .agg(outcome_aggs())
                    .with_columns(pl.lit(0).cast(pl.Int8).alias(f"{s}_dec"))
                )
                g = pl.concat([g, a.select(g.columns)])
                rows.append(
                    g.rename({f"{s}_dec": "decile", by: "when"}).with_columns(
                        score=pl.lit(s + tag), by=pl.lit(by), when=pl.col("when").cast(pl.String)
                    )
                )
    out = pl.concat(rows)
    out.write_csv(RESULTS / "m19_step1.csv")
    with pl.Config(tbl_rows=200, tbl_cols=16, tbl_width_chars=240, float_precision=2):
        for s in ("q1", "q7"):
            print(
                out.filter((pl.col("score") == s) & (pl.col("by") == "period")).sort(
                    "when", "decile"
                )
            )


# ----------------------------------------------------------------------------- step 2
EX_CASCADE = (datetime(2025, 9, 10, tzinfo=UTC), datetime(2025, 10, 12, tzinfo=UTC))


def as_book(q: pl.DataFrame) -> pl.DataFrame:
    """Columns the m18 book expects (hit_* for take-profit times)."""
    return q.rename({f"up_{k}": f"hit_{k}" for k in (20, 30, 50, 100)})


RULES: dict[str, pl.Expr] = {
    "Q0 every snapshot": pl.col("q7").is_not_null(),
    "QA quiet day (Q1 decile 1)": pl.col("q1_dec") == 1,
    "QB quiet week (Q7 decile 1)": pl.col("q7_dec") == 1,
    "QC quiet day and week": (pl.col("q1_dec") == 1) & (pl.col("q7_dec") == 1),
    "QD quiet week + early buying": (pl.col("q7_dec") <= 2)
    & (pl.col("tbr_diff") > 0)
    & (pl.col("trades_rel") > 1),
}


def rules() -> None:
    from m18_longshot import run_book, show

    q = as_book(load()).with_columns(prio=pl.col("q7"))
    out = []
    for name, mask in RULES.items():
        d = q.filter(mask.fill_null(False))
        print(name, d.height, "signals", flush=True)
        out += run_book(d, name, "prio", range(2020, 2027))
        # without entries whose holding window can cover the 2025-10-10/11 cascade
        dx = d.filter(~pl.col("open_time").is_between(*EX_CASCADE, closed="left"))
        out += run_book(dx, name + " ex-cascade", "prio", range(2025, 2026))
    df = pl.DataFrame(out)
    df.write_csv(RESULTS / "m19_rules.csv")
    show(df.filter(~pl.col("entry").str.ends_with("ex-cascade")))


if __name__ == "__main__":
    {"step1": step1, "rules": rules}[sys.argv[1]]()
