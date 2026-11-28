"""X6 library: the F candidate pool (pump shorts with the funding gate) joined with entry-time
metrics,
simulated once with no slot limit (as d01), plus scorecard and book helpers."""

from __future__ import annotations

import polars as pl
from common_ls import OWN, meta
from diag import annotate, scorecard
from e25_combined import book, sleeve_crash_short, sleeve_listing_weak, sleeve_long
from sim import Rule, simulate

KIND = {0: "time", 1: "stop", 2: "tp", 3: "liq", 4: "stop"}
m = meta()
RP = Rule(hold_min=4320, slots=100_000, cost=0.003, sl=1.0, liq_frac=None)


def pool() -> pl.DataFrame:
    d = pl.read_parquet(OWN / "feats.parquet").filter(
        pl.col("perp")
        & (pl.col("qv_base") >= 20_000)
        & (pl.col("open_time").dt.year() >= 2022)
        & (pl.col("ret_4h") >= 0.4)
        & (pl.col("rate").fill_null(0) >= 0)
    )
    mf = pl.read_parquet(OWN / "metric_feats.parquet")
    return d.join(mf, on="row", how="left")


def trades(sub: pl.DataFrame) -> pl.DataFrame:
    """Simulate (no slot limit, one position per coin while held) the rows of sub; keeps the gate columns."""  # noqa: E501
    t = simulate(
        sub.select("row", side=pl.lit(-1), prio="ret_4h"),
        RP, start_equity=1e12, perp_funding=True, source="perp",
    ).trades  # fmt: skip
    return t.with_columns(
        year=pl.col("open_time").dt.year(),
        exit_kind=pl.col("kind").replace_strict(KIND, default="time"),
    )


def card(t: pl.DataFrame) -> dict[str, float]:
    f = t.select(
        signal=pl.lit(0.9), threshold=pl.lit(0.9), net="net", mfe="mfe", mae="adv",
        tp=pl.lit(None, dtype=pl.Float64), stop=pl.lit(1.0), exit_kind="exit_kind", year="year",
    )  # fmt: skip
    return scorecard(annotate(f))


def as_book(t: pl.DataFrame) -> pl.DataFrame:
    qv = m["qv_base"].to_numpy()
    return t.select(
        "symbol", "adv", t_in=pl.col("t") + 300, t_out="t_out", net="net",
        qv=pl.col("row").map_elements(lambda r: float(qv[r]) * 288, return_dtype=pl.Float64),
        sleeve=pl.lit("P"), prio=pl.lit(2.0) + pl.col("prio").clip(0, 0.99),
    )  # fmt: skip


_lmc: pl.DataFrame | None = None


def lmc() -> pl.DataFrame:
    global _lmc
    if _lmc is None:
        _lmc = pl.concat(
            [sleeve_long(), sleeve_listing_weak(), sleeve_crash_short()], how="vertical_relaxed"
        )
    return _lmc


def run_book(ft: pl.DataFrame, slots: int = 10) -> pl.DataFrame:
    return book(pl.concat([lmc(), as_book(ft)], how="vertical_relaxed"), slots)
