"""w8 shared: build trades for the pump short at a ret_4h threshold, with and without the
funding filter,
joined back to feats (+ metrics). Trades from the booked simulator (slots=10, 72h, isolated 1x)."""

import polars as pl
from common_ls import OWN
from sim import Rule, simulate_yearly

RULE = Rule(hold_min=4320, slots=10, cost=0.003, sl=1.0)


def feats():
    f = pl.read_parquet(OWN / "feats.parquet").filter(
        pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("year") >= 2022)
    )
    m = pl.read_parquet(OWN / "metric_feats.parquet")
    return f.join(m, on="row", how="left")


def trades(F, thr, sign):
    """sign: 'pos' rate>=0, 'neg' rate<0, 'all'. Returns trades joined to F columns."""
    c = F.filter(pl.col("ret_4h") >= thr)
    if sign == "pos":
        c = c.filter(pl.col("rate") >= 0)
    elif sign == "neg":
        c = c.filter(pl.col("rate") < 0)
    c = c.select("row", pl.lit(-1).alias("side"), pl.col("ret_4h").alias("prio"))
    b = simulate_yearly(
        c, RULE, perp_funding=True, source="perp", years=(2022, 2023, 2024, 2025, 2026)
    )
    t = b.trades
    return t


if __name__ == "__main__":
    F = feats()
    print(F.height)
    t = trades(F, 0.4, "pos")
    print(t.columns, t.height, t["net"].mean() * 1e4)
    print(t.head(3))
