"""w6-C: episode table (compact), book trades by episode for W=0/10/20, share of gain by year/episode."""  # noqa: E501

import polars as pl

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(30)
E = pl.read_parquet("/tmp/w6_ep.parquet")
print(
    E.select(
        "ep",
        "day",
        "sigs",
        "coins",
        "dur",
        "low_min",
        "low_pct",
        "ew_at_60",
        "n_early",
        "n_late",
        (pl.col("early_net") * 1e4).round(0).alias("early_bp"),
        (pl.col("late_net") * 1e4).round(0).alias("late_bp"),
        pl.col("share_before_low").round(2).alias("sh_pre_low"),
        "last_vs_low",
    )
)
# low_min vs dur
print("eps with dur<20:", (E["dur"] < 20).sum(), " dur>=20:", (E["dur"] >= 20).sum())
print("low within first 5/20 min of start:", (E["low_min"] <= 5).sum(), (E["low_min"] <= 20).sum())
big = E.filter(pl.col("sigs") >= 20)
print(
    "big episodes (>=20 signals):",
    big.height,
    "low_min median",
    big["low_min"].median(),
    "dur median",
    big["dur"].median(),
)
print(
    "big: share with first-20-min entry before low:", (big["low_min"] > 20).sum(), "of", big.height
)
for W in (0, 10, 20, 30):
    tr = pl.read_parquet(f"/tmp/w6_trades_W{W}.parquet")
    g = (
        tr.group_by("ep")
        .agg(n=pl.len(), bp=pl.col("net").mean() * 1e4, tot=pl.col("net").sum())
        .sort("ep")
    )
    g = g.join(E.select("ep", "day"), on="ep")
    print(f"--- W={W}: trades by episode (n, mean bp)")
    print(g.sort("tot", descending=True).head(8))
    print(
        "   episodes used:",
        g.height,
        " trades from top episode:",
        g["n"].max(),
        " share of sum(net) from top-2 eps:",
        g.sort("tot", descending=True)["tot"][:2].sum() / g["tot"].sum(),
    )
    print(
        "   trades by year/ep count:",
        tr.group_by("year").agg(n=pl.len(), eps=pl.col("ep").n_unique()).sort("year").rows(),
    )
# W=20 without 2025-10-10 episode
tr = pl.read_parquet("/tmp/w6_trades_W20.parquet")
e1010 = E.filter(pl.col("day").str.starts_with("2025-10-10 20"))["ep"][0]
t2 = tr.filter(pl.col("ep") != e1010)
print("W20 excl 10-10: n", t2.height, "bp", t2["net"].mean() * 1e4)
print(
    "W20 10-10 trades",
    tr.filter(pl.col("ep") == e1010).height,
    "bp",
    tr.filter(pl.col("ep") == e1010)["net"].mean() * 1e4,
)
t0 = pl.read_parquet("/tmp/w6_trades_W0.parquet")
print(
    "W0 excl 10-10: n",
    t0.filter(pl.col("ep") != e1010).height,
    "bp",
    t0.filter(pl.col("ep") != e1010)["net"].mean() * 1e4,
)
print(
    "W0 10-10 trades",
    t0.filter(pl.col("ep") == e1010).height,
    "bp",
    t0.filter(pl.col("ep") == e1010)["net"].mean() * 1e4,
)
# 2025 W20 by episode
print(
    tr.filter(pl.col("year") == 2025)
    .group_by("ep")
    .agg(n=pl.len(), bp=pl.col("net").mean() * 1e4)
    .join(E.select("ep", "day"), on="ep")
)
print(
    t0.filter(pl.col("year") == 2025)
    .group_by("ep")
    .agg(n=pl.len(), bp=pl.col("net").mean() * 1e4)
    .join(E.select("ep", "day"), on="ep")
)
# 10-10 first 10 trades vs later in W0, by offset
t = t0.filter(pl.col("ep") == e1010).sort("off")
print(t.select("off", "net", "mae" if "mae" in t.columns else "adv", "symbol").head(12))
