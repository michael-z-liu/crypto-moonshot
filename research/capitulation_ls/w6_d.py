"""w6-D: signal net vs time from entry to index low; W sweep without 2025-10-10; early vs late paired by episode."""  # noqa: E501

import polars as pl
from common_ls import scores
from sim import Rule, simulate_yearly

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(200)
E = pl.read_parquet("/tmp/w6_ep.parquet")
S = pl.read_parquet("/tmp/w6_sig2.parquet")
big = E.filter(pl.col("sigs") >= 20).select("ep", "low_min")
s = S.join(big, on="ep").with_columns(rel=pl.col("off") + 5 - pl.col("low_min"))
b = s.with_columns(
    bk=pl.when(pl.col("rel") < -30)
    .then(pl.lit("a: >30m before low"))
    .when(pl.col("rel") < -10)
    .then(pl.lit("b: 10-30m before"))
    .when(pl.col("rel") < 0)
    .then(pl.lit("c: 0-10m before"))
    .when(pl.col("rel") <= 10)
    .then(pl.lit("d: 0-10m after"))
    .when(pl.col("rel") <= 30)
    .then(pl.lit("e: 10-30m after"))
    .otherwise(pl.lit("f: >30m after"))
)
print(
    "per signal (big episodes, unconstrained 1x TP20/72h, cost .4%), bucket by entry vs index low"
)
print(
    b.group_by("bk")
    .agg(
        n=pl.len(),
        eps=pl.col("ep").n_unique(),
        net_bp=pl.col("net").mean() * 1e4,
        tp=(pl.col("net") > 0.19).mean(),
        mae=pl.col("mae").mean(),
        mae_med=pl.col("mae").median(),
    )
    .sort("bk")
)
# same excluding 10-10 episode
e1010 = E.filter(pl.col("day").str.starts_with("2025-10-10 20"))["ep"][0]
print("excluding 10-10 episode")
print(
    b.filter(pl.col("ep") != e1010)
    .group_by("bk")
    .agg(
        n=pl.len(),
        eps=pl.col("ep").n_unique(),
        net_bp=pl.col("net").mean() * 1e4,
        mae=pl.col("mae").mean(),
    )
    .sort("bk")
)
# episode-equal-weighted: mean of per-episode means
pe = b.group_by("ep", "bk").agg(net=pl.col("net").mean())
print("episode-weighted net (bp) by bucket")
print(pe.group_by("bk").agg(eps=pl.len(), bp=pl.col("net").mean() * 1e4).sort("bk"))
# paired early(<20) vs late(>=20) per ep
p = E.filter(pl.col("n_late") > 0).filter(pl.col("n_early") > 0)
print(
    "episodes with both early(<20min) and late signals:",
    p.height,
    " late>early in",
    (p["late_net"] > p["early_net"]).sum(),
)
print(
    "mean diff late-early (bp) unweighted:",
    (p["late_net"] - p["early_net"]).mean() * 1e4,
    " median:",
    (p["late_net"] - p["early_net"]).median() * 1e4,
)
pp = p.filter(pl.col("ep") != e1010)
print(
    " excl 10-10: ",
    pp.height,
    (pp["late_net"] - pp["early_net"]).mean() * 1e4,
    (pp["late_net"] - pp["early_net"]).median() * 1e4,
    (pp["late_net"] > pp["early_net"]).sum(),
)
# book W sweep without 10-10
pct = scores("day5")
sg = S.with_columns(prio=pl.Series(pct[S["row"].to_numpy()]))
Y = (2022, 2023, 2024, 2025, 2026)
for excl in (False, True):
    for W in (0, 5, 10, 20, 30):
        c = sg.filter(pl.col("off") >= W)
        if excl:
            c = c.filter(pl.col("ep") != e1010)
        c = c.select("row", pl.lit(1).alias("side"), "prio")
        r = simulate_yearly(c, Rule(hold_min=4320, tp=0.2, slots=10, cost=0.004), years=Y)
        print(
            f"excl1010={excl} W={W}: n={r.summary['trades']} bp={r.summary['bp']:.0f} | "
            + " ".join(
                f"{x['year']}:{x['trades']}t {x['ret'] * 100:+.0f}%"
                for x in r.years.iter_rows(named=True)
            )
        )
# W by trades: share of signals lost by W per year
print(
    S.group_by("year")
    .agg(*[(pl.col("off") >= W).sum().alias(f"sig_W{W}") for W in (0, 10, 20, 30)])
    .sort("year")
)
