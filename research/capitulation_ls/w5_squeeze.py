"""W5e: why pump-short squeezes cluster; pump-cap, sleeve weights, weekday filters in the book."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN, meta
from e25_combined import book, sleeve_crash_short, sleeve_listing_weak, sleeve_long, sleeve_pump
from futures_ls import perp_window

pw = perp_window()
hours = np.load("/tmp/w5_btc_hours.npy")
btc = np.load("/tmp/w5_btc.npy")
feats = pl.read_parquet(
    OWN / "feats.parquet",
    columns=[
        "row",
        "mkt_r24h",
        "mkt_r4h",
        "r7d",
        "r30d",
        "ret_4h",
        "n_prior_pumps30",
        "listed_days",
        "year",
    ],
)
P = sleeve_pump()
m = meta()
# re-simulate to get row ids: sleeve_pump drops them; rebuild by joining on symbol & entry time
mm = pl.DataFrame(
    {"row": np.arange(m.height), "symbol": m["symbol"], "t0": m["open_time"].dt.epoch("s")}
)
P = (
    P.with_columns(t0=pl.col("t_in") - 300)
    .join(mm, on=["symbol", "t0"], how="left")
    .join(
        feats.select("row", "mkt_r24h", "r7d", "r30d", "ret_4h", "n_prior_pumps30", "listed_days"),
        on="row",
        how="left",
    )
)


def bfwd(ti, h=72):
    k = np.searchsorted(hours, ti, side="right") - 1
    return btc[min(k + h, len(btc) - 1)] / btc[k] - 1


def b7(ti):
    k = np.searchsorted(hours, ti, side="right") - 1
    return btc[k] / btc[max(k - 168, 0)] - 1


P = P.with_columns(
    bf=pl.Series([bfwd(x) for x in P["t_in"].to_list()]),
    b7=pl.Series([b7(x) for x in P["t_in"].to_list()]),
    day=pl.from_epoch("t_in").dt.date(),
    year=pl.from_epoch("t_in").dt.year(),
    delist=pl.Series(
        [
            pw.get(s, (0, 9e18))[1] < 1.7727e9 and pw.get(s, (0, 9e18))[1] - ti < 30 * 86400
            for s, ti in zip(P["symbol"].to_list(), P["t_in"].to_list(), strict=False)
        ]
    ),
)
print(
    "pump-short trades (r4>=40%, perp, 72h, iso 1x):",
    P.height,
    "mean bp",
    round(float(P["net"].mean()) * 1e4),
    "median",
    round(float(P["net"].median()) * 1e4),
    "days",
    P["day"].n_unique(),
)
bad = P.filter(pl.col("net") <= -0.30)
print(
    "trades <= -30%:",
    bad.height,
    "(",
    round(bad.height / P.height * 100, 1),
    "%) on",
    bad["day"].n_unique(),
    "days; sum net of those",
    round(float(bad["net"].sum()), 1),
    " vs total sum",
    round(float(P["net"].sum()), 1),
    " sum of all other trades",
    round(float(P.filter(pl.col("net") > -0.3)["net"].sum()), 1),
)
print(
    " by year bad/total:",
    P.group_by("year")
    .agg(
        n=pl.len(),
        bad=(pl.col("net") <= -0.3).sum(),
        liq=(pl.col("net") <= -0.99).sum(),
        bp=(pl.col("net").mean() * 1e4).round(),
    )
    .sort("year")
    .to_dicts(),
)
print(
    " delisting-within-30d share: bad",
    round(float(bad["delist"].mean()), 2),
    "all",
    round(float(P["delist"].mean()), 3),
    " mean net delist-trades",
    round(float(P.filter(pl.col("delist"))["net"].mean()) * 1e4),
    "bp n",
    int(P["delist"].sum()),
)
print(
    " young (<90d listed) share: bad",
    round(float((bad["listed_days"] < 90).mean()), 2),
    "all",
    round(float((P["listed_days"] < 90).mean()), 2),
)
print(" worst 15 trades:")
for r in P.sort("net").head(15).iter_rows(named=True):
    print(
        f"   {r['day']} {r['symbol']:14s} net {r['net'] * 100:+6.0f}% r4 {r['ret_4h'] * 100:.0f}% BTC7d-before {r['b7'] * 100:+.1f}% BTC fwd72 {r['bf'] * 100:+.1f}% mkt24h {r['mkt_r24h'] * 100:+.1f}% delist30 {r['delist']}"  # noqa: E501
    )
# clustering: loss mass by day / by week
dl = (
    P.group_by("day")
    .agg(n=pl.len(), loss=pl.col("net").filter(pl.col("net") < 0).sum(), s=pl.col("net").sum())
    .sort("loss")
)
print(" top 8 loss days:", dl.head(8).select("day", "n", pl.col("loss").round(2)).to_dicts())
tot = float(P.filter(pl.col("net") < 0)["net"].sum())
print(
    " share of total loss mass on worst 5 / 10 / 20 days:",
    [round(float(dl.head(k)["loss"].sum() / tot), 2) for k in (5, 10, 20)],
    "of",
    dl.height,
    "days;",
    "share of trades on those days",
    [round(float(dl.head(k)["n"].sum() / P.height), 2) for k in (5, 10, 20)],
)
# overlap in time: weeks with >=2 trades<= -30%
wk = bad.with_columns(w=pl.col("t_in") // (7 * 86400)).group_by("w").agg(n=pl.len())
print(
    " weeks with at least one <=-30% trade:",
    wk.height,
    " of which >=2:",
    int((wk["n"] >= 2).sum()),
    " (expected if independent over",
    int(P.with_columns(w=pl.col("t_in") // (7 * 86400))["w"].n_unique()),
    "active weeks)",
)
# BTC context
P = P.with_columns(
    btc7=pl.when(pl.col("b7") < -0.03)
    .then(pl.lit("BTC7d<-3%"))
    .when(pl.col("b7") > 0.05)
    .then(pl.lit("BTC7d>+5%"))
    .otherwise(pl.lit("mid"))
)
print(
    " by BTC 7d regime:",
    P.group_by("btc7")
    .agg(
        n=pl.len(),
        bp=(pl.col("net").mean() * 1e4).round(),
        med=(pl.col("net").median() * 1e4).round(),
        bad=(pl.col("net") <= -0.3).mean().round(3),
        liq=(pl.col("net") <= -0.99).sum(),
    )
    .to_dicts(),
)
print(
    " by BTC fwd72h tercile (outcome, not tradeable):",
    P.with_columns(q=pl.col("bf").qcut(3, labels=["BTC falls", "flat", "BTC rallies"]))
    .group_by("q")
    .agg(
        n=pl.len(),
        bp=(pl.col("net").mean() * 1e4).round(),
        bad=(pl.col("net") <= -0.3).mean().round(3),
    )
    .sort("q")
    .to_dicts(),
)
print(
    " corr(net, BTC fwd72) ",
    round(float(np.corrcoef(P["net"].to_numpy(), P["bf"].to_numpy())[0, 1]), 3),
    " corr(net, mkt_r24h at entry)",
    round(float(np.corrcoef(P["net"].to_numpy(), P["mkt_r24h"].fill_null(0).to_numpy())[0, 1]), 3),
)
print(
    " by prior pumps in 30d:",
    P.with_columns(q=pl.col("n_prior_pumps30").clip(0, 3))
    .group_by("q")
    .agg(
        n=pl.len(),
        bp=(pl.col("net").mean() * 1e4).round(),
        bad=(pl.col("net") <= -0.3).mean().round(3),
    )
    .sort("q")
    .to_dicts(),
)
# Same-day pumps count: pumps arriving in clusters (market-wide alt pump day)
cnt = P.group_by("day").agg(k=pl.len())
P = P.join(cnt, on="day")
print(
    " by # pump trades same day:",
    P.with_columns(q=pl.col("k").clip(1, 6))
    .group_by("q")
    .agg(
        n=pl.len(),
        bp=(pl.col("net").mean() * 1e4).round(),
        bad=(pl.col("net") <= -0.3).mean().round(3),
    )
    .sort("q")
    .to_dicts(),
)

print("\n=== book tweaks (LMC + regime pump, 10 slots) ===")
L, Pr, Cc, M = sleeve_long(), None, sleeve_crash_short(), sleeve_listing_weak()
from e25_combined import sleeve_pump_regime  # noqa: E402

Q = sleeve_pump_regime()
tr = pl.concat([L, Q, Cc, M], how="vertical_relaxed")


def show(lab, b):
    n = int(b["trades"].sum())
    bp = float((b["bp"] * b["trades"]).sum() / n)
    print(
        f"{lab:34s} {bp:+5.0f}bp min {b['ret'].min() * 100:+4.0f}% worst dd {b['dd'].min() * 100:+4.0f}% | "  # noqa: E501
        + " ".join(
            f"{r['year']}:{r['ret'] * 100:+.0f}%/dd{r['dd'] * 100:+.0f}"
            for r in b.iter_rows(named=True)
        )
    )


show("LQMC uncapped", book(tr, 10))
for cap in (1, 2, 3):
    show(f"LQMC pump cap {cap}", book(tr, 10, caps={"P": cap}))
for lab, w in (
    ("weights L1.5", {"L": 1.5}),
    ("weights L0.5", {"L": 0.5}),
    ("weights P0.5", {"P": 0.5}),
    ("weights N0.5", {"N": 0.5}),
    ("weights C1.5", {"C": 1.5}),
    ("weights C0.5", {"C": 0.5}),
):
    show(lab, book(tr, 10, weights=w))
# weekday
print("\nsleeve mean net by weekday group (n, bp) per year, weekend = Sat/Sun entry:")
for name, s in (("L", L), ("Q", Q), ("C", Cc), ("M", M)):
    s = s.with_columns(
        wd=pl.from_epoch("t_in").dt.weekday(), year=pl.from_epoch("t_in").dt.year()
    ).with_columns(we=pl.col("wd") >= 6)
    g = (
        s.group_by("year", "we")
        .agg(n=pl.len(), bp=(pl.col("net").mean() * 1e4).round())
        .sort("year", "we")
    )
    wide = {}
    for r in g.iter_rows(named=True):
        wide.setdefault(r["year"], {})["we" if r["we"] else "wk"] = (r["n"], int(r["bp"]))
    allwe = s.filter(pl.col("we"))["net"]
    allwk = s.filter(~pl.col("we"))["net"]
    se = np.sqrt(allwe.var() / len(allwe) + allwk.var() / len(allwk))
    print(
        f" {name}: pooled weekend n={len(allwe)} {allwe.mean() * 1e4:+.0f}bp vs weekday n={len(allwk)} {allwk.mean() * 1e4:+.0f}bp; diff t={(allwe.mean() - allwk.mean()) / se:+.2f} |",  # noqa: E501
        wide,
    )
