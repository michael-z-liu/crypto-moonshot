"""W5a: why crowded funding persists. Persistence of extreme rates, price after prints by regime,
squeeze vs continuation tails, concentration, dying-coin share."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN
from futures_ls import funding, funding_paid, spot_name

P = pl.read_parquet(
    OWN / "futures" / "perp_1h.parquet",
    columns=["symbol", "open_time", "open", "high", "low", "close", "quote_volume"],
)
P = P.with_columns(h=pl.col("open_time").dt.epoch("s")).sort("symbol", "h")
perp = {
    k[0]: tuple(g[c].to_numpy() for c in ("h", "open", "high", "low", "close", "quote_volume"))
    for k, g in P.group_by(["symbol"], maintain_order=True)
}
bh, bo, _, _, bcl, _ = perp["BTCUSDT"]


def btc7(t):
    k = np.searchsorted(bh, t, side="right") - 1
    j = np.searchsorted(bh, t - 7 * 86400, side="right") - 1
    return bcl[max(k, 0)] / bcl[max(j, 0)] - 1


f = (
    funding()
    .filter(pl.col("time").dt.year() >= 2021)
    .with_columns(t=pl.col("time").dt.epoch("s"))
    .sort("symbol", "t")
)
last_t = f.group_by("symbol").agg(end=pl.col("t").max()).to_dict(as_series=False)
end = dict(zip(last_t["symbol"], last_t["end"], strict=False))
DATA_END = f["t"].max()
print("funding rows", f.height, "symbols", f["symbol"].n_unique())
# 1. persistence: P(rate still >= thr after k prints | rate >= thr), per-year
f = f.with_columns(
    nxt1=pl.col("rate").shift(-1).over("symbol"),
    nxt3=pl.col("rate").shift(-3).over("symbol"),
    nxt9=pl.col("rate").shift(-9).over("symbol"),
    nxt27=pl.col("rate").shift(-27).over("symbol"),
    prev1=pl.col("rate").shift(1).over("symbol"),
    year=pl.col("time").dt.year(),
)
for thr in (0.001, 0.003):
    x = f.filter(pl.col("rate") >= thr)
    print(f"\n== persistence after rate >= {thr}: n={x.height}, symbols={x['symbol'].n_unique()}")
    print(
        " mean next rate k=1,3,9,27:",
        [
            round(float(x[c].drop_nulls().mean()) * 1e4, 1)
            for c in ("nxt1", "nxt3", "nxt9", "nxt27")
        ],
        "bp",
    )
    print(
        " share still >=thr at k=1,3,9,27:",
        [
            round(float((x[c].drop_nulls() >= thr).mean()), 2)
            for c in ("nxt1", "nxt3", "nxt9", "nxt27")
        ],
    )
    print(
        " share still >=0.0005 at k=9:",
        round(float((x["nxt9"].drop_nulls() >= 0.0005).mean()), 2),
        " share back <=0.0001 at k=9:",
        round(float((x["nxt9"].drop_nulls() <= 0.0001).mean()), 2),
    )
    print(
        " share previous print also >=thr:",
        round(float((x["prev1"].drop_nulls() >= thr).mean()), 2),
    )
    print(
        x.group_by("year")
        .agg(
            n=pl.len(), syms=pl.col("symbol").n_unique(), k1=(pl.col("nxt1") >= thr).mean().round(2)
        )
        .sort("year")
        .to_dicts()
    )
    top = x.group_by("symbol").agg(n=pl.len()).sort("n", descending=True)
    print(
        " top symbols by # extreme prints:",
        top.head(8).to_dicts(),
        " top10 share",
        round(float(top.head(10)["n"].sum() / x.height), 2),
    )
# negative
for thr in (-0.003, -0.01):
    x = f.filter(pl.col("rate") <= thr)
    print(f"\n== persistence after rate <= {thr}: n={x.height}, symbols={x['symbol'].n_unique()}")
    print(
        " share still <=thr at k=1,3,9:",
        [round(float((x[c].drop_nulls() <= thr).mean()), 2) for c in ("nxt1", "nxt3", "nxt9")],
    )
# funding cap: how many extreme prints sit at the +-cap values (0.02,0.03,0.04?)
print(
    "\nshare of prints >=0.003 at |rate|>=0.0199:",
    round(float((f.filter(pl.col("rate") >= 0.003)["rate"] >= 0.0199).mean()), 3),
)


# 2. price after prints, gross (no funding), by regime, tails
def paths(cond, side, lab):
    ev = f.filter(cond)
    rows, last = [], {}
    for r in ev.iter_rows(named=True):
        if r["t"] - last.get(r["symbol"], -1e18) < 3 * 86400:
            continue
        a = perp.get(r["symbol"])
        if a is None:
            continue
        h, o, hi, lo, c, qv = a
        i = np.searchsorted(h, r["t"] + 1)
        if i >= len(h) - 73 or i < 1 or qv[i - 1] < 50_000:
            continue
        last[r["symbol"]] = r["t"]
        row = {
            "symbol": r["symbol"],
            "t_in": int(h[i]),
            "year": r["year"],
            "b7": btc7(r["t"]),
            "dying": bool(
                end[r["symbol"]] < DATA_END - 3 * 86400 and end[r["symbol"]] - r["t"] < 14 * 86400
            ),
        }
        for hold in (24, 72):
            j = i + hold - 1
            row[f"c{hold}"] = c[j] / o[i] - 1
            row[f"mx{hold}"] = hi[i : j + 1].max() / o[i] - 1
            row[f"mn{hold}"] = lo[i : j + 1].min() / o[i] - 1
        # price run-up into the print
        k = max(i - 24, 0)
        row["pre24"] = o[i] / o[k] - 1
        row["t_out"] = int(h[i + 71]) + 3600
        rows.append(row)
    d = pl.DataFrame(rows)
    fp = funding_paid(
        np.array([spot_name(s) for s in d["symbol"].to_list()]),
        d["t_in"].to_numpy(),
        d["t_out"].to_numpy(),
        np.full(d.height, side),
    )
    d = d.with_columns(fund72=pl.Series(fp))
    print(
        f"\n######## {lab}: n={d.height} coins={d['symbol'].n_unique()} dying(<14d to delist)={d['dying'].mean():.3f}"  # noqa: E501
    )
    print(
        f" mean pre-24h move into print {d['pre24'].mean() * 100:+.1f}% med {d['pre24'].median() * 100:+.1f}%"  # noqa: E501
    )
    for hold in (24, 72):
        r = (d[f"c{hold}"] * -side * -1) if False else d[f"c{hold}"]
        # price pnl for side: short side=-1 -> -c ; long side=+1 -> +c
        pnl = r * side
        adv = d[f"mx{hold}"] if side < 0 else -d[f"mn{hold}"]
        print(
            f" {hold}h: price pnl mean {pnl.mean() * 1e4:+.0f} med {pnl.median() * 1e4:+.0f}bp  win {float((pnl > 0).mean()):.2f}  p5 {pnl.quantile(0.05) * 1e4:+.0f} p95 {pnl.quantile(0.95) * 1e4:+.0f} | adverse excursion >=+30%: {float((adv >= 0.3).mean()):.3f} >=+100%: {float((adv >= 0.99).mean()):.4f}"  # noqa: E501
        )
    print(
        f" funding received/paid over 72h (side-signed paid; neg=received): mean {d['fund72'].mean() * 1e4:+.0f}bp med {d['fund72'].median() * 1e4:+.0f}bp"  # noqa: E501
    )
    d = d.with_columns(
        pnl72=pl.col("c72") * side, net72=pl.col("c72") * side - 0.003 - pl.col("fund72")
    )
    print(
        " net72 by year:",
        d.group_by("year")
        .agg(
            n=pl.len(),
            price=(pl.col("pnl72").mean() * 1e4).round(),
            fund=(-pl.col("fund72").mean() * 1e4).round(),
            net=(pl.col("net72").mean() * 1e4).round(),
            med=(pl.col("net72").median() * 1e4).round(),
        )
        .sort("year")
        .to_dicts(),
    )
    d = d.with_columns(
        reg=pl.when(pl.col("b7") < -0.05)
        .then(pl.lit("btc7<-5%"))
        .when(pl.col("b7") > 0.05)
        .then(pl.lit("btc7>+5%"))
        .otherwise(pl.lit("btc7 flat"))
    )
    print(
        " by BTC 7d regime:",
        d.group_by("reg")
        .agg(
            n=pl.len(),
            price=(pl.col("pnl72").mean() * 1e4).round(),
            net=(pl.col("net72").mean() * 1e4).round(),
            med=(pl.col("net72").median() * 1e4).round(),
        )
        .sort("reg")
        .to_dicts(),
    )
    d = d.with_columns(
        pr=pl.when(pl.col("pre24") > 0.1)
        .then(pl.lit("ran up>10%"))
        .when(pl.col("pre24") < -0.1)
        .then(pl.lit("fell>10%"))
        .otherwise(pl.lit("flat"))
    )
    print(
        " by coin 24h move into print:",
        d.group_by("pr")
        .agg(
            n=pl.len(),
            price=(pl.col("pnl72").mean() * 1e4).round(),
            net=(pl.col("net72").mean() * 1e4).round(),
        )
        .sort("pr")
        .to_dicts(),
    )
    print(
        " dying vs not:",
        d.group_by("dying")
        .agg(
            n=pl.len(),
            net=(pl.col("net72").mean() * 1e4).round(),
            med=(pl.col("net72").median() * 1e4).round(),
        )
        .to_dicts(),
    )
    # clustering by day
    dd = (
        d.with_columns(day=pl.from_epoch("t_in").dt.date())
        .group_by("day")
        .agg(n=pl.len())
        .sort("n", descending=True)
    )
    print(
        " busiest days (events):",
        dd.head(5).to_dicts(),
        " top10 days share",
        round(float(dd.head(10)["n"].sum() / d.height), 2),
        "of",
        dd.height,
        "days",
    )
    d.write_parquet(
        OWN / f"w5_fund_{lab.split()[0]}_{lab.split()[-1].strip('%+-').replace('.', '')}.parquet"
    ) if False else None


paths(pl.col("rate") >= 0.001, -1, "short after +0.10%")
paths(pl.col("rate") >= 0.003, -1, "short after +0.30%")
paths(pl.col("rate") <= -0.003, 1, "long after -0.30%")
paths(pl.col("rate") <= -0.01, 1, "long after -1.00%")
