"""W5b: why the BTC-calm threshold matters for idiosyncratic crash shorts (sleeve C)."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN, scores
from futures_ls import perp_window
from sim import Rule, simulate

pw = perp_window()
d = pl.read_parquet(OWN / "feats.parquet").filter(
    pl.col("perp")
    & (pl.col("qv_base") >= 20_000)
    & (pl.col("open_time").dt.year() >= 2022)
    & (pl.col("ret_4h") <= -0.15)
)
pct = scores("day5")
d = d.with_columns(day5=pl.Series(pct[d["row"].to_numpy()]), ab=pl.col("btc_ret_4h").abs())
d = d.filter(pl.col("day5") <= 0.5)
print("candidates r4<=-15% & cnn<=0.5, perp:", d.height)
b = simulate(
    d.select("row", side=pl.lit(-1), prio=-pl.col("ret_4h")),
    Rule(hold_min=1440, slots=100_000, cost=0.003, sl=1.0, liq_frac=None),
    start_equity=1e12,
    perp_funding=True,
    source="perp",
)
t = b.trades
print(t.columns)
t = (
    t.join(
        d.select(
            "row",
            "ab",
            "btc_ret_4h",
            "mkt_r4h",
            "mkt_r24h",
            "eth_r4h",
            "spike_share",
            "spike_share1h",
            "ret_4h",
            "ret_24h",
            "listed_days",
            "r7d",
            "day",
            "year",
            "qv_base",
            "vol_mult",
            "btc_vol_mult",
        ),
        on="row",
        how="left",
        suffix="_d",
    )
    if "row" in t.columns
    else t
)
end_t = {s: v[1] for s, v in pw.items()}
t = t.with_columns(
    delist=pl.Series(
        [
            (end_t.get(s, 9e18) < 1.7727e9 and end_t.get(s, 9e18) - ti < 30 * 86400)
            for s, ti in zip(t["symbol"].to_list(), t["t"].to_list(), strict=False)
        ]
    ),
    new=pl.col("listed_days") < 90,
)
bands = [(0, 0.005), (0.005, 0.01), (0.01, 0.015), (0.015, 0.02), (0.02, 0.03), (0.03, 1)]
rows = []
for lo, hi in bands:
    x = (
        t.filter((pl.col("ab") > lo) & (pl.col("ab") <= hi))
        if lo > 0
        else t.filter(pl.col("ab") <= hi)
    )
    n = x.height
    rows.append(
        dict(
            band=f"{lo * 100:.1f}-{hi * 100:.1f}%",
            n=n,
            days=x["day"].n_unique(),
            bp=round(float(x["net"].mean()) * 1e4),
            med=round(float(x["net"].median()) * 1e4),
            win=round(float((x["net"] > 0).mean()), 2),
            mkt4=round(float(x["mkt_r4h"].mean()) * 100, 2),
            btc4=round(float(x["btc_ret_4h"].mean()) * 100, 2),
            spk=round(float(x["spike_share"].mean()), 3),
            r4=round(float(x["ret_4h"].mean()) * 100, 1),
            r24=round(float(x["ret_24h"].mean()) * 100, 1),
            young=round(float(x["new"].mean()), 2),
            delist=round(float(x["delist"].mean()), 3),
            p5=round(float(x["net"].quantile(0.05)) * 1e4),
            p95=round(float(x["net"].quantile(0.95)) * 1e4),
        )
    )
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(30)
print(pl.DataFrame(rows))
# market-driven vs coin-specific: coin move beyond market
t = t.with_columns(rel=pl.col("ret_4h") - pl.col("mkt_r4h"))
x1, x2 = t.filter(pl.col("ab") <= 0.01), t.filter((pl.col("ab") > 0.01) & (pl.col("ab") <= 0.02))
for lab, x in (("calm<=1%", x1), ("1-2%", x2)):
    print(
        lab,
        "mkt_r4h<=-2%:",
        round(float((x["mkt_r4h"] <= -0.02).mean()), 2),
        " btc4<0 share",
        round(float((x["btc_ret_4h"] < 0).mean()), 2),
        "net by btc direction:",
        x.group_by(pl.col("btc_ret_4h") < 0)
        .agg(n=pl.len(), bp=(pl.col("net").mean() * 1e4).round())
        .to_dicts(),
    )
    print(
        "   by year",
        x.group_by("year")
        .agg(
            n=pl.len(),
            bp=(pl.col("net").mean() * 1e4).round(),
            med=(pl.col("net").median() * 1e4).round(),
        )
        .sort("year")
        .to_dicts(),
    )
    print(
        "   by mkt_r4h tercile",
        x.with_columns(
            q=pl.col("mkt_r4h").qcut(3, labels=["mkt down", "mid", "mkt up"], allow_duplicates=True)
        )
        .group_by("q")
        .agg(
            n=pl.len(),
            bp=(pl.col("net").mean() * 1e4).round(),
            med=(pl.col("net").median() * 1e4).round(),
        )
        .sort("q")
        .to_dicts(),
    )
    dd = (
        x.group_by("day")
        .agg(n=pl.len(), bp=(pl.col("net").mean() * 1e4))
        .sort("n", descending=True)
    )
    print("   busiest days", dd.head(4).to_dicts())
# BTC move AFTER entry: does the loss come from BTC bouncing in the band?  Use btc hourly path from spot.  # noqa: E501
import os  # noqa: E402
import sys  # noqa: E402

os.environ["MOONSHOT_UNIVERSE"] = "all"
from common_ls import CACHE_ALL  # noqa: E402

sys.path.insert(0, str(CACHE_ALL.parent))
from common import load  # noqa: E402

bt = (
    load("BTCUSDT")
    .group_by_dynamic("open_time", every="1h")
    .agg(c=pl.col("close").last())
    .sort("open_time")
)
bh, bc = bt["open_time"].dt.epoch("s").to_numpy(), bt["c"].to_numpy()


def btc_fwd(ti, h=24):
    k = np.searchsorted(bh, ti, side="right") - 1
    return bc[min(k + h, len(bc) - 1)] / bc[k] - 1


t = t.with_columns(bf=pl.Series([btc_fwd(ti) for ti in t["t"].to_list()]))
print("\nBTC forward 24h after entry, by band:")
for lab, x in (
    ("<=1%", x1.join(t.select("row", "bf"), on="row")),
    ("1-2%", x2.join(t.select("row", "bf"), on="row")),
):
    print(
        lab,
        "mean BTC fwd24",
        round(float(x["bf"].mean()) * 100, 2),
        "corr(net,btcfwd)",
        round(float(np.corrcoef(x["net"].to_numpy(), x["bf"].to_numpy())[0, 1]), 2),
        "coin-leg (net + btc beta1 hedge) mean bp",
        round(float((x["net"] + x["bf"]).mean()) * 1e4),
    )
# worst trades by band
print(
    "\nworst 6 trades 1-2% band:",
    x2.sort("net")
    .head(6)
    .select("symbol", "day", "net", "ret_4h", "btc_ret_4h", "delist", "new")
    .to_dicts(),
)
print(
    "worst 6 trades <=1% band:",
    x1.sort("net")
    .head(6)
    .select("symbol", "day", "net", "ret_4h", "btc_ret_4h", "delist", "new")
    .to_dicts(),
)
# bootstrap noise: difference <=1% vs 1-2%
rng = np.random.default_rng(0)
a, c = x1["net"].to_numpy(), x2["net"].to_numpy()
diff = [rng.choice(a, len(a)).mean() - rng.choice(c, len(c)).mean() for _ in range(5000)]
print(
    f"\nbootstrap diff (<=1% minus 1-2%) mean {np.mean(diff) * 1e4:+.0f}bp  p5 {np.percentile(diff, 5) * 1e4:+.0f} p95 {np.percentile(diff, 95) * 1e4:+.0f}  n {len(a)} vs {len(c)}"  # noqa: E501
)
