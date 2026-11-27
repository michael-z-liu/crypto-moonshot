"""W5f: 2025-10-11 pump rows; where the book drawdown sits; bad-week independence check; 1-2% calm band without SRM."""  # noqa: E501

from __future__ import annotations

import numpy as np
import polars as pl
from e25_combined import (
    book,
    sleeve_crash_short,
    sleeve_listing_weak,
    sleeve_long,
    sleeve_pump,
    sleeve_pump_regime,
)

P = sleeve_pump().with_columns(day=pl.from_epoch("t_in").dt.date())
x = P.filter(pl.col("day") == pl.date(2025, 10, 11))
print(
    "2025-10-11 pump rows:",
    x.height,
    "mean net",
    round(float(x["net"].mean()), 2),
    "n<=-30%",
    int((x["net"] <= -0.3).sum()),
    "n liq",
    int((x["net"] <= -0.99).sum()),
    "n winners",
    int((x["net"] > 0).sum()),
)
print(
    x.sort("net").select("symbol", pl.col("net").round(2), pl.from_epoch("t_in").alias("t")).head(8)
)
Q = sleeve_pump_regime()
print(
    "regime Q: n",
    Q.height,
    " rows on 2025-10-11:",
    Q.filter(pl.from_epoch("t_in").dt.date() == pl.date(2025, 10, 11)).height,
)
L, C, M = sleeve_long(), sleeve_crash_short(), sleeve_listing_weak()
tr = pl.concat([L, Q, C, M], how="vertical_relaxed")
out: list = []
b = book(tr, 10, trades_out=out)
t = pl.DataFrame(out).with_columns(day=pl.from_epoch("t_out").dt.date())
print(
    "book 2025 pnl by sleeve (sum pnl $ on $10k start):",
    t.filter(pl.col("year") == 2025)
    .group_by("sleeve")
    .agg(pnl=pl.col("pnl").sum().round(), n=pl.len(), worst=pl.col("pnl").min().round())
    .to_dicts(),
)
for y in (2025, 2026):
    ty = t.filter(pl.col("year") == y).sort("t_out").with_columns(cum=pl.col("pnl").cum_sum())
    ty = ty.with_columns(peak=pl.col("cum").cum_max(), dd=pl.col("cum") - pl.col("cum").cum_max())
    w = ty.sort("dd").head(1)
    print(
        y,
        "worst point in book:",
        w.select("day", "dd").to_dicts(),
        "worst 5 trades by pnl:",
        ty.sort("pnl").head(5).select("day", "sleeve", pl.col("pnl").round()).to_dicts(),
    )
# weekly independence of bad pump trades
bad = P.filter(pl.col("net") <= -0.3).with_columns(w=pl.col("t_in") // (7 * 86400))
nw = P.with_columns(w=pl.col("t_in") // (7 * 86400))["w"].n_unique()
lam = bad.height / nw
exp2 = nw * (1 - np.exp(-lam) * (1 + lam))
print(
    f"bad-trade weeks: poisson lambda {lam:.2f} over {nw} active weeks -> expected weeks with >=2: {exp2:.1f}; observed {int((bad.group_by('w').agg(n=pl.len())['n'] >= 2).sum())}"  # noqa: E501
)
# bad per day excluding the 2025-10-11 date
print(
    "trades <=-30% excluding 2025-10-11:",
    int((P.filter(pl.col("day") != pl.date(2025, 10, 11))["net"] <= -0.3).sum()),
    "mean bp ex that day",
    round(float(P.filter(pl.col("day") != pl.date(2025, 10, 11))["net"].mean()) * 1e4),
)
# C band without SRM
# C sleeve band re-check without the -100% trade
from common_ls import OWN, scores  # noqa: E402
from sim import Rule, simulate  # noqa: E402

d = pl.read_parquet(OWN / "feats.parquet").filter(
    pl.col("perp")
    & (pl.col("qv_base") >= 20_000)
    & (pl.col("open_time").dt.year() >= 2022)
    & (pl.col("ret_4h") <= -0.15)
)
pct = scores("day5")
d = d.filter(pl.Series(pct[d["row"].to_numpy()] <= 0.5)).with_columns(ab=pl.col("btc_ret_4h").abs())
tt = simulate(
    d.select("row", side=pl.lit(-1), prio=-pl.col("ret_4h")),
    Rule(hold_min=1440, slots=100_000, cost=0.003, sl=1.0, liq_frac=None),
    start_equity=1e12,
    perp_funding=True,
    source="perp",
).trades.join(d.select("row", "ab"), on="row")
for lab, f in (
    ("<=1%", pl.col("ab") <= 0.01),
    ("1-2%", (pl.col("ab") > 0.01) & (pl.col("ab") <= 0.02)),
    ("<=2%", pl.col("ab") <= 0.02),
    ("2-5%", (pl.col("ab") > 0.02) & (pl.col("ab") <= 0.05)),
):
    y = tt.filter(f)
    z = y.filter(pl.col("net") > -0.99)
    print(
        f"C band {lab:5s} n={y.height:>3} mean {y['net'].mean() * 1e4:+5.0f}bp med {y['net'].median() * 1e4:+5.0f} | trades <=-50%: {int((y['net'] <= -0.5).sum())} | mean without the -100% trades {z['net'].mean() * 1e4:+5.0f}bp (n={z.height}) | se {y['net'].std() / np.sqrt(y.height) * 1e4:.0f}bp"  # noqa: E501
    )
