"""w2: why leverage fails on capitulation longs. Part A: MAE vs final return, clustering by day."""

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, simulate_yearly

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(200)
m = meta()
pct = scores("day5")
yr = m["year"].to_numpy()
idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.995))
c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
Y = (2022, 2023, 2024, 2025, 2026)
b = simulate_yearly(c, Rule(hold_min=4320, tp=0.2, slots=10, cost=0.003, lev=1), years=Y)
t = b.trades.with_columns(
    day=pl.col("open_time").dt.date(),
    r4=pl.Series(m["ret_4h"].to_numpy()[b.trades["row"].to_numpy()]),
)
print("taken trades", t.height, "mean bp", b.summary["bp"])
print("all candidates", c.height)
print(b.years)
print("== MAE (max adverse to exit, 1x) by year; share >=x and share of those finishing profitable")
for y in [*list(Y), 0]:
    s = t if y == 0 else t.filter(pl.col("year") == y)
    a = s["adv"].to_numpy()
    n = s["net"].to_numpy()
    row = [
        f"{y or 'all'} n={len(a)} mean net {n.mean() * 100:+.1f}% med adv {np.median(a) * 100:.0f}%"
    ]
    for th in (0.2, 0.33, 0.49, 0.5, 0.6):
        k = a >= th
        row.append(f">={th:.0%}:{k.sum()}({(n[k] > 0).sum()} prof)")
    print(" | ".join(row))
print(
    "== adv deciles all",
    np.round(np.percentile(t["adv"].to_numpy(), [10, 25, 50, 75, 90, 95, 99]) * 100, 1),
)
print("== final net by adv bucket (all years)")
a = t["adv"].to_numpy()
n = t["net"].to_numpy()
for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.33), (0.33, 0.5), (0.5, 1.5)]:
    k = (a >= lo) & (a < hi)
    print(
        f"adv {lo:.0%}-{hi:.0%}: n={k.sum()} mean net {n[k].mean() * 100:+.1f}% win {(n[k] > 0).mean():.0%} kind tp share {(t['kind'].to_numpy()[k] == 2).mean():.0%}"  # noqa: E501
    )
print("== days with the most trades with adv>=0.32 (3x liq level), all years")
x = (
    t.filter(pl.col("adv") >= 0.32)
    .group_by("day")
    .agg(
        n=pl.len(),
        syms=pl.col("symbol").unique().sort().str.join(","),
        mean_net=pl.col("net").mean(),
        mx=pl.col("adv").max(),
    )
    .sort("n", descending=True)
)
print(x.head(15))
print("trades/day on 2025-10-10 vs others")
print(t.group_by("day").len().sort("len", descending=True).head(8))
d = t.filter(pl.col("day") == pl.date(2025, 10, 10))
print(
    d.select("symbol", "open_time", "stake", "adv", "ret", "net", "kind", "hold_min", "r4").sort(
        "open_time"
    )
)
d2 = t.filter(pl.col("day").is_between(pl.date(2025, 10, 9), pl.date(2025, 10, 12)))
print(d2.height)
t.write_parquet("/tmp/w2_trades.parquet")
