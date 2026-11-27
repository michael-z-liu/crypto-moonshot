"""w2 part B: 2025-10-10 anatomy from 1m candles, majors wicks, all-candidate MAE, leverage survival."""  # noqa: E501

import datetime as dt

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, paths, simulate_yearly

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_cols(20)
K = "/Users/mike/code/crypto-moonshot/data/klines/spot/interval=1m/symbol={}/2025-10.parquet"
UTC = "UTC"


def T(h, mi, d=10):
    return pl.datetime(2025, 10, d, h, mi, time_zone=UTC)


m = meta()
pct = scores("day5")
yr = m["year"].to_numpy()
idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.995))
c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
Y = (2022, 2023, 2024, 2025, 2026)
# ---- all candidates (no slot / liquidity limit)
ball = simulate_yearly(
    c,
    Rule(hold_min=4320, tp=0.2, slots=10000, liq_frac=None, cost=0.003),
    start_equity=1e9,
    years=Y,
)
t = ball.trades.with_columns(day=pl.col("open_time").dt.date())
print(
    "all candidates traded (one per coin at a time):",
    t.height,
    "mean net",
    round(t["net"].mean() * 100, 1),
)
a = t["adv"].to_numpy()
n = t["net"].to_numpy()
print(
    "MAE share >=33%:",
    round((a >= 0.33).mean() * 100, 1),
    ">=50%:",
    round((a >= 0.5).mean() * 100, 1),
    "| of >=33% profitable:",
    round((n[a >= 0.33] > 0).mean() * 100),
    "% ; mean net of them",
    round(n[a >= 0.33].mean() * 100, 1),
)
print(
    t.group_by("year")
    .agg(
        n=pl.len(),
        mean=pl.col("net").mean(),
        p33=(pl.col("adv") >= 0.33).sum(),
        p50=(pl.col("adv") >= 0.5).sum(),
        maxadv=pl.col("adv").max(),
    )
    .sort("year")
)
# profitable trades with deep MAE
pr = t.filter((pl.col("net") > 0) & (pl.col("adv") >= 0.33)).select(
    "symbol", "open_time", "adv", "net", "kind"
)
print("profitable but MAE>=33%:", pr.height, "of", int((a >= 0.33).sum()))
# days
print("all-cand days with >=1 trade adv>=0.33:")
print(
    t.filter(pl.col("adv") >= 0.33)
    .group_by("day")
    .agg(n=pl.len(), nday=pl.len())
    .sort("n", descending=True)
    .head(12)
)
print("trades per day, share of days: ", t.group_by("day").len()["len"].describe())
# worst adverse by year after removing named crash days
for y in Y:
    s = (
        t.filter(pl.col("year") == y)
        .sort("adv", descending=True)
        .head(3)
        .select("symbol", "day", "adv", "net")
    )
    print(y, s.to_dicts())
# survival leverage: per year, max lev s.t. no trade liquidated: 1/(maxadv+mm)
print("== survival leverage = 1/(max adv + 1% mm), per year (all candidates)")
for y in Y:
    mx = t.filter(pl.col("year") == y)["adv"].max()
    print(y, f"max adv {mx:.3f} -> lev {1 / (mx + 0.01):.2f}")
mx = t["adv"].max()
print("all", mx, 1 / (mx + 0.01))
t2 = t.filter(
    ~pl.col("day").is_in(
        [dt.date(2022, 5, 11), dt.date(2022, 5, 12), dt.date(2022, 11, 8), dt.date(2025, 10, 10)]
    )
)
print(
    "excluding LUNA/FTT/10-10: max adv",
    t2["adv"].max(),
    "-> lev",
    1 / (t2["adv"].max() + 0.01),
    "; q99 adv",
    np.percentile(t2["adv"], 99),
)
# ---- leverage runs on the 10-slot book
print("== book at lev (10 slots, liquidity cap 0.5%, cost .003)")
for lev in (1, 1.25, 1.5, 2, 3):
    b = simulate_yearly(c, Rule(hold_min=4320, tp=0.2, slots=10, cost=0.003, lev=lev), years=Y)
    tr = b.trades
    print(
        f"lev {lev}: ",
        " ".join(
            f"{r['year']}:{r['ret'] * 100:+.0f}%(liq{r['liq']})"
            for r in b.years.iter_rows(named=True)
        ),
        f"| mean net {b.summary['bp'] / 100:+.1f}%",
    )
    if lev in (2, 3):
        t25 = tr.filter(pl.col("year") == 2025)
        print(
            "   2025 liq by day:",
            t25.filter(pl.col("kind") == 3)
            .group_by(pl.col("open_time").dt.date().alias("d"))
            .len()
            .sort("d")
            .to_dicts(),
        )
        print(
            "   2025 sum net*stake share on 10-10:",
            float(t25.filter(pl.col("open_time").dt.date() == pl.date(2025, 10, 10))["pnl"].sum()),
            "of total",
            float(t25["pnl"].sum()),
        )
        # equity after day
        print(
            "   2025 pnl by trade (stake, net):",
            [
                (round(s), round(x, 2))
                for s, x in zip(t25.sort("t")["stake"], t25.sort("t")["net"], strict=False)
            ][:25],
        )
# stop 25% with lev
print("== stop 25% with leverage (stop_slip 1%)")
for lev in (2, 3):
    b = simulate_yearly(
        c,
        Rule(hold_min=4320, tp=0.2, sl=0.25, slots=10, cost=0.003, lev=lev, stop_slip=0.01),
        years=Y,
    )
    tr = b.trades
    t25 = tr.filter(pl.col("year") == 2025)
    print(
        f"lev {lev} sl .25:",
        " ".join(
            f"{r['year']}:{r['ret'] * 100:+.0f}%(liq{r['liq']})"
            for r in b.years.iter_rows(named=True)
        ),
    )
    print("   kinds 2025", t25.group_by("kind").len().sort("kind").to_dicts())
# gap: first step where adv>=.25 : adv at that step vs previous step's adv, for all candidates
p = paths(t["row"].to_numpy())
adv = np.nan_to_num(-p[:, 1], nan=0)
gaps = []
for i in range(len(t)):
    j = np.flatnonzero(adv[i] >= 0.25)
    if len(j) and j[0] <= 95:  # in the 5m part
        prev = adv[i, j[0] - 1] if j[0] > 0 else 0
        gaps.append((t["symbol"][i], str(t["day"][i]), prev, adv[i, j[0]], j[0]))
print("trades crossing 25% adverse inside first 8h:", len(gaps))
jump = [g for g in gaps if g[3] >= 0.33]
print(
    " of which the same 5m bar's low is >=33% (gap through 3x liq level):",
    len(jump),
    " with previous bar adv <10%:",
    sum(1 for g in gaps if g[2] < 0.10),
)
# concurrent adverse on 10-10 at 1x
d = t.filter(pl.col("day") == pl.date(2025, 10, 10))
print("10-10 candidates", d.height, "mean adv", d["adv"].mean())
# probability simulation: expected log growth
