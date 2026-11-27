"""w2 part D: 10-10 candidate cohort: how a 25% stop would have filled on 1m data; perp-path leverage check; day-clustering numbers."""  # noqa: E501

import datetime as dt

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, simulate_yearly

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(200)
K = "/Users/mike/code/crypto-moonshot/data/klines/spot/interval=1m/symbol={}/2025-10.parquet"
m = meta()
pct = scores("day5")
yr = m["year"].to_numpy()
idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.995))
c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
Y = (2022, 2023, 2024, 2025, 2026)
ball = simulate_yearly(
    c,
    Rule(hold_min=4320, tp=0.2, slots=10000, liq_frac=None, cost=0.003),
    start_equity=1e9,
    years=Y,
)
t = ball.trades.with_columns(day=pl.col("open_time").dt.date())
d = t.filter(pl.col("day") == dt.date(2025, 10, 10)).sort("open_time")
print("10-10 candidates", d.height, "by entry-time (5m open_time) counts:")
print(
    d.group_by(pl.col("open_time").dt.strftime("%H:%M").alias("hm"))
    .agg(n=pl.len(), mean_adv=pl.col("adv").mean(), mean_net=pl.col("net").mean())
    .sort("hm")
)
print(
    "10-10 cohort: mean 1x net",
    d["net"].mean(),
    "TP share",
    (d["kind"] == 2).mean(),
    "share adv>=.33",
    (d["adv"] >= 0.33).mean(),
    "median adv",
    d["adv"].median(),
)
# entries before 21:10 (pre-wick) vs after
pre = d.filter(pl.col("open_time") < dt.datetime(2025, 10, 10, 21, 10, tzinfo=dt.UTC))
post = d.filter(pl.col("open_time") >= dt.datetime(2025, 10, 10, 21, 10, tzinfo=dt.UTC))
print(
    "entered before 21:10 (open_time):",
    pre.height,
    "mean adv",
    pre["adv"].mean(),
    "mean net",
    pre["net"].mean(),
    "| after:",
    post.height,
    post["adv"].mean(),
    post["net"].mean(),
)
for L in (1.25, 1.5, 2, 3):
    lv = 1 / L - 0.01
    print(
        f"lev {L}: liq level {lv:.1%}; share of 10-10 candidates with adv>=level {(d['adv'] >= lv).mean():.0%}; of other-day candidates {(t.filter(pl.col('day') != dt.date(2025, 10, 10))['adv'] >= lv).mean():.1%}"  # noqa: E501
    )


# 1m stop fills
def stop_fill(sym, ot, lvl):
    try:
        w = pl.read_parquet(
            K.format(sym), columns=["open_time", "open", "high", "low", "close"]
        ).sort("open_time")
    except Exception:
        return None
    t0 = ot + dt.timedelta(minutes=5)
    w = w.filter((pl.col("open_time") >= t0) & (pl.col("open_time") < t0 + dt.timedelta(hours=6)))
    if w.height < 300:
        return None
    px = float(w["open"][0])
    lo = w["low"].to_numpy() / px - 1
    op = w["open"].to_numpy() / px - 1
    j = np.flatnonzero(lo <= -lvl)
    if not len(j):
        return (0.0, 0, 0.0)
    j = j[0]
    fill = min(-lvl, op[j])  # gap through the stop fills at the bar open
    lo[j - 1] if j > 0 else 0
    return (fill, j, op[j] - (-lvl))


res = []
for lvl in (0.25, 0.15):
    out = []
    for s, ot, adv, net in zip(d["symbol"], d["open_time"], d["adv"], d["net"], strict=False):
        r = stop_fill(s, ot, lvl)
        if r is None:
            continue
        out.append((s, adv, net, *r))
    obs = pl.DataFrame(
        out, schema=["sym", "adv", "net72", "fill", "min_to_stop", "gap_below_level"], orient="row"
    )
    st = obs.filter(pl.col("fill") < 0)
    print(
        f"== stop {lvl:.0%} on 1m data: coverage {obs.height}/{d.height}; stopped {st.height}; mean fill {st['fill'].mean():.3f}; fill worse than -{lvl + 0.10:.0%}: {(st['fill'] < -(lvl + 0.10)).sum()}; mean gap past level {st['gap_below_level'].mean():.3f}; median minute-of-stop {st['min_to_stop'].median()}"  # noqa: E501
    )
    if len(st):
        for L in (1, 2, 3):
            liq = 1 / L - 0.01
            rr = np.where(-st["fill"].to_numpy() >= liq, -1.0, L * (st["fill"].to_numpy() - 0.003))
            print(
                f"   lev {L}: mean return on margin of stopped positions {rr.mean():+.2f}; stopped-out would have ended (72h-hold, 1x) mean {st['net72'].mean():+.3f}, TP-hit share-of-stopped {(st['net72'] > 0.19).mean():.0%}"  # noqa: E501
            )
    if lvl == 0.25:
        Ostop = obs
# first-10 slots exposures: sum of adv
f10 = d.sort("open_time").head(10)
print(
    "first 10 taken: mean adv",
    f10["adv"].mean(),
    "-> concurrent mtm at 1x gross 1.0 ~",
    f10["adv"].mean(),
    "; at 1.5x:",
    1.5 * f10["adv"].mean(),
)
# perp path leverage check
print("== perp paths (source=perp), q>=.995, 10 slots")
for lev in (1, 2, 3):
    b = simulate_yearly(
        c, Rule(hold_min=4320, tp=0.2, slots=10, cost=0.003, lev=lev), years=Y, source="perp"
    )
    print(
        f"lev {lev}:",
        " ".join(
            f"{r['year']}:{r['ret'] * 100:+.0f}%(n{r['trades']},liq{r['liq']})"
            for r in b.years.iter_rows(named=True)
        ),
    )
bp = simulate_yearly(
    c,
    Rule(hold_min=4320, tp=0.2, slots=10000, liq_frac=None, cost=0.003),
    start_equity=1e9,
    years=(2025,),
    source="perp",
).trades
bs = t.filter(pl.col("year") == 2025)
print(
    "perp coverage 2025 cands",
    bp.height,
    "of",
    bs.height,
    "; perp mean adv",
    bp["adv"].mean(),
    " vs spot (same 2025):",
    bs["adv"].mean(),
    "; perp share adv>=.33",
    (bp["adv"] >= 0.33).mean(),
    "spot",
    (bs["adv"] >= 0.33).mean(),
)
x = bp.join(bs.select("row", spot_adv="adv"), on="row").with_columns(
    day=pl.col("open_time").dt.date()
)
print(
    "on 10-10 perp vs spot adv (matched rows), mean",
    x.filter(pl.col("day") == dt.date(2025, 10, 10))["adv"].mean(),
    x.filter(pl.col("day") == dt.date(2025, 10, 10))["spot_adv"].mean(),
)
# day clustering of the 162 booked trades: distinct days
tb = pl.read_parquet("/tmp/w2_trades.parquet").with_columns(day=pl.col("open_time").dt.date())
print(
    "book trades",
    tb.height,
    "on distinct days",
    tb["day"].n_unique(),
    "; trades with adv>=.33:",
    (tb["adv"] >= 0.33).sum(),
    "on",
    tb.filter(pl.col("adv") >= 0.33)["day"].n_unique(),
    "days",
)
g = (
    tb.group_by("day")
    .agg(n=pl.len(), pnl=pl.col("pnl").sum(), mean=pl.col("net").mean())
    .sort("pnl")
)
print(g.head(5))
print(
    "share of 2025 candidates (all) on 10-10:", d.height / t.filter(pl.col("year") == 2025).height
)
