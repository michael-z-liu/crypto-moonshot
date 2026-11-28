"""W3-B: P&L by day, selection decomposition (day lift vs within-day), exit-shape diagnostics."""

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import STEP_END_MIN, Rule, exits, paths, simulate_yearly

Y = (2022, 2023, 2024, 2025, 2026)
d = pl.read_parquet("/tmp/w3_tab.parquet").with_columns(pl.col("day").cast(pl.Date))
m = meta()
yr = m["year"].to_numpy()
# ---------- selection decomposition
print("== selection decomposition (rows 2022+, top 0.5% by score): lift = day-level + within-day")
dmean = d.with_columns(
    dm_o=pl.col("o72").mean().over("day"),
    dm_y=pl.col("y1h").clip(-0.2, 0.2).mean().over("day"),
    cnt=pl.len().over("day"),
)
base_o = d["o72"].mean()
print(f" all-rows mean o72 {base_o * 1e4:+.0f}bp")
for sc in ("day5", "tp20day", "day10b", "day_tcn"):
    s = dmean.filter(pl.col(sc) >= 0.995)
    top = s.group_by("day").agg(n=pl.len()).sort("n", descending=True)
    sh = top["n"].head(5).sum() / s.height
    print(
        f" {sc:9s} n={s.height} days={s['day'].n_unique()} top5-days share={sh:.2f} mean o72 {s['o72'].mean() * 1e4:+.0f}bp "  # noqa: E501
        f"= day-mean part {(s['dm_o'].mean() - base_o) * 1e4:+.0f} + within-day {(s['o72'] - s['dm_o']).mean() * 1e4:+.0f} (+base {base_o * 1e4:+.0f}); "  # noqa: E501
        f"y1h clip mean {s['y1h'].clip(-0.2, 0.2).mean() * 1e4:+.0f} hit20 {(s['t_tp'] >= 0).mean():.2f}"  # noqa: E501
    )
print(" same, restricted to the 11 biggest days of day5 picks")
big = (
    dmean.filter(pl.col("day5") >= 0.995)
    .group_by("day")
    .agg(n=pl.len())
    .sort("n", descending=True)
    .head(11)["day"]
    .to_list()
)
for sc in ("day5", "tp20day"):
    s = dmean.filter((pl.col(sc) >= 0.995) & pl.col("day").is_in(big))
    print(
        f"  {sc:8s} on those days n={s.height} o72 {s['o72'].mean() * 1e4:+.0f} hit20 {(s['t_tp'] >= 0).mean():.2f}; all rows on those days o72 {dmean.filter(pl.col('day').is_in(big))['o72'].mean() * 1e4:+.0f}"  # noqa: E501
    )
# ---------- book
pct = scores("day5")
idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.995))
c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
rule = Rule(hold_min=4320, tp=0.2, slots=10, cost=0.004)
b = simulate_yearly(c, rule, years=Y)
t = b.trades.with_columns(day=pl.col("open_time").dt.date())
print(
    f"== book: trades {t.height}, mean net {t['net'].mean() * 1e4:+.0f}bp, win {(t['net'] > 0).mean():.2f}, TP {(t['kind'] == 2).mean():.2f}, candidates {c.height}"  # noqa: E501
)
g = (
    t.group_by("day")
    .agg(
        n=pl.len(),
        net=pl.col("net").mean(),
        pnl=(pl.col("stake") * pl.col("net")).sum(),
        pnl_unit=pl.col("net").sum(),
    )
    .sort("pnl_unit", descending=True)
)
tot = g["pnl_unit"].sum()
print(" share of summed per-trade net by day (days with trades:", g.height, "):")
for k in (1, 3, 5, 11):
    print(f"  top {k} days: {g['pnl_unit'].head(k).sum() / tot:.2f}  trades {g['n'].head(k).sum()}")
print(
    g.head(14)
    .with_columns(net_bp=pl.col("net") * 1e4, share=pl.col("pnl_unit") / tot)
    .select("day", "n", "net_bp", "share")
)
from common_ls import first_per_event  # noqa: E402

fe = first_per_event((yr >= 2022) & np.isfinite(pct) & (pct >= 0.995))
fed = pl.DataFrame({"day": m["day"].to_numpy()[fe]}).group_by("day").agg(ncand=pl.len())
broad = fed.filter(pl.col("ncand") > 30)
print(
    f" first_per_event candidates {len(fe)}; on days with >30 candidates: {broad['ncand'].sum()} over {broad.height} days; other days {fed.height - broad.height} days {len(fe) - broad['ncand'].sum()} cands"  # noqa: E501
)
gg = g.join(fed, on="day", how="left").with_columns(broad=pl.col("ncand") > 30)
for flag in (True, False):
    z = gg.filter(pl.col("broad") == flag)
    print(
        f"  book trades on broad={flag} days: {z.height} days, {z['n'].sum()} trades, mean {z['pnl_unit'].sum() / max(z['n'].sum(), 1) * 1e4:+.0f}bp, share of summed net {z['pnl_unit'].sum() / tot:.2f}"  # noqa: E501
    )
print(
    " per-year trades by day (kind mix): ",
    t.group_by("kind").agg(n=pl.len(), net=pl.col("net").mean() * 1e4).sort("kind").to_dicts(),
)
# day-to-day dispersion: mean net per day across the 26 days
dd = g.filter(pl.col("n") >= 5)
print(
    f" days with >=5 trades: {dd.height}; mean-net by day min {dd['net'].min() * 1e4:+.0f} median {dd['net'].median() * 1e4:+.0f} max {dd['net'].max() * 1e4:+.0f}; days negative {(dd['net'] < 0).sum()}"  # noqa: E501
)
# breadth buckets of crash days
cr = d.filter(pl.col("ret_4h") <= -0.10)
br = cr.group_by("day").agg(
    nb=pl.len(),
    o=pl.col("o72").mean(),
    hit=(pl.col("t_tp") >= 0).mean(),
    btc=pl.col("btc_ret_4h").mean(),
    p5=pl.col("day5").max(),
)
br = br.with_columns(
    bucket=pl.when(pl.col("nb") >= 30)
    .then(pl.lit("d:30+"))
    .when(pl.col("nb") >= 10)
    .then(pl.lit("c:10-29"))
    .when(pl.col("nb") >= 3)
    .then(pl.lit("b:3-9"))
    .otherwise(pl.lit("a:1-2"))
)
print("== crash-row outcome by breadth of the day (crash rows 2022+, liquid ret_4h<=-10%)")
x = cr.join(br.select("day", "bucket"), on="day")
print(
    x.group_by("bucket")
    .agg(
        days=pl.col("day").n_unique(),
        rows=pl.len(),
        o72_bp=pl.col("o72").mean() * 1e4,
        hit20=(pl.col("t_tp") >= 0).mean(),
        y1h_bp=pl.col("y1h").clip(-0.2, 0.2).mean() * 1e4,
        btc4h=pl.col("btc_ret_4h").mean() * 100,
        day5_mean=pl.col("day5").mean(),
    )
    .sort("bucket")
)
print(
    " day5>=.995 rows by bucket:",
    d.filter(pl.col("day5") >= 0.995)
    .join(br.select("day", "bucket"), on="day", how="left")
    .group_by("bucket")
    .agg(rows=pl.len(), o72=pl.col("o72").mean() * 1e4)
    .sort("bucket")
    .to_dicts(),
)
# ---------- exits on candidate rows (no slots)
r = idx
p = paths(r)
hi = np.nan_to_num(p[:, 0], nan=-1)
lo = np.nan_to_num(p[:, 1], nan=0)
cl = np.nan_to_num(p[:, 2], nan=0)
peak = hi.max(1)
print(f"== peak (max high within 72h, rel. entry) of the {len(r)} candidate rows (no slot limit)")
print(
    " quantiles 10/25/50/75/90:",
    [round(float(np.quantile(peak, q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
)
for x in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0):
    print(f"  P(peak>={x:.2f}) = {(peak >= x).mean():.3f}")
print(
    " close at 72h quantiles:",
    [round(float(np.quantile(cl[:, -1], q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
    "mean",
    round(float(cl[:, -1].mean()), 3),
)
print("== fixed TP sweep, 72h, unconstrained mean net (cost .4%) over candidate rows")
for tp in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, None):
    ret, _, k = exits(p, np.ones(len(r)), Rule(hold_min=4320, tp=tp))
    print(f"  tp {tp}: {np.mean(ret - 0.004) * 1e4:+.0f}bp hit {np.mean(k == 2):.2f}")
ret20, te, k20 = exits(p, np.ones(len(r)), Rule(hold_min=4320, tp=0.2))
hit = k20 == 2
print(
    f" hit20 {hit.mean():.3f}; median minutes to TP {np.median(STEP_END_MIN[te[hit]])}; share hit within 1h {np.mean(STEP_END_MIN[te[hit]] <= 60):.2f}, 4h {np.mean(STEP_END_MIN[te[hit]] <= 240):.2f}, 24h {np.mean(STEP_END_MIN[te[hit]] <= 1440):.2f}"  # noqa: E501
)
miss = ~hit
print(
    f" non-hit: mean 72h close {cl[miss, -1].mean() * 1e4:+.0f}bp median {np.median(cl[miss, -1]) * 1e4:+.0f}, share closing <0: {np.mean(cl[miss, -1] < 0):.2f}; mean peak {peak[miss].mean():.3f}"  # noqa: E501
)
print(
    f" hit-rows' loss given miss: net mean {np.mean(ret20[miss] - 0.004) * 1e4:+.0f}bp; win contribution {hit.mean() * (0.2 - 0.004) * 1e4:+.0f}bp"  # noqa: E501
)
# after TP
print("== after +20% (price level 1.2), rows that hit")
steps_after = {"1h": 60, "4h": 240, "24h": 1440, "to 72h": 4320}
idxh = np.flatnonzero(hit)
for lab, dt in steps_after.items():
    vals = []
    for i in idxh:
        t0 = STEP_END_MIN[te[i]]
        tgt = min(t0 + dt, 4320)
        j = int(np.searchsorted(STEP_END_MIN, tgt))
        j = min(j, 159)
        vals.append((1 + cl[i, j]) / 1.2 - 1)
    vals = np.array(vals)
    print(
        f"  close {lab:6s} after TP vs 1.2: mean {vals.mean() * 1e4:+.0f}bp median {np.median(vals) * 1e4:+.0f} P(up) {np.mean(vals > 0):.2f} P(<-10%) {np.mean(vals < -0.1):.2f}"  # noqa: E501
    )
fut = []
dd_ = []
for i in idxh:
    j0 = te[i]
    seg = hi[i, j0:]
    fut.append(seg.max() / 1.2 - 1 + (1.0 / 1.2 - 1) + 0) if False else fut.append(
        (1 + seg.max()) / 1.2 - 1
    )
    dd_.append((1 + lo[i, j0 + 1 :].min()) / 1.2 - 1 if j0 + 1 < 160 else 0.0)
fut = np.array(fut)
dd_ = np.array(dd_)
print(
    f"  further max high after TP: median {np.median(fut):.3f} mean {fut.mean():.3f}; P(+10% more) {np.mean(fut >= 0.1):.2f}, P(+25% more) {np.mean(fut >= 0.25):.2f}; worst low after TP: median {np.median(dd_):.3f}, P(<-10%) {np.mean(dd_ < -0.1):.2f} P(<-20%) {np.mean(dd_ < -0.2):.2f}"  # noqa: E501
)
# bar overshoot at the TP bar
ov = np.array([(1 + cl[i, te[i]]) / 1.2 - 1 for i in idxh])
print(
    f"  TP bar close vs 1.2: mean {ov.mean():+.3f} median {np.median(ov):+.3f}; TP bar high vs 1.2 mean {np.mean([(1 + hi[i, te[i]]) / 1.2 - 1 for i in idxh]):+.3f}"  # noqa: E501
)
# fade (short at 1.2) with 24h hold and 72h hold, cost .4%
for lab, dt in (("4h", 240), ("24h", 1440), ("72h", 4320)):
    f = []
    for i in idxh:
        t0 = STEP_END_MIN[te[i]]
        j = min(int(np.searchsorted(STEP_END_MIN, min(t0 + dt, 4320))), 159)
        f.append(-((1 + cl[i, j]) / 1.2 - 1) - 0.004)
    print(f"  fade at TP hold {lab}: mean {np.mean(f) * 1e4:+.0f}bp")
# reversal frequency after arm levels (close-based), vs trail
print("== trailing diagnostics: of rows whose best close reached arm level A, what happens")
best = np.maximum.accumulate(cl, axis=1)
for A in (0.1, 0.15, 0.2, 0.3):
    sel = best[:, -1] >= A
    c72 = cl[sel, -1]
    # fell below entry by 72h; fell more than 10% from the best by 72h
    print(
        f"  arm {A}: n={sel.sum()} ({sel.mean():.2f}); mean best {best[sel, -1].mean():.3f}; mean 72h close {c72.mean():.3f}; mean (close72 - best) {np.mean(c72 - best[sel, -1]):+.3f}; close72<0: {np.mean(c72 < 0):.2f}; close72<A: {np.mean(c72 < A):.2f}; 72h close ge best-0.05: {np.mean(c72 >= best[sel, -1] - 0.05):.2f}"  # noqa: E501
    )
for A, Tr in ((0.2, 0.1), (0.3, 0.1), (0.1, 0.1)):
    ret, te2, kk = exits(p, np.ones(len(r)), Rule(hold_min=4320, trail=Tr, trail_arm=A))
    print(
        f"  trail arm {A} trail {Tr} 72h: mean net {np.mean(ret - 0.004) * 1e4:+.0f}bp (trail exits {np.mean(kk == 4):.2f})"  # noqa: E501
    )
# peak-given-hit bimodality: distribution of peak among those reaching 20
pk = peak[hit]
print(
    " peak among hits: quantiles 10/25/50/75/90:",
    [round(float(np.quantile(pk, q)), 2) for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
)
# whether more upside is predictable: corr(day5 score, further max) among hits
print(
    " corr(day5 pct, peak) among hits:",
    round(float(np.corrcoef(pct[r][hit], pk)[0, 1]), 3),
    "; all candidates",
    round(float(np.corrcoef(pct[r], peak)[0, 1]), 3),
)
# low-score control: crash rows low day5 hitting 20
