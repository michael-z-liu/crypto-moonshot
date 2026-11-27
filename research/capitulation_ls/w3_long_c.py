"""W3-C: exit rules in the slot book, ensemble ceiling, forecast calibration, target linkage."""

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, simulate_yearly

Y = (2022, 2023, 2024, 2025, 2026)
m = meta()
yr = m["year"].to_numpy()
pct = scores("day5")
idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.995))
c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
print("== exit rules in the 10-slot spot book (cost .4%), fresh $10k book per year")
rules = {
    "TP20/72h": Rule(hold_min=4320, tp=0.2, slots=10),
    "TP30/72h": Rule(hold_min=4320, tp=0.3, slots=10),
    "TP50/72h": Rule(hold_min=4320, tp=0.5, slots=10),
    "TP10/72h": Rule(hold_min=4320, tp=0.1, slots=10),
    "no TP 72h": Rule(hold_min=4320, slots=10),
    "no TP 24h": Rule(hold_min=1440, slots=10),
    "trail arm.2/.1 72h": Rule(hold_min=4320, trail=0.1, trail_arm=0.2, slots=10),
    "trail arm.3/.1 72h": Rule(hold_min=4320, trail=0.1, trail_arm=0.3, slots=10),
    "trail arm.3/.1 + TP50": Rule(hold_min=4320, trail=0.1, trail_arm=0.3, tp=0.5, slots=10),
}
books = {}
for k, r in rules.items():
    b = simulate_yearly(c, r, years=Y)
    books[k] = b
    ys = b.years.sort("year")
    print(
        f" {k:24s} n={b.summary['trades']:>3} {b.summary['bp']:+5.0f}bp  yearly ret {[round(v * 100) for v in ys['ret'].to_list()]}  mean hold h {b.trades['hold_min'].mean() / 60:.0f}"  # noqa: E501
    )


# day-cluster bootstrap of paired per-trade difference (same rows) TP20 vs trail on unconstrained candidates not needed; use book trades by day  # noqa: E501
def dayboot(a, b, B=4000):
    ta = a.trades.with_columns(day=pl.col("open_time").dt.date())
    tb = b.trades.with_columns(day=pl.col("open_time").dt.date())
    ga = ta.group_by("day").agg(s=pl.col("net").sum(), n=pl.len())
    gb = tb.group_by("day").agg(s2=pl.col("net").sum(), n2=pl.len())
    g = ga.join(gb, on="day", how="full", coalesce=True).fill_null(0)
    rng = np.random.default_rng(0)
    out = []
    s, n, s2, n2 = g["s"].to_numpy(), g["n"].to_numpy(), g["s2"].to_numpy(), g["n2"].to_numpy()
    for _ in range(B):
        i = rng.integers(0, len(g), len(g))
        out.append(s[i].sum() / n[i].sum() - s2[i].sum() / n2[i].sum())
    out = np.array(out)
    return np.mean(out), np.quantile(out, [0.05, 0.95])


for k in ("trail arm.3/.1 72h", "trail arm.2/.1 72h", "no TP 72h", "TP30/72h"):
    mu, ci = dayboot(books[k], books["TP20/72h"])
    print(
        f" per-trade {k} minus TP20 (day-cluster bootstrap): {mu * 1e4:+.0f}bp, 90% CI [{ci[0] * 1e4:+.0f},{ci[1] * 1e4:+.0f}] over {books[k].trades['open_time'].dt.date().n_unique()} days"  # noqa: E501
    )
# worst-case profile: loss tail per rule
for k in ("TP20/72h", "no TP 72h", "trail arm.3/.1 72h"):
    t = books[k].trades
    print(
        f" {k:22s} p5 net {t['net'].quantile(0.05) * 1e4:+.0f} worst {t['net'].min() * 1e4:+.0f} maxDD {books[k].summary['maxdd']:.2f}"  # noqa: E501
    )
# ---------- ensemble ceiling
d = pl.read_parquet("/tmp/w3_tab.parquet").with_columns(
    pl.col("day").cast(pl.Date),
    y1c=pl.col("y1h").clip(-0.2, 0.2),
    y4c=pl.col("y4h").clip(-0.2, 0.5),
)
names = ["day5", "day10b", "day_transformer", "day_tcn", "day_gru"]
d = d.with_columns(
    ens=sum(pl.col(n).rank() / pl.len() for n in names) / len(names),
    ens3=(pl.col("day5").rank() + pl.col("day_transformer").rank() + pl.col("day_gru").rank())
    / 3
    / pl.len(),
)


def wic(s, a, b):
    return s.select(
        pl.corr(pl.col(a) - pl.col(a).mean().over("day"), pl.col(b) - pl.col(b).mean().over("day"))
    ).item()


print("== within-day IC (pooled demeaned) on all 2022+ rows")
for a in [*names, "ens", "ens3"]:
    print(
        f" {a:16s} y1c {wic(d, a, 'y1c'):+.4f} y4c {wic(d, a, 'y4c'):+.4f} o72 {wic(d, a, 'o72'):+.4f}"  # noqa: E501
    )
top = d.filter(pl.col("day5") >= 0.98)
print(f"== inside the day5>=0.98 pool (n={top.height}): within-day IC")
for a in [*names, "tp20day", "ens"]:
    print(
        f" {a:16s} y1c {wic(top, a, 'y1c'):+.3f} y4c {wic(top, a, 'y4c'):+.3f} o72 {wic(top, a, 'o72'):+.3f}"  # noqa: E501
    )
# target linkage
cr = d.filter(pl.col("ret_4h") <= -0.10)
print("== how much does the 1h outcome tell about the 72h outcome (day-demeaned correlation)")
for nm, s in (("all rows", d), ("crash rows", cr), ("day5>=.98", top)):
    print(
        f" {nm:11s} corr(y1c,o72) {wic(s, 'y1c', 'o72'):+.3f}  corr(y4c,o72) {wic(s, 'y4c', 'o72'):+.3f}  corr(y1c,y4c) {wic(s, 'y1c', 'y4c'):+.3f}; raw (not demeaned) corr(y1c,o72) {s.select(pl.corr('y1c', 'o72')).item():+.3f}"  # noqa: E501
    )
# predictable variance share
for y in ("y1c", "o72"):
    v_all = d[y].var()
    within = d.select((pl.col(y) - pl.col(y).mean().over("day")).var()).item()
    print(
        f" {y}: within-day variance share (all rows) {within / v_all:.3f}; crash {cr.select((pl.col(y) - pl.col(y).mean().over('day')).var()).item() / cr[y].var():.3f}; day5>=.98 {top.select((pl.col(y) - pl.col(y).mean().over('day')).var()).item() / top[y].var():.3f}"  # noqa: E501
    )
# ---------- forecast calibration on the top set
P = scores("day5", "path").astype(np.float32)  # percent, net of cost, [n,32,3]
rows = top["row"].to_numpy()
Pr = P[rows]
c8 = top["c8"].to_numpy()
c1 = top["c1"].to_numpy()
print("== day5 path forecast on pool day5>=0.98: predicted vs realised (percent, net of .3% cost)")
for lab, step, real in (("1h", 3, c1), ("8h", 31, c8)):
    real = np.clip(real - 0.003, -0.3, 0.3) * 100
    lo, md, hi = Pr[:, step, 0], Pr[:, step, 1], Pr[:, step, 2]
    print(
        f" {lab}: pred median mean {md.mean():+.2f} sd {md.std():.2f}; realised mean {real.mean():+.2f} sd {real.std():.2f}; coverage <q10 {np.mean(real < lo):.2f} >q90 {np.mean(real > hi):.2f}; corr(pred median, realised) {np.corrcoef(md, real)[0, 1]:+.3f}; mean band q90-q10 {np.mean(hi - lo):.1f}"  # noqa: E501
    )
allr = np.flatnonzero(np.isfinite(pct) & (yr >= 2022))
Pa = P[allr]
real = (
    np.clip(
        np.nan_to_num(np.asarray(__import__("common_ls").fwd2()[allr][:, 2, 11])) - 0.003, -0.3, 0.3
    )
    * 100
)
print(
    f" all rows 1h: pred median sd {Pa[:, 3, 1].std():.2f} vs realised sd {real.std():.2f}; coverage <q10 {np.mean(real < Pa[:, 3, 0]):.2f} >q90 {np.mean(real > Pa[:, 3, 2]):.2f}"  # noqa: E501
)
