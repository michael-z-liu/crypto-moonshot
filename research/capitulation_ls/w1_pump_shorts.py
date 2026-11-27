"""W1: why do pump shorts fail on perps? Diagnostics for the mechanisms (no rescue attempts).
Sections: A gap perp vs spot, B stop / path shape, C squeeze predictability, D model top decile,
E regime (year, BTC 7d)."""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
from common_ls import CACHE_ALL, OWN, first_per_event, meta
from sim import paths, perp_paths
from sklearn.metrics import roc_auc_score

pl.Config.set_tbl_width_chars(400)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_rows(60)
K = 96 + 63  # path index of hour 72
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
bp = 1e4

f = pl.read_parquet(OWN / "feats.parquet").filter(
    pl.col("perp")
    & (pl.col("qv_base") >= 20_000)
    & (pl.col("year") >= 2021)
    & (pl.col("ret_4h") >= THR)
)
DEDUP = (sys.argv[2] if len(sys.argv) > 2 else "1") == "1"
if DEDUP:
    keep = first_per_event(np.isin(np.arange(meta().height), f["row"].to_numpy()))
    f = f.filter(pl.col("row").is_in(keep.tolist()))
f = f.sort("row")
rows = f["row"].to_numpy()
pp, ok = perp_paths(rows)
f = f.filter(pl.Series(ok))
rows = f["row"].to_numpy()
sp = paths(rows)
nanfrac = np.isnan(pp[:, 2, : K + 1]).mean(1)
ph = np.nan_to_num(pp[:, 0, : K + 1], nan=0)
pc = pp[:, 2, : K + 1]
# last valid close (frozen / delisted perps end early)
lastc = np.array([c[~np.isnan(c)][-1] if (~np.isnan(c)).any() else 0.0 for c in pc])
perp_short = -lastc
spot_short = -sp[:, 2, K]
adv_p = ph.max(1)
adv_s = np.nan_to_num(sp[:, 0, : K + 1], nan=0).max(1)
t_peak = ph.argmax(1)  # index of the max adverse high
hours_peak = np.where(t_peak < 96, t_peak * 5 / 60, 8 + (t_peak - 96))
iso = np.where(adv_p >= 0.99, -1.0, perp_short) - 0.003

# funding paid by a short over 72h
from futures_ls import funding_paid  # noqa: E402

t0 = f["t"].to_numpy() + 300
fund = funding_paid(f["symbol"].to_numpy(), t0, t0 + 72 * 3600, -np.ones(len(f)))
f = f.with_columns(
    perp_short=pl.Series(perp_short),
    spot_short=pl.Series(spot_short),
    adv_p=pl.Series(adv_p),
    adv_s=pl.Series(adv_s),
    hp=pl.Series(hours_peak),
    iso=pl.Series(iso),
    fund=pl.Series(fund),
    nanfrac=pl.Series(nanfrac),
    gap=pl.Series(perp_short - spot_short),
)
print(f"=== sample: perp-active pumps ret_4h>={THR}, dedup={DEDUP}, 2021+: n={f.height}")

# ---------------- A: gap
print("\n=== A1 perp vs spot short, 72h (bp, before cost)")
print(
    f.group_by("year")
    .agg(
        n=pl.len(),
        spot=pl.col("spot_short").mean() * bp,
        perp_raw=pl.col("perp_short").mean() * bp,
        perp_iso=(pl.col("iso").mean() + 0.003) * bp,
        spot_med=pl.col("spot_short").median() * bp,
        perp_med=pl.col("perp_short").median() * bp,
        gap_med=pl.col("gap").median() * bp,
        fund_bp=pl.col("fund").mean() * bp,
    )
    .sort("year")
)
print(
    f.select(
        n=pl.len(),
        spot=pl.col("spot_short").mean() * bp,
        perp_raw=pl.col("perp_short").mean() * bp,
        perp_iso=(pl.col("iso").mean() + 0.003) * bp,
        gap_mean=pl.col("gap").mean() * bp,
        gap_med=pl.col("gap").median() * bp,
        fund=pl.col("fund").mean() * bp,
        sq100_perp=(pl.col("adv_p") >= 1).mean(),
        sq100_spot=(pl.col("adv_s") >= 1).mean(),
        sq50_perp=(pl.col("adv_p") >= 0.5).mean(),
        sq50_spot=(pl.col("adv_s") >= 0.5).mean(),
        frozen_share=(pl.col("nanfrac") > 0.05).mean(),
    )
)
g = f.sort("gap")
tot = g["gap"].sum()
print(
    "gap decomposition: mean gap",
    round(float(g["gap"].mean()) * bp),
    "bp; mean gap excluding the k most negative gaps (perp rose more):",
)
for k in (0, 3, 5, 10, 20, 50):
    print("  k", k, round(float(g["gap"].slice(k, None).mean()) * bp), "bp")
print("share of |gap|>=50% events:", int((g["gap"].abs() >= 0.5).sum()), "of", g.height)
print(
    g.head(10).select(
        "symbol",
        "open_time",
        "ret_4h",
        "spot_short",
        "perp_short",
        "adv_p",
        "adv_s",
        "nanfrac",
        "gap",
    )
)
# the squeeze subset: perp doubles, does spot?
sq = f.filter(pl.col("adv_p") >= 1)
print(
    "perp >=+100%:",
    sq.height,
    "of those spot also >=+100%:",
    int((sq["adv_s"] >= 1).sum()),
    "; spot adv median",
    round(float(sq["adv_s"].median()), 2),
    "perp adv median",
    round(float(sq["adv_p"].median()), 2),
)
print(
    "events where adv_p>=1 contribute to mean perp_raw (bp):",
    round(float(sq["perp_short"].sum() / f.height) * bp),
    "; same events on spot:",
    round(float(sq["spot_short"].sum() / f.height) * bp),
)
print("events with adv_p>=1 but adv_s<1:", int(((f["adv_p"] >= 1) & (f["adv_s"] < 1)).sum()))
# funding: entry rate vs squeeze
print(
    f.group_by(pl.col("rate").qcut(5, allow_duplicates=True))
    .agg(
        n=pl.len(),
        sq100=(pl.col("adv_p") >= 1).mean(),
        perp=pl.col("perp_short").mean() * bp,
        fund=pl.col("fund").mean() * bp,
    )
    .sort("rate")
)
print(
    "funding paid by short, squeeze vs not (bp of notional over 72h):",
    round(float(sq["fund"].mean()) * bp),
    round(float(f.filter(pl.col("adv_p") < 1)["fund"].mean()) * bp),
)

# ---------------- B: path shape and stops
print("\n=== B1 touch / revert table (perp paths, 72h)")
rows_b = []
for th in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
    m = adv_p >= th
    fin = perp_short[m]
    rows_b.append(
        dict(
            touch=th,
            n=int(m.sum()),
            share=round(float(m.mean()), 3),
            final_below_entry=round(float((fin > 0).mean()), 3),
            final_gt_minus_th=round(float((fin > -th).mean()), 3),
            mean_final_bp=round(float(fin.mean()) * bp),
            med_final_bp=round(float(np.median(fin)) * bp),
            med_hours_to_peak=round(float(np.median(hours_peak[m])), 1),
        )
    )
print(pl.DataFrame(rows_b))
print(
    "max adverse quantiles (perp):",
    {q: round(float(np.quantile(adv_p, q)), 3) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
)
print("corr(max adverse, final short):", round(float(np.corrcoef(adv_p, perp_short)[0, 1]), 3))
print(
    "\n=== B2 stop-loss value (hourly highs, trigger at thr, exit exactly at thr, no slip; 30bp cost)"  # noqa: E501
)
base = perp_short - 0.003 - fund
print("hold no stop mean:", round(float(base.mean()) * bp), "bp (cross margin)")
rows_b = []
for th in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
    hit = adv_p >= th
    r = np.where(hit, -th - 0.003 - fund, base)
    rows_b.append(
        dict(
            sl=th,
            hit=round(float(hit.mean()), 3),
            mean_bp=round(float(r.mean()) * bp),
            loss_from_stopped_winners=round(
                float(((r - base)[hit & (perp_short > 0)]).sum() / len(r)) * bp
            ),
            gain_from_stopped_losers=round(
                float(((r - base)[hit & (perp_short <= 0)]).sum() / len(r)) * bp
            ),
        )
    )
print(pl.DataFrame(rows_b))
# the same by year at sl 0.5
for y in sorted(set(f["year"].to_list())):
    mk = f["year"].to_numpy() == y
    h5 = adv_p[mk] >= 0.5
    print(
        y,
        "n",
        int(mk.sum()),
        "touch50",
        round(float(h5.mean()), 3),
        "of touchers ending below entry",
        round(float((perp_short[mk][h5] > 0).mean()), 3) if h5.any() else None,
    )
print(
    "distribution of final short return, all trades: p5/p25/p50/p75/p95 (bp)",
    [round(float(np.quantile(perp_short, q)) * bp) for q in (0.05, 0.25, 0.5, 0.75, 0.95)],
)
# what share of the total profit is made by trades that were under water >=30% first
for th in (0.2, 0.3, 0.5):
    m = adv_p >= th
    print(
        f"trades that went >= +{th:.0%} against: {m.mean():.1%} of trades, their mean final {perp_short[m].mean() * bp:+.0f} bp; the rest {perp_short[~m].mean() * bp:+.0f} bp"  # noqa: E501
    )

# ---------------- C: predictability
print("\n=== C1 base rates and clustering")
f = f.with_columns(sq100=pl.col("adv_p") >= 1, sq50=pl.col("adv_p") >= 0.5)
print(
    f.group_by("year")
    .agg(n=pl.len(), sq50=pl.col("sq50").mean(), sq100=pl.col("sq100").mean())
    .sort("year")
)
sqd = f.filter(pl.col("sq100"))
cnt = sqd.group_by("symbol").len().sort("len", descending=True)
print("sq100 events", sqd.height, "coins", cnt.height, "top coins", cnt.head(8).to_dicts())
print(
    "share of sq100 in coins with >=2 sq100:",
    round(float(cnt.filter(pl.col("len") >= 2)["len"].sum() / sqd.height), 3),
)
allc = f.group_by("symbol").agg(n=pl.len(), s=pl.col("sq100").sum())
print(
    "coin-level: coins with >=2 events",
    int((allc["n"] >= 2).sum()),
    "; P(sq100 | coin had earlier sq100)? compute chronologically:",
)
srt = f.sort("open_time")
seen: dict[str, int] = {}
prev_pos, prev_n, nonprev_pos, nonprev_n = 0, 0, 0, 0
for s, q in zip(srt["symbol"].to_list(), srt["sq100"].to_list(), strict=False):
    if seen.get(s, 0) > 0:
        prev_n += 1
        prev_pos += q
    else:
        nonprev_n += 1
        nonprev_pos += q
    seen[s] = seen.get(s, 0) + int(q)
print(
    "  after an earlier squeeze in the same coin:",
    prev_pos,
    "/",
    prev_n,
    "; otherwise",
    nonprev_pos,
    "/",
    nonprev_n,
)
print(
    "timing of peak adverse among sq100 (hours): quantiles",
    [round(float(np.quantile(sqd["hp"], q)), 1) for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
)
print(
    "\n=== C2 univariate AUC (sq100 / sq50) by feature, whole sample (pooled, then by-year median)"
)
feats = [
    "ret_4h",
    "ret_24h",
    "vol_mult",
    "listed_days",
    "qv_base",
    "rate",
    "r7d",
    "r30d",
    "r90d",
    "n_prior_pumps30",
    "taker",
    "taker1h",
    "off_high4h",
    "wick_up",
    "r1h",
    "r5m",
    "mkt_r24h",
    "btc_r24h",
    "vol30d",
    "dhi90d",
    "fund_age_h",
]
out = []
yr = f["year"].to_numpy()
for c in feats:
    x = np.nan_to_num(f[c].fill_null(0).to_numpy().astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    row = {"feat": c}
    for lab, y in (("sq100", f["sq100"].to_numpy()), ("sq50", f["sq50"].to_numpy())):
        row[lab] = round(float(roc_auc_score(y, x)), 3)
        aucs = [
            roc_auc_score(y[yr == yy], x[yr == yy])
            for yy in range(2022, 2027)
            if y[yr == yy].sum() >= 3
        ]
        row[lab + "_yr_min_max"] = f"{min(aucs):.2f}-{max(aucs):.2f}" if aucs else ""
    out.append(row)
print(pl.DataFrame(out))
# signal-to-noise: number of sq100 events per year, so AUC noise
print(
    "sq100 count by year", f.filter(pl.col("sq100")).group_by("year").len().sort("year").to_dicts()
)
# bootstrap AUC noise for a random feature with the same positives (2025)
rng = np.random.default_rng(0)
for yy in (2023, 2025):
    y = f["sq100"].to_numpy()[yr == yy]
    a = [roc_auc_score(y, rng.random(len(y))) for _ in range(300)]
    print(f"null AUC sd for {yy}: n={len(y)}, positives={int(y.sum())}, sd={np.std(a):.3f}")

# ---------------- D: model top decile
print(
    "\n=== D model ranking vs squeeze exposure (walk-forward 2022-26, perp-active pumps in sample)"
)
for name in ("nn_short_transformer_72h", "nn_short_mlp_72h", "e08_short_72h", "e19_base_72h"):
    p = pl.read_parquet(OWN / f"{name}.parquet").select("row", pq="pq")
    x = f.join(p, on="row", how="inner").filter(
        pl.col("pq").is_not_nan() & pl.col("pq").is_not_null() & (pl.col("year") >= 2022)
    )
    if x.height < 100:
        print(name, "too few joined rows", x.height)
        continue
    x = x.with_columns(
        b=pl.col("pq").qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True),
        clip=pl.col("spot_short").clip(-0.5, 1.0),
    )
    print(
        name,
        "n joined",
        x.height,
        "rank corr(pq, ret_4h)=",
        round(float(x.select(pl.corr(pl.col("pq").rank(), pl.col("ret_4h").rank()))[0, 0]), 3),
        "corr(pq,listed_days)=",
        round(float(x.select(pl.corr(pl.col("pq").rank(), pl.col("listed_days").rank()))[0, 0]), 3),
    )
    print(
        x.group_by("b")
        .agg(
            n=pl.len(),
            ret4=pl.col("ret_4h").mean(),
            listed=pl.col("listed_days").median(),
            clip_spot=pl.col("clip").mean() * bp,
            spot=pl.col("spot_short").mean() * bp,
            perp_raw=pl.col("perp_short").mean() * bp,
            iso=pl.col("iso").mean() * bp,
            med_perp=pl.col("perp_short").median() * bp,
            sq50=pl.col("sq50").mean(),
            sq100=pl.col("sq100").mean(),
            adv_mean=pl.col("adv_p").mean(),
        )
        .sort("b")
    )
    top = x.filter(pl.col("pq") >= 0.9)
    rest = x.filter(pl.col("pq") < 0.9)
    print(
        "  top decile vs rest: iso",
        round(float(top["iso"].mean()) * bp),
        round(float(rest["iso"].mean()) * bp),
        "| median perp",
        round(float(top["perp_short"].median()) * bp),
        round(float(rest["perp_short"].median()) * bp),
        "| sq100",
        round(float(top["sq100"].mean()), 3),
        round(float(rest["sq100"].mean()), 3),
    )
    # bootstrap se of top-decile iso mean
    v = top["iso"].to_numpy()
    se = np.std(v) / np.sqrt(len(v))
    print("  top decile iso mean se (bp):", round(float(se) * bp), "n", len(v))

# ---------------- E: regime
print("\n=== E regime")
os.environ["MOONSHOT_UNIVERSE"] = "all"
sys.path.insert(0, str(CACHE_ALL.parent))
from common import load  # noqa: E402

b = (
    load("BTCUSDT")
    .group_by_dynamic("open_time", every="1h")
    .agg(c=pl.col("close").last())
    .sort("open_time")
)
bt, bc = b["open_time"].dt.epoch("s").to_numpy(), b["c"].to_numpy()


def btc_trend(t: np.ndarray, days: int) -> np.ndarray:
    k = np.searchsorted(bt, t - 3600, side="right") - 1
    j = np.searchsorted(bt, t - 3600 - days * 86400, side="right") - 1
    return bc[np.maximum(k, 0)] / bc[np.maximum(j, 0)] - 1


f = f.with_columns(
    b7=pl.Series(btc_trend(f["t"].to_numpy(), 7)), b30=pl.Series(btc_trend(f["t"].to_numpy(), 30))
)
f = f.with_columns(b7_up=pl.col("b7") > -0.03)
f = f.with_columns(net=pl.col("iso") - pl.col("fund"))
print(
    f.group_by("b7_up").agg(
        n=pl.len(),
        net_bp=pl.col("net").mean() * bp,
        spot_bp=pl.col("spot_short").mean() * bp,
        med_perp=pl.col("perp_short").median() * bp,
        sq50=pl.col("sq50").mean(),
        sq100=pl.col("sq100").mean(),
        ret4=pl.col("ret_4h").mean(),
        listed_med=pl.col("listed_days").median(),
    )
)
print(
    f.group_by(["year", "b7_up"])
    .agg(
        n=pl.len(),
        net_bp=pl.col("net").mean() * bp,
        med_perp=pl.col("perp_short").median() * bp,
        sq50=pl.col("sq50").mean(),
        sq100=pl.col("sq100").mean(),
    )
    .sort(["year", "b7_up"])
)
print(
    "share of pumps with b7 <= -3% by year:",
    f.group_by("year")
    .agg(share=(~pl.col("b7_up")).mean(), b7=pl.col("b7").mean())
    .sort("year")
    .to_dicts(),
)
# what in the market differs: pump composition by year
print(
    f.group_by("year")
    .agg(
        n=pl.len(),
        young=(pl.col("listed_days") < 60).mean(),
        ret4=pl.col("ret_4h").mean(),
        r30=pl.col("r30d").median(),
        r90=pl.col("r90d").median(),
        dhi90=pl.col("dhi90d").median(),
        mkt24=pl.col("mkt_r24h").median(),
        rate=pl.col("rate").median(),
        net=pl.col("net").mean() * bp,
        med=pl.col("perp_short").median() * bp,
    )
    .sort("year")
)
# by young vs old and year
print(
    f.with_columns(young=pl.col("listed_days") < 90)
    .group_by(["year", "young"])
    .agg(
        n=pl.len(),
        net=pl.col("net").mean() * bp,
        med=pl.col("perp_short").median() * bp,
        sq100=pl.col("sq100").mean(),
    )
    .sort(["year", "young"])
)
# is the 2025-26 mean a handful of events? top contributions
s = f.filter(pl.col("year") >= 2025).sort("net", descending=True)
print(
    "2025-26 n",
    s.height,
    "mean net",
    round(float(s["net"].mean()) * bp),
    "bp; mean excluding top 10 winners",
    round(float(s["net"].slice(10, None).mean()) * bp),
    "; excluding top 10 winners and 10 losers",
    round(float(s["net"].slice(10, s.height - 20).mean()) * bp),
)
s = f.filter(pl.col("year") < 2025)
print(
    "2021-24 n",
    s.height,
    "mean net",
    round(float(s["net"].mean()) * bp),
    "median perp short",
    round(float(s["perp_short"].median()) * bp),
    "mean spot short",
    round(float(s["spot_short"].mean()) * bp),
)
# 72h mean of altcoin market: median 72h perp return of the whole sample by year (market drift on pumped coins)  # noqa: E501
print(
    "mean 72h return of ALL pumped coins (long), by year, bp:",
    f.group_by("year")
    .agg(r=(-pl.col("perp_short")).mean() * bp, rm=(-pl.col("perp_short")).median() * bp)
    .sort("year")
    .to_dicts(),
)
