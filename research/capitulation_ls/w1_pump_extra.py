"""W1 extra diagnostics for the pump-short note: rank control, loss decomposition of the model's
top decile, regime table on a deduped sample, bootstrap of the BTC-7d split, squeeze hazard,
metric AUC power, ALPACA case."""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
from common_ls import CACHE_ALL, OWN, first_per_event, meta
from futures_ls import funding_paid
from sim import paths, perp_paths
from sklearn.metrics import roc_auc_score

pl.Config.set_tbl_width_chars(300)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_rows(40)
K = 96 + 63
bp = 1e4


def build(thr: float, dedup: bool) -> pl.DataFrame:
    f = pl.read_parquet(OWN / "feats.parquet").filter(
        pl.col("perp")
        & (pl.col("qv_base") >= 20_000)
        & (pl.col("year") >= 2021)
        & (pl.col("ret_4h") >= thr)
    )
    if dedup:
        keep = first_per_event(np.isin(np.arange(meta().height), f["row"].to_numpy()))
        f = f.filter(pl.col("row").is_in(keep.tolist()))
    f = f.sort("row")
    rows = f["row"].to_numpy()
    pp, ok = perp_paths(rows)
    f = f.filter(pl.Series(ok))
    rows = f["row"].to_numpy()
    sp = paths(rows)
    pc = pp[:, 2, : K + 1]
    lastc = np.array([c[~np.isnan(c)][-1] if (~np.isnan(c)).any() else 0.0 for c in pc])
    ph = np.nan_to_num(pp[:, 0, : K + 1], nan=0)
    adv = ph.max(1)
    t0 = f["t"].to_numpy() + 300
    fund = funding_paid(f["symbol"].to_numpy(), t0, t0 + 72 * 3600, -np.ones(len(f)))
    ps, ss = -lastc, -sp[:, 2, K]
    iso = np.where(adv >= 0.99, -1.0, ps)
    # first 8h max adverse (5m resolution) for hazard
    adv8 = ph[:, :96].max(1)
    return f.with_columns(
        perp_short=pl.Series(ps),
        spot_short=pl.Series(ss),
        adv_p=pl.Series(adv),
        adv8=pl.Series(adv8),
        iso_raw=pl.Series(iso),
        fund=pl.Series(fund),
        net=pl.Series(iso - 0.003 - fund),
        sq100=pl.Series(adv >= 1),
        sq50=pl.Series(adv >= 0.5),
    )


# ------------------------------------------------ 1 rank control (ret_4h >= 0.15, all spikes)
d = build(0.15, False)
p = pl.read_parquet(OWN / "nn_short_transformer_72h.parquet").select("row", "pq")
x = d.join(p, on="row").filter(
    pl.col("pq").is_not_nan() & pl.col("pq").is_not_null() & (pl.col("year") >= 2022)
)
x = x.with_columns(
    size=pl.col("ret_4h").cut(
        [0.2, 0.3, 0.4, 0.6], labels=["15-20", "20-30", "30-40", "40-60", "60+"]
    )
)
print(
    "=== 1a within pump-size bins: model top 30% vs bottom 30% (net = iso liquidation - cost - funding), transformer"  # noqa: E501
)
rows_ = []
for sz in ["15-20", "20-30", "30-40", "40-60", "60+"]:
    g = x.filter(pl.col("size") == sz)
    hi, lo = g.filter(pl.col("pq") >= 0.7), g.filter(pl.col("pq") <= 0.3)
    ic = float(g.select(pl.corr(pl.col("pq").rank(), pl.col("net").rank()))[0, 0])
    ics = float(
        g.select(pl.corr(pl.col("pq").rank(), pl.col("spot_short").clip(-0.5, 1).rank()))[0, 0]
    )
    rows_.append(
        dict(
            size=sz,
            n=g.height,
            ic_clip_spot=round(ics, 3),
            ic_net=round(ic, 3),
            hi_n=hi.height,
            hi_net=round(float(hi["net"].mean()) * bp),
            lo_net=round(float(lo["net"].mean()) * bp),
            hi_sq100=round(float(hi["sq100"].mean()), 3),
            lo_sq100=round(float(lo["sq100"].mean()), 3),
            hi_med=round(float(hi["perp_short"].median()) * bp),
            lo_med=round(float(lo["perp_short"].median()) * bp),
        )
    )
print(pl.DataFrame(rows_))
print(
    "overall rank IC pq vs clipped spot short:",
    round(
        float(
            x.select(pl.corr(pl.col("pq").rank(), pl.col("spot_short").clip(-0.5, 1).rank()))[0, 0]
        ),
        3,
    ),
    "vs net:",
    round(float(x.select(pl.corr(pl.col("pq").rank(), pl.col("net").rank()))[0, 0]), 3),
    "vs ret_4h rank:",
    round(float(x.select(pl.corr(pl.col("pq").rank(), pl.col("ret_4h").rank()))[0, 0]), 3),
)
print(
    "IC of ret_4h alone vs clipped spot short:",
    round(
        float(
            x.select(pl.corr(pl.col("ret_4h").rank(), pl.col("spot_short").clip(-0.5, 1).rank()))[
                0, 0
            ]
        ),
        3,
    ),
    "vs net:",
    round(float(x.select(pl.corr(pl.col("ret_4h").rank(), pl.col("net").rank()))[0, 0]), 3),
)

print("\n=== 1b loss decomposition, model top decile vs rest (bp per trade), transformer, 2022-26")
top, rest = x.filter(pl.col("pq") >= 0.9), x.filter(pl.col("pq") < 0.9)
for lab, g in (("top10%", top), ("rest", rest)):
    clipv = float(g["spot_short"].clip(-0.5, 1.0).mean()) * bp
    spot = float(g["spot_short"].mean()) * bp
    perp = float(g["perp_short"].mean()) * bp
    iso = float(g["iso_raw"].mean()) * bp
    fund = float(g["fund"].mean()) * bp
    print(
        f"{lab:7s} n={g.height}: clipped spot {clipv:+.0f} | unclipped spot {spot:+.0f} (clip removes {spot - clipv:+.0f}) | "  # noqa: E501
        f"perp raw {perp:+.0f} (perp-vs-spot {perp - spot:+.0f}) | iso-liquidation {iso:+.0f} ({iso - perp:+.0f}) | "  # noqa: E501
        f"cost -30 | funding paid {fund:.0f} -> net {iso - 30 - fund:+.0f}"
    )
    print(
        f"        sq100 {g['sq100'].mean():.3f}  sq50 {g['sq50'].mean():.3f}  ret_4h mean {g['ret_4h'].mean():.2f}  mean adverse {g['adv_p'].mean():.2f}"  # noqa: E501
    )
print(
    "share of top-decile trades with ret_4h >= 0.4:",
    round(float((top["ret_4h"] >= 0.4).mean()), 3),
    "; of rest:",
    round(float((rest["ret_4h"] >= 0.4).mean()), 3),
)

# ------------------------------------------------ 2 regime on a deduped (24h lock) sample
print("\n=== 2 regime table on deduped (coin lock 24h) sample, ret_4h >= 0.15")
dd = build(0.15, True)
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


# perp age (hours since first funding record) and perp 24h quote volume before entry
from futures_ls import perp_active, perp_window, spot_name  # noqa: E402,F401

win, act = perp_window(), perp_active()
age, pqv = [], []
for s, t in zip(dd["symbol"].to_list(), dd["t"].to_numpy(), strict=False):
    w = win.get(s)
    age.append((t - w[0]) / 86400 if w else np.nan)
    a = act.get(s)
    if a is None:
        pqv.append(np.nan)
        continue
    h, q = a
    i = np.searchsorted(h, t, side="right")
    pqv.append(q[max(0, i - 24) : i].sum())
dd = dd.with_columns(
    b7=pl.Series(btc_trend(dd["t"].to_numpy(), 7)),
    perp_age_d=pl.Series(age),
    perp_qv24=pl.Series(pqv),
)
dd = dd.with_columns(young_perp=pl.col("perp_age_d") < 60, neg_rate=pl.col("rate") < 0)
print(
    dd.group_by("year")
    .agg(
        n=pl.len(),
        net=pl.col("net").mean() * bp,
        med_fade=pl.col("perp_short").median() * bp,
        mean_fade=pl.col("perp_short").mean() * bp,
        sq50=pl.col("sq50").mean(),
        sq100=pl.col("sq100").mean(),
        fund_paid=pl.col("fund").mean() * bp,
        neg_rate=pl.col("neg_rate").mean(),
        young_perp=pl.col("young_perp").mean(),
        perp_qv24_med_musd=pl.col("perp_qv24").median() / 1e6,
        ret4=pl.col("ret_4h").median(),
        b7=pl.col("b7").mean(),
        b7_low=(pl.col("b7") < -0.03).mean(),
    )
    .sort("year")
)
print("\nby BTC 7d bucket (pooled / 2024-26 / 2021-23), deduped sample")
for lab, sub in (
    ("all", dd),
    ("2021-23", dd.filter(pl.col("year") <= 2023)),
    ("2024-26", dd.filter(pl.col("year") >= 2024)),
):
    print(
        lab,
        sub.group_by(pl.col("b7") > -0.03)
        .agg(
            n=pl.len(),
            net=pl.col("net").mean() * bp,
            med=pl.col("perp_short").median() * bp,
            sq50=pl.col("sq50").mean(),
            sq100=pl.col("sq100").mean(),
        )
        .sort("b7")
        .to_dicts(),
    )
rng = np.random.default_rng(0)
for thr in (0.15, 0.4):
    s = dd.filter(pl.col("ret_4h") >= thr)
    for lab, sub in (("all", s), ("2024-26", s.filter(pl.col("year") >= 2024))):
        up = sub.filter(pl.col("b7") > -0.03)["net"].to_numpy()
        dn = sub.filter(pl.col("b7") <= -0.03)["net"].to_numpy()
        diffs = [
            rng.choice(up, len(up)).mean() - rng.choice(dn, len(dn)).mean() for _ in range(2000)
        ]
        lo, hi = np.percentile(diffs, [5, 95])
        print(
            f"BTC-7d split bootstrap ret_4h>={thr} {lab}: up n={len(up)} {up.mean() * bp:+.0f}, down n={len(dn)} {dn.mean() * bp:+.0f}, diff 90% CI [{lo * bp:+.0f},{hi * bp:+.0f}] bp"  # noqa: E501
        )
# fade vs b7 continuous: median fade by b7 tercile in 2025-26
for yrs, lab in (((2021, 2024), "2021-24"), ((2025, 2026), "2025-26")):
    s = dd.filter(pl.col("year").is_between(*yrs))
    print(
        lab,
        s.group_by(pl.col("b7").qcut(3, labels=["low", "mid", "high"], allow_duplicates=True))
        .agg(
            n=pl.len(),
            med=pl.col("perp_short").median() * bp,
            net=pl.col("net").mean() * bp,
            sq100=pl.col("sq100").mean(),
        )
        .sort("b7")
        .to_dicts(),
    )

# ------------------------------------------------ 3 squeeze hazard
print(
    "\n=== 3 hazard: P(final adverse reaches +100%) given the pump short is already x against by hour 8 (dedup, ret_4h>=0.25)"  # noqa: E501
)
h = dd.filter(pl.col("ret_4h") >= 0.25)
print("n", h.height, "unconditional sq100", round(float(h["sq100"].mean()), 3))
for xth in (0.1, 0.2, 0.3, 0.5):
    g = h.filter(pl.col("adv8") >= xth)
    gg = h.filter(pl.col("adv8") < xth)
    print(
        f" adverse>={xth:.0%} within 8h: n={g.height} P(sq100)={g['sq100'].mean():.3f} mean final {g['perp_short'].mean() * bp:+.0f}bp P(final profit)={float((g['perp_short'] > 0).mean()):.2f} | others sq100={gg['sq100'].mean():.3f}"  # noqa: E501
    )
print(
    "share of all sq100 events that were already >=+30% adverse within 8h:",
    round(float(((h["adv8"] >= 0.3) & h["sq100"]).sum() / h["sq100"].sum()), 3),
    "; already >= +50%:",
    round(float(((h["adv8"] >= 0.5) & h["sq100"]).sum() / h["sq100"].sum()), 3),
)

# ------------------------------------------------ 4 metric features power
print("\n=== 4 futures metrics AUC for sq100 / sq50 (deduped, ret_4h>=0.25, with metrics)")
m = pl.read_parquet(OWN / "metric_feats.parquet")
mm = dd.filter(pl.col("ret_4h") >= 0.25).join(m, on="row", how="inner")
print(
    "rows with metrics:",
    mm.height,
    "sq100 positives:",
    int(mm["sq100"].sum()),
    "sq50:",
    int(mm["sq50"].sum()),
)
out = []
for c in [c for c in m.columns if c != "row"] + ["rate", "ret_4h"]:
    v = np.nan_to_num(
        mm[c].to_numpy().astype(float), nan=np.nanmedian(mm[c].to_numpy().astype(float))
    )
    out.append(
        {
            "feat": c,
            "auc_sq100": round(float(roc_auc_score(mm["sq100"].to_numpy(), v)), 3),
            "auc_sq50": round(float(roc_auc_score(mm["sq50"].to_numpy(), v)), 3),
        }
    )
print(pl.DataFrame(out))
y = mm["sq100"].to_numpy()
a = [roc_auc_score(y, rng.random(len(y))) for _ in range(500)]
print(
    "null AUC sd for this sample:",
    round(float(np.std(a)), 3),
    "(a feature needs |AUC-0.5| > ~",
    round(1.64 * float(np.std(a)), 2),
    "to be distinguishable)",
)

# ------------------------------------------------ 5 ALPACA
print("\n=== 5 ALPACA 2025-04-29 case: perp vs spot closes (relative to entry)")
a = (
    d.filter(
        (pl.col("symbol") == "ALPACAUSDT")
        & (pl.col("open_time").dt.date() == pl.date(2025, 4, 29))
        & (pl.col("ret_4h") > 0.5)
    )
    .sort("open_time")
    .head(1)
)
r = a["row"].to_numpy()
pp, _ = perp_paths(r)
sp = paths(r)
for hr in (1, 4, 8, 12, 24, 36, 48, 60, 72):
    k = 96 + hr - 9 if hr > 8 else hr * 12 - 1
    print(
        f"  h{hr:>2}: perp close {pp[0, 2, k]:+.2f} high {pp[0, 0, k]:+.2f} | spot close {sp[0, 2, k]:+.2f} high {sp[0, 0, k]:+.2f}"  # noqa: E501
    )
print("event:", a.select("symbol", "open_time", "ret_4h").to_dicts())

# ------------------------------------------------ 6 is the BTC-7d effect a beta to the following market bounce?  # noqa: E501
print(
    "\n=== 6 BTC forward 72h return vs BTC trailing 7d, and the pumped coin's 72h return (deduped, ret_4h>=0.25, 2021-26)"  # noqa: E501
)


def btc_fwd(t, hours):
    k = np.searchsorted(bt, t, side="right") - 1
    j = np.searchsorted(bt, t + hours * 3600, side="right") - 1
    return bc[np.minimum(j, len(bc) - 1)] / bc[np.maximum(k, 0)] - 1


z = dd.filter(pl.col("ret_4h") >= 0.25)
z = z.with_columns(bf=pl.Series(btc_fwd(z["t"].to_numpy(), 72)), coin72=-pl.col("perp_short"))
for lab, sub in (("all", z), ("2024-26", z.filter(pl.col("year") >= 2024))):
    print(
        lab,
        "corr(BTC 7d trailing, BTC fwd72):",
        round(float(sub.select(pl.corr("b7", "bf"))[0, 0]), 3),
        "| corr(BTC fwd72, coin 72h return):",
        round(float(sub.select(pl.corr("bf", "coin72"))[0, 0]), 3),
        "| beta of coin 72h on BTC fwd72:",
        round(float(np.polyfit(sub["bf"].to_numpy(), sub["coin72"].to_numpy(), 1)[0]), 2),
    )
    print(
        "   ",
        sub.group_by(pl.col("b7") > -0.03)
        .agg(
            n=pl.len(),
            btc_fwd72=pl.col("bf").mean(),
            coin72_mean=pl.col("coin72").mean(),
            coin72_med=pl.col("coin72").median(),
        )
        .sort("b7")
        .to_dicts(),
    )
