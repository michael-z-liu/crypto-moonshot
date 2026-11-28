"""W9: why weekly large-cap reversal failed: horizons, regime, trend, dispersion, correlation."""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
from common_ls import CACHE_ALL

os.environ["MOONSHOT_UNIVERSE"] = "all"
sys.path.insert(0, str(CACHE_ALL.parent))
from common import load

COINS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","LINKUSDT","DOTUSDT","LTCUSDT","TRXUSDT","BCHUSDT","NEARUSDT","ATOMUSDT","UNIUSDT","ETCUSDT","FILUSDT","APTUSDT","ARBUSDT","OPUSDT","INJUSDT","SUIUSDT","AAVEUSDT","XLMUSDT","HBARUSDT","ICPUSDT","VETUSDT","ALGOUSDT","SANDUSDT"]  # fmt: skip  # noqa: E501
C = {}
for s in COINS:
    d = load(s).filter(pl.col("open_time").dt.hour().eq(0) & pl.col("open_time").dt.minute().eq(0))
    C[s] = dict(zip(d["open_time"].dt.epoch("s").to_list(), d["open"].to_list(), strict=True))
D = 86400
T0 = int(pl.datetime(2021, 1, 4, time_zone="UTC").dt_epoch if False else 1609718400)  # 2021-01-04
T1 = 1792368000  # 2026-10-19 approx
import datetime as dt  # noqa: E402


def yr(t):
    return dt.datetime.fromtimestamp(t, dt.UTC).year


def ret(s, t, a, b):
    c = C[s]
    x, y = c.get(t + a * D), c.get(t + b * D)
    return y / x - 1 if x and y else None


COST = 0.004


def t_of(x):
    x = np.array(x)
    return (
        float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else float("nan")
    )


print(
    "== 1. horizons: formation F, holding H (non-overlapping, step H), bp/period, k=5 of up to 30"
)
print(
    "F,H,n | losers-basket (gross) | winners-basket (gross) | long-short gross | t(LS) | LS net of 0.8% (both legs)"  # noqa: E501
)
res_h = {}
for F in (1, 3, 7, 14, 28):
    for H in (1, 3, 7, 14, 28):
        LB, WB, LS = [], [], []
        for t in range(T0, T1, H * D):
            p, f = {}, {}
            for s in C:
                a, b = ret(s, t, -F, 0), ret(s, t, 0, H)
                if a is not None and b is not None:
                    p[s], f[s] = a, b
            if len(p) < 15:
                continue
            o = sorted(p, key=p.get)
            lo, hi = o[:5], o[-5:]
            bk = np.mean(list(f.values()))
            lmean = np.mean([f[s] for s in lo])
            w = np.mean([f[s] for s in hi])
            LB.append(lmean - bk)
            WB.append(w - bk)
            LS.append(lmean - w)
        print(
            f"{F:2d},{H:2d},{len(LS):3d} | {np.mean(LB) * 1e4:6.0f} | {np.mean(WB) * 1e4:6.0f} | {np.mean(LS) * 1e4:6.0f} | t {t_of(LS):5.2f} | {(np.mean(LS) - 2 * COST) * 1e4:6.0f}"  # noqa: E501
        )
# weekly base data
rows = []
for t in range(T0, T1, 7 * D):
    p, f, p30, p1 = {}, {}, {}, {}
    for s in C:
        a, b, m = ret(s, t, -7, 0), ret(s, t, 0, 7), ret(s, t, -30, 0)
        if None not in (a, b, m):
            p[s], f[s], p30[s] = a, b, m
    if len(p) < 15:
        continue
    btc_p, btc_f = p.get("BTCUSDT"), f.get("BTCUSDT")
    o = sorted(p, key=p.get)
    lo = o[:5]
    bk = np.mean(list(f.values()))
    rows.append(
        dict(
            t=t,
            year=yr(t),
            n=len(p),
            bk=bk,
            lo=np.mean([f[s] for s in lo]),
            btc_p=btc_p,
            btc_f=btc_f,
            disp=np.std(list(f.values())),
            mad=np.mean(np.abs(np.array(list(f.values())) - bk)),
            lo_mad=np.mean([p[s] for s in lo]) - np.mean(list(p.values())),
            lo30=np.mean([p30[s] for s in lo]),
            lo_f=[f[s] for s in lo],
            lo_30=[p30[s] for s in lo],
            allf=f,
            all30=p30,
            allp=p,
            lo_names=lo,
            fmax=max(f.values()),
            fmin=min(f.values()),
        )
    )
W = pl.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows])
ex = np.array([r["lo"] - r["bk"] for r in rows])
print("\n== 2. regime (losers minus basket, gross bp/week, k=5, 7d/7d)")
bp = np.array([r["btc_p"] for r in rows])
bf = np.array([r["btc_f"] for r in rows])
for nm, mask in [
    ("BTC past week up", bp > 0),
    ("BTC past week down", bp <= 0),
    ("BTC next week up (look-ahead, diagnostic)", bf > 0),
    ("BTC next week down (look-ahead)", bf <= 0),
    ("BTC past week < -5%", bp < -0.05),
    ("BTC past week > +5%", bp > 0.05),
]:
    print(
        f"{nm:45s} n={mask.sum():3d} excess {ex[mask].mean() * 1e4:6.0f} bp t {t_of(ex[mask]):5.2f}; losers abs {np.mean([r['lo'] for r, m in zip(rows, mask, strict=False) if m]) * 1e4:6.0f}; basket {np.mean([r['bk'] for r, m in zip(rows, mask, strict=False) if m]) * 1e4:6.0f}"  # noqa: E501
    )
print(
    "corr(excess, BTC next-week ret) =",
    round(float(np.corrcoef(ex, bf)[0, 1]), 2),
    " beta of losers basket vs basket (weekly):",
    round(float(np.polyfit([r["bk"] for r in rows], [r["lo"] for r in rows], 1)[0]), 2),
)
print(
    "by year excess:",
    {
        y: (
            round(float(ex[W["year"].to_numpy() == y].mean() * 1e4)),
            int((W["year"].to_numpy() == y).sum()),
        )
        for y in sorted(set(W["year"].to_list()))
    },
)

print(
    "\n== 3. losers by 30d trend: of the 5 weekly losers, split each coin by 30d return (rank within the losers) and by absolute 30d < -20%"  # noqa: E501
)
dn, up, ab, bl = [], [], [], []
persist = []
for r in rows:
    bk = r["bk"]
    z = sorted(zip(r["lo_30"], r["lo_f"], strict=False))
    dn += [f - bk for _, f in z[:2]]
    up += [f - bk for _, f in z[-2:]]
    for m, f in zip(r["lo_30"], r["lo_f"], strict=False):
        (ab if m < -0.2 else bl).append(f - bk)
print(
    f"losers w/ worst 30d (2 of 5): {np.mean(dn) * 1e4:6.0f} bp (n {len(dn)}, t {t_of(dn):.2f}); best 30d (2 of 5): {np.mean(up) * 1e4:6.0f} bp (t {t_of(up):.2f})"  # noqa: E501
)
print(
    f"losers with 30d<-20%: {np.mean(ab) * 1e4:6.0f} bp (n {len(ab)}, t {t_of(ab):.2f}); 30d>=-20%: {np.mean(bl) * 1e4:6.0f} bp (n {len(bl)}, t {t_of(bl):.2f})"  # noqa: E501
)
# idiosyncratic: all coin-weeks, regress fwd excess on 7d excess & 30d excess (pooled)
X7, X30, Y = [], [], []
for r in rows:
    m7 = np.mean(list(r["allp"].values()))
    m30 = np.mean(list(r["all30"].values()))
    bk = r["bk"]
    for s in r["allf"]:
        X7.append(r["allp"][s] - m7)
        X30.append(r["all30"][s] - m30 - (r["allp"][s] - m7))
        Y.append(r["allf"][s] - bk)
X = np.c_[np.ones(len(Y)), X7, X30]
b = np.linalg.lstsq(X, np.array(Y), rcond=None)[0]
res = np.array(Y) - X @ b
se = np.sqrt(np.diag(res.var() * np.linalg.inv(X.T @ X)))
print(
    "pooled regression fwd7 excess ~ past7 excess + prior (days8-30) excess: coef",
    np.round(b[1:], 3),
    "naive t",
    np.round(b[1:] / se[1:], 1),
    "(overlap-free weekly, but cross-section correlated: t overstated)",
)
print("pooled coin-week fwd excess vs 7d excess corr:", round(float(np.corrcoef(X7, Y)[0, 1]), 3))
# worst single names among loser selections: coin contributions
from collections import defaultdict  # noqa: E402

cc = defaultdict(list)
for r in rows:
    for s, f in zip(r["lo_names"], r["lo_f"], strict=False):
        cc[s].append(f - r["bk"])
top = sorted(cc.items(), key=lambda kv: -len(kv[1]))[:8]
print(
    "most frequent losers (name, weeks, mean excess bp):",
    [(s[:-4], len(v), round(np.mean(v) * 1e4)) for s, v in top],
)
# newly listed? count coins available per year
print(
    "coins per week by year:",
    {
        y: round(float(np.mean([r["n"] for r in rows if r["year"] == y])), 1)
        for y in sorted(set(W["year"].to_list()))
    },
)

print("\n== 4. dispersion across coins of weekly fwd returns vs 0.4% cost")
print(
    "year: xs std of 7d fwd ret | mean abs dev | mean (bottom5 past-loser mean past ret - avg) | basket 7d ret std | spread(top5-bot5 fwd, ex-post max) "  # noqa: E501
)
for y in sorted(set(W["year"].to_list())):
    rr = [r for r in rows if r["year"] == y]
    ex_post = []
    for r in rr:
        v = sorted(r["allf"].values())
        ex_post.append(np.mean(v[-5:]) - np.mean(v[:5]))
    print(
        f"{y}: std {np.mean([r['disp'] for r in rr]) * 100:5.1f}% | MAD {np.mean([r['mad'] for r in rr]) * 100:5.1f}% | loser past gap {np.mean([r['lo_mad'] for r in rr]) * 100:6.1f}% | basket std {np.std([r['bk'] for r in rr]) * 100:5.1f}% | ex-post top5-bot5 {np.mean(ex_post) * 100:5.1f}%"  # noqa: E501
    )
allsd = W["disp"].mean()
print(
    f"all: xs std {allsd * 100:.1f}%/week; a perfect-foresight rank would earn ~{np.mean([np.mean(sorted(r['allf'].values())[-5:]) - np.mean(sorted(r['allf'].values())[:5]) for r in rows]) * 100:.1f}%/wk; costs 0.4% = {COST / allsd * 100:.1f}% of one xs std"  # noqa: E501
)
# predictive power: rank IC of 7d past vs 7d fwd
ics = []
from scipy.stats import spearmanr  # noqa: E402

for r in rows:
    k = list(r["allp"])
    ics.append(spearmanr([r["allp"][s] for s in k], [r["allf"][s] for s in k])[0])
ics = np.array(ics)
print(
    f"weekly rank IC (past7 vs fwd7): mean {ics.mean():.3f}, t {t_of(ics):.2f}; by year",
    {
        y: round(float(ics[W["year"].to_numpy() == y].mean()), 3)
        for y in sorted(set(W["year"].to_list()))
    },
)
print(
    f"implied: IC {ics.mean():.3f} * xs std {allsd * 100:.1f}% * ~1.2 (top/bottom-5 score of 30) ~ {-ics.mean() * allsd * 1.17 * 1e4:.0f} bp/wk reversal edge on k=5 vs cost 40 bp"  # noqa: E501
)

print("\n== 5. correlation: avg pairwise corr of daily and weekly returns by year")
days = list(range(T0, T1, D))
for y in range(2021, 2027):
    ds = [t for t in days if yr(t) == y]
    M = []
    for s in C:
        r = [
            (C[s].get(t + D) / C[s].get(t) - 1) if C[s].get(t) and C[s].get(t + D) else np.nan
            for t in ds
        ]
        M.append(r)
    M = np.array(M)
    ok = np.isfinite(M).mean(1) > 0.9
    M = M[ok]
    M = M[:, np.isfinite(M).all(0)]
    cm = np.corrcoef(M)
    iu = np.triu_indices(len(M), 1)
    # weekly
    ws = [t for t in range(T0, T1, 7 * D) if yr(t) == y]
    Wk = np.array(
        [
            [
                (C[s].get(t + 7 * D) / C[s].get(t) - 1)
                if C[s].get(t) and C[s].get(t + 7 * D)
                else np.nan
                for t in ws
            ]
            for s in C
        ]
    )[ok]
    Wk = Wk[:, np.isfinite(Wk).all(0)]
    cw = np.corrcoef(Wk) if Wk.shape[1] > 5 else None
    # share of variance explained by first PC (daily)
    ev = np.linalg.eigvalsh(cm)[-1] / len(M)
    print(
        f"{y}: coins {ok.sum()}, daily avg pairwise corr {cm[iu].mean():.2f}, weekly {cw[iu].mean() if cw is not None else float('nan'):.2f} (n weeks {Wk.shape[1]}), PC1 share {ev * 100:.0f}%"  # noqa: E501
    )
# idiosyncratic share of weekly xs variance: xs var vs total
tot = np.var([x for r in rows for x in r["allf"].values()])
idio = np.mean([np.var(list(r["allf"].values())) for r in rows])
print(
    f"weekly pooled var {tot:.4f} vs mean xs var {idio:.4f}: market share {(1 - idio / tot) * 100:.0f}%"  # noqa: E501
)
