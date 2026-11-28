"""w11: why long pumps at negative funding fail (diagnosis). Unconstrained trades, perp, 1x, 30 bp cost."""  # noqa: E501

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN
from diag import annotate, calibration, scorecard, table
from sim import Rule, simulate

KIND = {0: "time", 1: "stop", 2: "tp", 3: "liq", 4: "stop"}
F = pl.read_parquet(OWN / "feats.parquet").filter(
    pl.col("perp")
    & (pl.col("qv_base") >= 20_000)
    & (pl.col("year") >= 2022)
    & (pl.col("ret_4h") >= 0.15)
)
F = F.join(pl.read_parquet(OWN / "metric_feats.parquet"), on="row", how="left")


def run(hold, tp=None, pool=None, side=1):
    c = F if pool is None else F.filter(pool)
    cand = c.select("row", side=pl.lit(side), prio="ret_4h")
    r = Rule(hold_min=hold, tp=tp, slots=100_000, cost=0.003, liq_frac=None)
    t = simulate(cand, r, start_equity=1e12, perp_funding=True, source="perp").trades
    t = t.with_columns(exit_kind=pl.col("kind").replace_strict(KIND, default="time"))
    return t.join(F.drop("t", "row_x", strict=False), on="row", how="left", suffix="_f")


T = pl.concat(
    [
        run(4320, None, (pl.col("ret_4h") >= 0.2) & (pl.col("rate") < 0)),
        run(4320, None, (pl.col("ret_4h") >= 0.2) & ~(pl.col("rate") < 0)),
    ]
).unique("row")
T4 = pl.concat(
    [
        run(4320, None, (pl.col("ret_4h") >= 0.4) & (pl.col("rate") < 0)),
        run(4320, None, (pl.col("ret_4h") >= 0.4) & ~(pl.col("rate") < 0)),
    ]
).unique("row")
T = T.with_columns(pr=pl.col("ret"), fr=-pl.col("fund"))
T4 = T4.with_columns(pr=pl.col("ret"), fr=-pl.col("fund"))  # price return, funding received
neg = pl.col("rate") < 0


def stats(d, label):
    q = d["pr"].quantile
    return (
        f"| {label} | {d.height} | {q(0.05):+.2f} | {q(0.25):+.2f} | {q(0.5):+.2f} | {q(0.75):+.2f} | {q(0.95):+.2f} | "  # noqa: E501
        f"{(d['pr'] > 0.3).mean():.1%} | {(d['pr'] < -0.3).mean():.1%} | {d['pr'].mean() * 1e4:+.0f} | {d['fr'].mean() * 1e4:+.0f} | {d['net'].mean() * 1e4:+.0f} |"  # noqa: E501
    )


H = (
    "| group | n | q05 | q25 | q50 | q75 | q95 | >+30% | <-30% | mean price bp | funding recv bp | net bp |\n|"  # noqa: E501
    + "---|" * 12
)
out = []
for thr in (0.2, 0.4):
    out.append(f"\n### 72h price-return distribution, ret_4h >= {thr}, long, no TP\n\n" + H)
    S = (T if thr == 0.2 else T4).filter(pl.col("ret_4h") >= thr)
    for lab, e in (("rate<0", neg), ("rate>=0", ~neg)):
        out.append(stats(S.filter(e), lab + " all"))
    for y in range(2022, 2027):
        for lab, e in (("rate<0", neg), ("rate>=0", ~neg)):
            s = S.filter(e & (pl.col("year") == y))
            if s.height:
                out.append(stats(s, f"{lab} {y}"))
# funding rate size
S = T.filter((pl.col("ret_4h") >= 0.2) & neg)
out.append(
    f"\nrate<0 & r4>=.2: median last rate {S['rate'].median() * 1e4:+.1f} bp/8h, mean {S['rate'].mean() * 1e4:+.1f}; "  # noqa: E501
    f"share rate < -5bp {(S['rate'] < -5e-4).mean():.0%}; funding received over 72h mean {S['fr'].mean() * 1e4:+.0f} bp, median {S['fr'].median() * 1e4:+.0f} bp; "  # noqa: E501
    f"price-return mean {S['pr'].mean() * 1e4:+.0f} bp, median {S['pr'].median() * 1e4:+.0f}. Share of trades where funding received > 0: {(S['fr'] > 0).mean():.0%}"  # noqa: E501
)
# 2025 vs 2026
out.append(
    "\n### Year anatomy (r4>=.2, rate<0)\n\n| year | n | price mean bp | median bp | >+30% | <-30% | fund recv bp | net bp | mkt_r24h mean | rate mean bp | share rate<0 among pumps | continuing (ret>0) |\n|"  # noqa: E501
    + "---|" * 12
)
A = T.filter(pl.col("ret_4h") >= 0.2)
for y in range(2022, 2027):
    s = A.filter(neg & (pl.col("year") == y))
    a = A.filter(pl.col("year") == y)
    out.append(
        f"| {y} | {s.height} | {s['pr'].mean() * 1e4:+.0f} | {s['pr'].median() * 1e4:+.0f} | {(s['pr'] > 0.3).mean():.1%} | {(s['pr'] < -0.3).mean():.1%} | {s['fr'].mean() * 1e4:+.0f} | {s['net'].mean() * 1e4:+.0f} | {s['mkt_r24h'].mean():+.3f} | {s['rate'].mean() * 1e4:+.1f} | {s.height / a.height:.0%} | {(s['pr'] > 0).mean():.0%} |"  # noqa: E501
    )
# monthly 2025/26 for r4>=.2 rate<0
out.append("\n2025-26 by half-year (r4>=.2, rate<0): price mean / n / share>+30% / share<-30%\n")
A2 = A.filter(neg & (pl.col("year") >= 2025)).with_columns(
    h=pl.col("open_time").dt.year().cast(pl.String)
    + "H"
    + ((pl.col("open_time").dt.month() > 6).cast(pl.Int8) + 1).cast(pl.String)
)
for h, s in A2.group_by("h", maintain_order=False):
    out.append(
        f"- {h[0]}: n={s.height}, price {s['pr'].mean() * 1e4:+.0f} bp, median {s['pr'].median() * 1e4:+.0f}, >+30% {(s['pr'] > 0.3).mean():.0%}, <-30% {(s['pr'] < -0.3).mean():.0%}, net {s['net'].mean() * 1e4:+.0f}"  # noqa: E501
    )
# clustering: pumps are on the same days?
d = A.filter(neg).group_by("day").agg(n=pl.len(), m=pl.col("pr").mean())
out.append(
    f"\nr4>=.2 & rate<0: {A.filter(neg).height} trades on {d.height} days; share of trades on days with >=3 such pumps {d.filter(pl.col('n') >= 3)['n'].sum() / d['n'].sum():.0%}."  # noqa: E501
)

# feature screen on rate<0 & r4>=.2: spearman with price return, 2022-24 vs 2025-26
feats = [
    "ret_4h",
    "ret_24h",
    "rate",
    "rate24",
    "off_high4h",
    "off_low4h",
    "wick_up",
    "taker",
    "taker4h",
    "vol_mult",
    "vol4h",
    "n_hot_bars4h",
    "listed_days",
    "qv_base",
    "r7d",
    "r30d",
    "dhi30d",
    "vol30d",
    "n_prior_pumps30",
    "oi_ch4h",
    "oi_ch24h",
    "oi_to_vol",
    "top_ls_acc",
    "top_ls_pos",
    "ls_acc",
    "taker_ls",
    "ls_acc_ch4h",
    "mkt_r4h",
    "mkt_r24h",
    "btc_ret_4h",
    "hour",
]  # y24h/y72h excluded: forward labels
B = A.filter(neg)
out.append(
    "\n### Entry-time features vs 72h price return (r4>=.2, rate<0); Spearman, 2022-24 | 2025-26 | all; and AUC for squeeze (>+30%) vs collapse (<-30%)\n\n| feature | rho 22-24 | rho 25-26 | rho all | AUC squeeze-vs-collapse |\n|---|---|---|---|---|"  # noqa: E501
)


def rho(d, f):
    d = d.select(f, "pr").drop_nulls().drop_nans()
    return (
        float(d.select(pl.corr(pl.col(f).rank(), pl.col("pr").rank())).item())
        if d.height > 20
        else float("nan")
    )


def auc(d, f):
    d = (
        d.filter((pl.col("pr") > 0.3) | (pl.col("pr") < -0.3))
        .select(f, "pr")
        .drop_nulls()
        .drop_nans()
    )
    if d.height < 20:
        return float("nan")
    r = d.with_columns(r=pl.col(f).rank())
    pos = r.filter(pl.col("pr") > 0.3)
    n1 = pos.height
    n0 = r.height - n1
    return float((pos["r"].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


rs = []
for f in feats:
    rs.append(
        (
            f,
            rho(B.filter(pl.col("year") <= 2024), f),
            rho(B.filter(pl.col("year") >= 2025), f),
            rho(B, f),
            auc(B, f),
        )
    )
for f, a, b, c, u in sorted(rs, key=lambda x: -abs(x[3]) if x[3] == x[3] else 0):
    out.append(f"| {f} | {a:+.2f} | {b:+.2f} | {c:+.2f} | {u:.2f} |")

# signal: oriented percentile of best features chosen on 2022-24 only (stable-sign ones), tested 2025-26  # noqa: E501
cand_f = [
    (f, a, b)
    for f, a, b, c, u in rs
    if a == a and b == b and abs(a) > 0.08 and np.sign(a) == np.sign(b)
]
out.append(
    "\nFeatures with |rho 22-24| > .08 and same sign in 25-26: "
    + (", ".join(f"{f} ({a:+.2f}/{b:+.2f})" for f, a, b in cand_f) or "none")
)


def make_signal(d):
    if not cand_f:
        return pl.Series([0.5] * d.height)
    cols = [
        (d[f].fill_null(d[f].median()).fill_nan(d[f].median()).rank() / d.height)
        * (1 if a > 0 else -1)
        + (0 if a > 0 else 1)
        for f, a, b in cand_f
    ]
    s = sum(cols) / len(cols)
    return s


SIG = make_signal(B)
B = B.with_columns(sig=SIG.rank() / B.height)
for lab, d in (
    ("22-24", B.filter(pl.col("year") <= 2024)),
    ("25-26", B.filter(pl.col("year") >= 2025)),
):
    out.append(
        f"signal vs price return {lab}: Spearman {float(d.select(pl.corr(pl.col('sig').rank(), pl.col('pr').rank())).item()):+.3f}; top-half price mean {d.filter(pl.col('sig') >= 0.5)['pr'].mean() * 1e4:+.0f} bp, bottom-half {d.filter(pl.col('sig') < 0.5)['pr'].mean() * 1e4:+.0f}; top-half net {d.filter(pl.col('sig') >= 0.5)['net'].mean() * 1e4:+.0f}"  # noqa: E501
    )

# grid of long variants -> best; scorecard
out.append(
    "\n### Variant grid (unconstrained; net bp, years = mean net by year 22/23/24/25/26)\n\n| r4 min | rate | hold | tp | n | net bp | median bp | years bp |\n|---|---|---|---|---|---|---|---|"  # noqa: E501
)
best = None
res = {}
for hold in (1440, 4320):
    for tp in (None, 0.3):
        for thr in (0.2, 0.3, 0.4):
            for rl, e in (("<0", pl.col("rate") < 0), ("<-5bp", pl.col("rate") < -5e-4)):
                s = run(hold, tp, (pl.col("ret_4h") >= thr) & e)
                if s.height < 60:
                    continue
                res[(hold, tp, thr, rl)] = s
                ys = [s.filter(pl.col("year") == y)["net"].mean() for y in range(2022, 2027)]
                ytxt = "/".join(f"{v * 1e4:+.0f}" if v is not None else "na" for v in ys)
                out.append(
                    f"| {thr} | {rl} | {hold // 60}h | {tp} | {s.height} | {s['net'].mean() * 1e4:+.0f} | {s['net'].median() * 1e4:+.0f} | {ytxt} |"  # noqa: E501
                )
                pos = sum(1 for v in ys if v is not None and v > 0)
                key = (pos, s["net"].mean())
                if best is None or key > best[0]:
                    best = (key, hold, tp, thr, rl, e)
_, hold, tp, thr, rl, e = best
out.append(f"\nBest by (positive years, mean net): r4>={thr}, rate{rl}, {hold // 60}h, tp={tp}.")
t = res[(hold, tp, thr, rl)]


def fr(d):
    s = d["ret_4h"].rank() / d.height
    return d.select(
        signal=s,
        threshold=pl.lit(float(univ.filter(pl.col("ret_4h") >= thr)["ret_4h"].rank().min())),
        net="net",
        mfe="mfe",
        mae="adv",
        tp=pl.lit(tp, dtype=pl.Float64),
        stop=pl.lit(None, dtype=pl.Float64),
        exit_kind="exit_kind",
        year="year",
    )


# signal = oriented feature composite if available (percentile within the negative-funding pool), else ret_4h percentile  # noqa: E501
t = t.with_columns(sg=make_signal(t)).with_columns(sg=pl.col("sg").rank() / t.height)
ent = t.with_columns(signal=pl.col("sg"), threshold=pl.lit(0.5)).select(
    "signal",
    "threshold",
    "net",
    "mfe",
    pl.col("adv").alias("mae"),
    tp=pl.lit(tp, dtype=pl.Float64),
    stop=pl.lit(None, dtype=pl.Float64),
    exit_kind="exit_kind",
    year="year",
)
card = {f"long r4>={thr} rate{rl} {hold // 60}h tp={tp}": scorecard(annotate(ent))}
out.append(
    "\n### Scorecard (diag.py), best long variant (signal = oriented-percentile composite of the stable-sign features, pool percentile, theta = 0.5 (top half))\n\n"  # noqa: E501
    + table(card)
)
h = ent.with_columns(y=t["year"]).filter(pl.col("year") >= 2025).drop("year").rename({"y": "year"})
out.append(
    "Holdout 2025-26 only (features chosen on 2022-24): "
    + str(
        {
            k: round(v, 3)
            for k, v in scorecard(annotate(h)).items()
            if k in ("n", "E_o", "median_o", "win")
        }
    )
    + "; top half: "
    + str(
        {
            k: round(v, 3)
            for k, v in scorecard(annotate(h.filter(pl.col("signal") >= 0.5))).items()
            if k in ("n", "E_o", "median_o", "win")
        }
    )
)
out.append(
    "Note: E_o clips net at +-1, so squeezes above +100% are truncated; the unclipped mean is "
    + f"{t['net'].mean() * 1e4:+.0f} bp, share net > +100%: {(t['net'] > 1).mean():.1%}"
)
out.append(
    "Calibration by composite-signal decile (E_o, win, p_tp, p_stopped):\n\n```\n"
    + str(calibration(ent, None))
    + "\n```"
)
# reference: pump short at rate<0 and rate>=0 same pool, 72h
Ts = pl.concat(
    [
        simulate(
            F.filter((pl.col("ret_4h") >= 0.2) & q).select("row", side=pl.lit(-1), prio="ret_4h"),
            Rule(hold_min=4320, sl=1.0, slots=100_000, cost=0.003, liq_frac=None),
            start_equity=1e12,
            perp_funding=True,
            source="perp",
        ).trades
        for q in (pl.col("rate") < 0, ~(pl.col("rate") < 0))
    ]
).join(F.select("row", "rate"), on="row")
for lab, e2 in (("rate<0", pl.col("rate") < 0), ("rate>=0", pl.col("rate") >= 0)):
    s = Ts.filter(e2)
    out.append(
        f"short ref r4>=.2 {lab}: n={s.height}, net {s['net'].mean() * 1e4:+.0f} bp (long same pool above)"  # noqa: E501
    )
long_all = T.filter(pl.col("ret_4h") >= 0.2)
for lab, e2 in (("rate<0", neg), ("rate>=0", ~neg)):
    s = long_all.filter(e2)
    out.append(f"long ref r4>=.2 {lab}: n={s.height}, net {s['net'].mean() * 1e4:+.0f} bp")
print("\n".join(out))
