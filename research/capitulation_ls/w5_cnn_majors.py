"""W5d: (1) CNN flush scores during BTC 3-day-drop episodes; (2) majors vs alts bounce."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, paths, simulate

m = meta()
t = m["open_time"].dt.epoch("s").to_numpy()
pct = scores("day5")
E = pl.read_parquet("/tmp/w5_episodes.parquet").filter(pl.col("thr") == -0.08)
hours = np.load("/tmp/w5_btc_hours.npy")
sig = np.isfinite(pct) & (pct >= 0.995)
print(
    "== CNN day5 >= 0.995 rows inside each -8% BTC episode window [-72h, +24h] (scored 2022-26 to 2026-11-01) =="  # noqa: E501
)
rows = []
for r in E.iter_rows(named=True):
    te = int(np.datetime64(r["t"]).astype("datetime64[s]").astype(int))
    w = (t >= te - 72 * 3600) & (t <= te + 24 * 3600)
    rows.append(
        dict(
            t=r["t"],
            d3=r["d3"],
            n_sig=int((sig & w).sum()),
            n_scored=int((np.isfinite(pct) & w).sum()),
            a7=r["a7"],
            b7=r["b7"],
        )
    )
X = pl.DataFrame(rows).filter(pl.col("n_scored") > 0)
print(
    "episodes scored",
    X.height,
    " with >=1 signal row",
    int((X["n_sig"] > 0).sum()),
    " with >=10",
    int((X["n_sig"] >= 10).sum()),
    " total sig rows",
    int(X["n_sig"].sum()),
)
for lab, f in (
    ("n_sig>=10 (flush)", pl.col("n_sig") >= 10),
    ("1<=n_sig<10", (pl.col("n_sig") >= 1) & (pl.col("n_sig") < 10)),
    ("n_sig==0", pl.col("n_sig") == 0),
):
    y = X.filter(f)
    if y.height:
        print(
            f"{lab:20s} n={y.height:>2} basket fwd7 mean {y['a7'].mean() * 100:+.1f}% BTC fwd7 {y['b7'].mean() * 100:+.1f}%  share a7>0 {float((y['a7'] > 0).mean()):.2f}"  # noqa: E501
        )
print(
    "episodes with no signal:",
    X.filter(pl.col("n_sig") == 0).select("t", "d3", "a7").to_dicts()[:30],
)
# base rate of signal days overall: per-day count of signal rows
day = t // 86400
cnt = {}
for d_ in day[sig]:
    cnt[d_] = cnt.get(d_, 0) + 1
v = np.array(sorted(cnt.values(), reverse=True))
print(
    "days with any pct>=0.995 signal:",
    len(v),
    " days with >=30:",
    int((v >= 30).sum()),
    " share of rows on top 11 days:",
    round(float(v[:11].sum() / v.sum()), 2),
    " row/day median",
    np.median(v),
)
# 5m volume spike intensity at the slide's worst day vs the median flush day cannot be scored after 2026-11-01.  # noqa: E501

print("\n== Majors vs alts after crashes: ret_4h <= x (2022+), perp-free spot paths ==")
MAJ = {
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "TRXUSDT",
    "BCHUSDT",
    "NEARUSDT",
    "MATICUSDT",
    "ATOMUSDT",
    "UNIUSDT",
    "ETCUSDT",
    "FILUSDT",
    "APTUSDT",
}
sym = m["symbol"].to_numpy()
yr = m["year"].to_numpy()
r4 = m["ret_4h"].to_numpy()
b4 = m["btc_ret_4h"].to_numpy()
qv = m["qv_base"].to_numpy()
ism = np.isin(sym, list(MAJ))
ok = (yr >= 2022) & (r4 <= -0.08)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(30)
out = []
for lab, g in (("majors", ism), ("alts", ~ism)):
    for lo, hi in ((-0.10, -0.08), (-0.15, -0.10), (-0.25, -0.15), (-1, -0.25)):
        idx = np.flatnonzero(ok & g & (r4 > lo) & (r4 <= hi))
        if len(idx) < 5:
            continue
        P = paths(
            idx
        )  # [n,3,160]: hi, lo, close vs entry; 5m steps first 96 (8h), then hourly to 72h
        c24 = P[:, 2, 96 + 15]  # close after 24h (hour index 24 -> step 96+15)
        mx24 = P[:, 0, : 96 + 16].max(1)
        mn24 = P[:, 1, : 96 + 16].min(1)
        c72 = P[:, 2, -1]
        # beta proxy: crash vs BTC move in the same 4h
        out.append(
            dict(
                grp=lab,
                r4=f"{lo * 100:.0f}..{hi * 100:.0f}%",
                n=len(idx),
                coins=len(set(sym[idx])),
                days=len(set(t[idx] // 86400)),
                mean_r4=round(float(r4[idx].mean()) * 100, 1),
                btc4=round(float(b4[idx].mean()) * 100, 2),
                med_qv_day_musd=round(float(np.median(qv[idx])) * 288 / 1e6, 2),
                c24=round(float(c24.mean()) * 100, 2),
                c24med=round(float(np.median(c24)) * 100, 2),
                mx24=round(float(mx24.mean()) * 100, 1),
                mn24=round(float(mn24.mean()) * 100, 1),
                tp10=round(float((mx24 >= 0.10).mean()), 2),
                c72=round(float(c72.mean()) * 100, 2),
            )
        )
print(pl.DataFrame(out))
# long-sleeve-style rule on both groups with identical rule: r4<=-10%, TP10, 24h, 1x, cost 0.3%
print("\nSame rule (r4<=-10%, TP 10%, 24h, cost 0.3%, 1 position/coin, no slot limit):")
for lab, g in (
    ("majors", ism),
    ("alts all", ~ism),
    ("alts qv_day>=20M$", (~ism) & (qv * 288 >= 20e6)),
    ("alts qv_day>=5M$", (~ism) & (qv * 288 >= 5e6)),
):
    idx = np.flatnonzero((yr >= 2022) & (r4 <= -0.10) & g)
    b = simulate(
        pl.DataFrame({"row": idx, "side": 1, "prio": -r4[idx]}),
        Rule(hold_min=1440, tp=0.10, slots=100_000, cost=0.003, liq_frac=None),
        start_equity=1e12,
    )
    x = b.trades
    print(
        f"{lab:20s} n={x.height:>4} mean {x['net'].mean() * 1e4:+5.0f}bp med {x['net'].median() * 1e4:+5.0f} win {float((x['net'] > 0).mean()):.2f} p5 {x['net'].quantile(0.05) * 1e4:+5.0f} | by year",  # noqa: E501
        x.group_by("year")
        .agg(n=pl.len(), bp=(pl.col("net").mean() * 1e4).round())
        .sort("year")
        .to_dicts(),
    )
# CNN-scored subset: same pct threshold in both groups
for q in (0.98, 0.995):
    for lab, g in (("majors", ism), ("alts", ~ism)):
        idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= q) & g)
        if len(idx) < 5:
            continue
        b = simulate(
            pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]}),
            Rule(hold_min=4320, tp=0.20, slots=100_000, cost=0.004, liq_frac=None),
            start_equity=1e12,
        )
        x = b.trades
        print(
            f"day5>={q} {lab:7s} TP20/72h n={x.height:>5} mean {x['net'].mean() * 1e4:+5.0f}bp med {x['net'].median() * 1e4:+5.0f} mean r4 {r4[idx].mean() * 100:.1f}%  med r4 {np.median(r4[idx]) * 100:.1f}%"  # noqa: E501
        )
# crash depth distribution over all spikes: share of coins by tier
idx = np.flatnonzero((yr >= 2022) & (r4 <= -0.10))
print(
    "\ncrash rows r4<=-10%: majors",
    int(ism[idx].sum()),
    "alts",
    int((~ism[idx]).sum()),
    " median r4 majors",
    round(float(np.median(r4[idx][ism[idx]])) * 100, 1),
    "alts",
    round(float(np.median(r4[idx][~ism[idx]])) * 100, 1),
    " p10 depth majors",
    round(float(np.percentile(r4[idx][ism[idx]], 10)) * 100, 1),
    "alts",
    round(float(np.percentile(r4[idx][~ism[idx]], 10)) * 100, 1),
)
