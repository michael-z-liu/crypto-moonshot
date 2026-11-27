"""w4 B: average post-listing path by hour, volume decay, young-coin decay, BTC regime."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN
from e13_listing_book import L, perp
from e18_perp_listings import perp as perp18
from futures_ls import spot_name

pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
D = pl.read_parquet(OWN / "w4_spot_trades.parquet")
weak_by = dict(zip(D["symbol"].to_list(), D["weak"].to_list(), strict=False))
HRS = [1, 6, 12, 24, 36, 48, 60, 72, 84, 96, 120, 168, 240, 336, 504, 720, 1000]
NH = 1001


def grid(a: tuple, t0: int) -> np.ndarray | None:
    h, o = a[0], a[1]
    base = np.searchsorted(h, t0)
    if base >= len(h) - NH or h[base] - t0 > 3600 * 6:
        return None
    idx = np.searchsorted(h, h[base] + np.arange(NH) * 3600)
    ok = h[np.minimum(idx, len(h) - 1)] == h[base] + np.arange(NH) * 3600
    out = np.where(ok, o[np.minimum(idx, len(h) - 1)], np.nan)
    return out


# ---- paths: spot listings (perp present at listing), base = perp open at listing hour
paths, syms, yrs = [], [], []
for r in L.iter_rows(named=True):
    a = perp.get(r["symbol"])
    if a is None:
        continue
    t0 = int(r["listed"].timestamp())
    if a[0][0] > t0 + 3600:
        continue
    g = grid(a, t0)
    if g is None or not np.isfinite(g[0]):
        continue
    paths.append(g / g[0])
    syms.append(r["symbol"])
    yrs.append(r["listed"].year)
P = np.array(paths)
print("spot listings with perp at listing & >=1000h:", len(P))
lp = np.log(P)
print("hour | mean log | median log | share below 1 | p25 | p75")
for hr in HRS:
    x = lp[:, hr]
    x = x[np.isfinite(x)]
    print(
        f"{hr:5d} {x.mean():+.3f} {np.median(x):+.3f} {(x < 0).mean():.2f} {np.percentile(x, 25):+.3f} {np.percentile(x, 75):+.3f}"  # noqa: E501
    )
# running local extrema: hour of max in first 168h, hour of min in first 720h
hmax = np.nanargmax(P[:, :169], axis=1)
hmin = np.nanargmin(P[:, :721], axis=1)
print(
    "hour of max within first 168h (24h bins):",
    np.histogram(hmax, bins=[0, 24, 48, 72, 96, 120, 144, 169])[0],
)
print(
    "hour of min within first 720h (7d bins):",
    np.histogram(hmin, bins=[0, 24, 72, 168, 336, 504, 721])[0],
)
# hourly median drift per 12h window (is there a bounce around 48-96h?)
print("median log change per 12h window:")
for a0 in range(0, 240, 12):
    d = lp[:, a0 + 12] - lp[:, a0]
    d = d[np.isfinite(d)]
    print(
        f"  {a0:3d}-{a0 + 12:3d}h mean {d.mean():+.4f} median {np.median(d):+.4f} up-share {(d > 0).mean():.2f}"  # noqa: E501
    )

# weak vs strong at 72h: forward path from hour 72 (rebased)
W = np.array([weak_by.get(s) is True for s in syms])
S = np.array([weak_by.get(s) is False for s in syms])
print(f"\nrebased to hour 72 (n weak {W.sum()}, strong {S.sum()}) mean/median log:")
for hr in (96, 120, 168, 240, 336, 504, 720, 792):
    for lab, m in (("weak", W), ("strong", S)):
        x = lp[m, hr] - lp[m, 72]
        x = x[np.isfinite(x)]
        print(
            f"  h{hr - 72:4d}d {lab:6s} n={len(x):3d} mean {x.mean():+.3f} median {np.median(x):+.3f} p25 {np.percentile(x, 25):+.3f} p75 {np.percentile(x, 75):+.3f}"  # noqa: E501
        )
print("path BEFORE hour 72 by weak/strong (log vs listing): max up to h72 / h72 value")
for lab, m in (("weak", W), ("strong", S)):
    mx = np.nanmax(lp[m, :73], axis=1)
    print(
        f"  {lab}: median peak {np.median(mx):+.3f}, median h72 {np.nanmedian(lp[m, 72]):+.3f}, median hour-of-peak {np.median(np.nanargmax(lp[m, :73], axis=1))}"  # noqa: E501
    )

# ---- unconditional fixed 30d short return by entry delay (no TP/SL), split weak at entry
print(
    "\nunconditional 30d short log-return (-log) by entry delay, all spot listings w/ perp at listing:"  # noqa: E501
)
for dly in (6, 12, 24, 36, 48, 60, 72, 84, 96, 120, 144, 168):
    x = -(lp[:, dly + 720] - lp[:, dly])
    wk = lp[:, dly] <= 0
    ok = np.isfinite(x)
    print(
        f"  d{dly:3d}: n={ok.sum()} mean {np.nanmean(x):+.3f} median {np.nanmedian(x):+.3f} | weak@entry n={np.sum(ok & wk)} mean {np.nanmean(x[wk]):+.3f} med {np.nanmedian(x[wk]):+.3f} | strong n={np.sum(ok & ~wk)} mean {np.nanmean(x[~wk]):+.3f} med {np.nanmedian(x[~wk]):+.3f}"  # noqa: E501
    )

# ---- volume decay: perp quote volume per day relative to day 1, weak vs strong
vol = {}
for r in L.iter_rows(named=True):
    a = perp.get(r["symbol"])
    if a is None or r["symbol"] not in weak_by:
        continue
    t0 = int(r["listed"].timestamp())
    base = np.searchsorted(a[0], t0)
    if base + 24 * 31 >= len(a[0]):
        continue
    q = a[5][base : base + 24 * 31]
    days = q.reshape(31, 24).sum(1)
    vol[r["symbol"]] = days
print("\nperp quote volume per day / day-1 volume (median), weak vs strong at 72h")
for dday in (0, 1, 2, 3, 4, 6, 9, 14, 20, 27, 30):
    row = []
    for lab, flag in (("weak", True), ("strong", False)):
        v = [x[dday] / x[0] for s, x in vol.items() if weak_by[s] is flag and x[0] > 0]
        row.append(f"{lab} {np.median(v):.2f} (n={len(v)})")
    print(f"  day {dday + 1:2d}: " + " | ".join(row))


# ---- young decay: weekly 7d short return by age week, spot-listed coins vs old perps
def weekly(a: tuple, t0: int | None, ages: range, weak_only: bool | None) -> list[float]:
    h, o, hi, _lo, c, qv = a
    out = []
    base = np.searchsorted(h, t0) if t0 is not None else 0
    p0 = o[base]
    for wk in ages:
        i = np.searchsorted(h, h[base] + wk * 7 * 86400)
        j = i + 168
        if j >= len(h) or i < 1 or qv[i - 1] < 50_000 or h[j] - h[i] != 168 * 3600:
            out.append(np.nan)
            continue
        if weak_only is True and o[i] > p0:
            out.append(np.nan)
            continue
        if weak_only is False and o[i] <= p0:
            out.append(np.nan)
            continue
        up = hi[i : j + 1].max() / o[i] - 1
        out.append(-1.0 if up >= 1.0 else -(c[j] / o[i] - 1) - 0.003)
    return out


ages = range(1, 27)
res = {k: [] for k in ("all", "weak", "strong")}
for r in L.iter_rows(named=True):
    a = perp.get(r["symbol"])
    if a is None:
        continue
    t0 = int(r["listed"].timestamp())
    if a[0][0] > t0 + 3600:
        continue
    for k, w in (("all", None), ("weak", True), ("strong", False)):
        res[k].append(weekly(a, t0, ages, w))
print("\nweekly 7d short return (net 0.3% cost, no funding) by age week of spot-listed coins")
print("age_wk | all: n mean median | weak(<=list price): n mean | strong: n mean")
for k, wk in enumerate(ages):
    s = []
    for key in ("all", "weak", "strong"):
        x = np.array(res[key])[:, k]
        x = x[np.isfinite(x)]
        s.append((len(x), x.mean() if len(x) else np.nan, np.median(x) if len(x) else np.nan))
    print(
        f"  {wk:2d}   {s[0][0]:3d} {s[0][1]:+.3f} {s[0][2]:+.3f} | {s[1][0]:3d} {s[1][1]:+.3f} | {s[2][0]:3d} {s[2][1]:+.3f}"  # noqa: E501
    )
for lo_, hi_ in ((1, 4), (5, 13), (14, 26)):
    x = np.array(res["all"])[:, lo_ - 1 : hi_].ravel()
    x = x[np.isfinite(x)]
    xw = np.array(res["weak"])[:, lo_ - 1 : hi_].ravel()
    xw = xw[np.isfinite(xw)]
    print(
        f"weeks {lo_}-{hi_}: all mean {x.mean():+.4f} n={len(x)} | weak mean {xw.mean():+.4f} n={len(xw)}"  # noqa: E501
    )
# baseline: perps older than 1y (any coin), random weekly shorts
base_r = []
for _sym, a in perp18.items():
    h = a[0]
    if len(h) < 24 * 365 + 168 * 3:
        continue
    for wk in range(53, int(len(h) / 168) - 2, 4):
        r_ = weekly(a, None, range(wk, wk + 1), None)
        base_r.append(r_[0])
base_r = np.array(base_r)
base_r = base_r[np.isfinite(base_r)]
print(
    f"baseline weekly short of perps aged >1y (every 4th week): n={len(base_r)} mean {base_r.mean():+.4f} median {np.median(base_r):+.4f}"  # noqa: E501
)
# share of cumulative decline by age (spot-listed coins, perp): median log price at age relative to listing  # noqa: E501
print("\nmedian/mean log price by age (rel. listing price):")
for dd in (3, 7, 14, 30, 41):
    x = lp[:, dd * 24]
    x = x[np.isfinite(x)]
    print(f"  day {dd}: median {np.median(x):+.3f} mean {x.mean():+.3f}")

# ---- BTC regime
B = perp18["BTCUSDT"]
bh, bo = B[0], B[1]


def btc_ret(t: int, a: int, b: int) -> float:
    i, j = np.searchsorted(bh, t + a * 3600), np.searchsorted(bh, t + b * 3600)
    if j >= len(bh) or i >= len(bh):
        return np.nan
    return bo[j] / bo[i] - 1


rows = []
for r in D.iter_rows(named=True):
    t_listing = int(L.filter(pl.col("symbol") == r["symbol"])["listed"][0].timestamp())
    rows.append({"symbol": r["symbol"], "btc_pre30": btc_ret(t_listing, -720, 0), "btc_0_72": btc_ret(t_listing, 0, 72),  # noqa: E501
                 "btc_fwd30": btc_ret(r["t_in"], 0, 720), "btc_pre72": btc_ret(t_listing, -72, 0)})  # fmt: skip  # noqa: E501
D2 = D.join(pl.DataFrame(rows), on="symbol")
print(f"\nBTC regime vs listing outcome (n={D2.height})")
for col in ("btc_pre30", "btc_0_72", "btc_fwd30"):
    x = (
        D2.with_columns(b=pl.col(col).qcut(3, allow_duplicates=True))
        .group_by("b")
        .agg(
            n=pl.len(),
            net=pl.col("net").mean(),
            weak=pl.col("weak").mean(),
            since=pl.col("since_list").median(),
        )
        .sort("b")
    )
    print(col)
    print(x)
nn = D2.drop_nulls(["btc_fwd30"])
print("corr(net, btc_fwd30) =", np.corrcoef(nn["net"].to_numpy(), nn["btc_fwd30"].to_numpy())[0, 1])
print(
    "corr(since_list, btc_0_72) =",
    np.corrcoef(D2.drop_nulls("btc_0_72")["since_list"], D2.drop_nulls("btc_0_72")["btc_0_72"])[
        0, 1
    ],
)
print(
    D2.group_by("year")
    .agg(
        n=pl.len(),
        btc_pre30=pl.col("btc_pre30").median(),
        btc_fwd30=pl.col("btc_fwd30").median(),
        weak=pl.col("weak").mean(),
        net=pl.col("net").mean(),
    )
    .sort("year")
)
_ = spot_name
