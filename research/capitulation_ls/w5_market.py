"""W5c: market-level drops: slide vs flush. BTC 3-day drop episodes, 5m volume spikes across 20
majors, forward BTC / basket returns; the 2026-11-14 slide as a case."""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
from common_ls import CACHE_ALL

os.environ["MOONSHOT_UNIVERSE"] = "all"
sys.path.insert(0, str(CACHE_ALL.parent))
from common import load

MAJ = [
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
    "ATOMUSDT",
    "UNIUSDT",
    "ETCUSDT",
    "FILUSDT",
    "AAVEUSDT",
    "XLMUSDT",
]
H = {}
SP = {}
for s in MAJ:
    try:
        d = (
            load(s)
            .filter(pl.col("open_time") >= pl.datetime(2020, 1, 1, time_zone="UTC"))
            .sort("open_time")
        )
    except Exception as e:
        print("skip", s, e)
        continue
    d = d.with_columns(
        vm=pl.col("quote_volume")
        / pl.col("quote_volume").rolling_median(288, min_samples=100).shift(1),
        ret5=pl.col("close") / pl.col("open") - 1,
    )
    spk = d.filter((pl.col("vm") >= 15) & (pl.col("ret5") <= -0.005)).select("open_time")
    SP[s] = spk["open_time"].dt.epoch("s").to_numpy()
    h = (
        d.group_by_dynamic("open_time", every="1h")
        .agg(c=pl.col("close").last(), hi=pl.col("high").max(), lo=pl.col("low").min())
        .sort("open_time")
    )
    H[s] = h
print("loaded", len(H), "last bar", H["BTCUSDT"]["open_time"].max())
hours = H["BTCUSDT"]["open_time"].dt.epoch("s").to_numpy()
close = {
    s: h.join(H["BTCUSDT"].select("open_time"), on="open_time", how="right")
    .sort("open_time")["c"]
    .forward_fill()
    .to_numpy()
    for s, h in H.items()
}
btc = close["BTCUSDT"]
n = len(hours)
d3 = np.full(n, np.nan)
d3[72:] = btc[72:] / btc[:-72] - 1
ret1h = np.full(n, np.nan)
ret1h[1:] = btc[1:] / btc[:-1] - 1
alts = [s for s in close if s != "BTCUSDT"]
_r = np.stack(
    [np.concatenate([[0.0], np.nan_to_num(close[s][1:] / close[s][:-1] - 1)]) for s in alts]
)
_r = np.where(np.abs(_r) > 0.6, 0.0, _r)
basket = np.cumprod(1 + _r.mean(0))
# spike counts per hour across the majors (distinct coins with >=1 spike bar)
spk_h = np.zeros(n)
for t in SP.values():
    hi = np.unique(np.searchsorted(hours, t, side="right") - 1)
    spk_h[hi[(hi >= 0) & (hi < n)]] += 1


def fwd(arr, i, h):
    return arr[min(i + h, n - 1)] / arr[i] - 1


def episodes(thr, lock_h=7 * 24):
    out, last = [], -(10**9)
    for i in range(72, n):
        if d3[i] <= thr and i - last >= lock_h and hours[i] >= 1640995200:  # 2022+
            out.append(i)
            last = i
    return out


print(
    "peak-hour distinct majors with a 15x down bar, per day (2022+): pctiles 50/75/90/95/99",
    np.percentile(
        [spk_h[i : i + 24].max() for i in range(24 * 730, n - 24, 24)], [50, 75, 90, 95, 99]
    ),
)
print(
    "daily total down-spike bars pctiles 50/75/90/95/99",
    np.percentile(
        [spk_h[i : i + 24].sum() for i in range(24 * 730, n - 24, 24)], [50, 75, 90, 95, 99]
    ),
)
print(
    "\n=== BTC 3-day-drop episodes (2022+), forward BTC / basket (20 majors ex BTC), flush metrics ==="  # noqa: E501
)
rows = []
for thr in (-0.08, -0.12, -0.15, -0.20):
    for i in episodes(thr):
        w = slice(i - 72, i + 1)
        worst1h = np.nanmin(ret1h[w])
        # share of the 3d fall realised in the worst 6 hours
        r6 = np.array([btc[j] / btc[j - 6] - 1 for j in range(i - 66, i + 1)])
        share6 = r6.min() / d3[i]
        spk_coins = float(
            np.max(spk_h[i - 24 : i + 1])
        )  # max distinct majors with a >=15x down 5m bar in one hour, last 24h
        spk_tot = float(spk_h[i - 72 : i + 1].sum())
        rows.append(
            dict(
                thr=thr,
                t=str(np.datetime64(int(hours[i]), "s").astype("datetime64[D]")),
                d3=d3[i],
                worst1h=worst1h,
                share6h=share6,
                spk_coins24=spk_coins,
                spk_hours72=spk_tot,
                b3=fwd(btc, i, 72),
                b7=fwd(btc, i, 168),
                b14=fwd(btc, i, 336),
                a3=fwd(basket, i, 72),
                a7=fwd(basket, i, 168),
                a14=fwd(basket, i, 336),
                prior7=btc[i] / btc[max(i - 168 - 72, 0)] - 1,
            )
        )
E = pl.DataFrame(rows)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_rows(80)
for thr in (-0.08, -0.12, -0.15, -0.20):
    x = E.filter(pl.col("thr") == thr)
    print(
        f"thr {thr}: n={x.height} BTC fwd3/7/14 mean {x['b3'].mean() * 100:+.1f}/{x['b7'].mean() * 100:+.1f}/{x['b14'].mean() * 100:+.1f}% med {x['b3'].median() * 100:+.1f}/{x['b7'].median() * 100:+.1f}/{x['b14'].median() * 100:+.1f} | basket fwd3/7/14 mean {x['a3'].mean() * 100:+.1f}/{x['a7'].mean() * 100:+.1f}/{x['a14'].mean() * 100:+.1f}% share basket7>0 {float((x['a7'] > 0).mean()):.2f}"  # noqa: E501
    )
x = E.filter(pl.col("thr") == -0.08).sort("t")
print(
    x.select(
        "t", "d3", "worst1h", "share6h", "spk_coins24", "spk_hours72", "b3", "b7", "a3", "a7", "a14"
    )
)
# flush vs slide split at -8%: worst-hour move or spike breadth
med_sp = float(x["spk_coins24"].median())
for lab, f in (
    ("flush (peak hour >= 6 majors with a 15x down bar)", pl.col("spk_coins24") >= 6),
    ("slide (peak hour < 6)", pl.col("spk_coins24") < 6),
    ("worst 1h <= -3%", pl.col("worst1h") <= -0.03),
    ("worst 1h > -3%", pl.col("worst1h") > -0.03),
):
    y = x.filter(f)
    print(
        f"{lab:48s} n={y.height:>2} d3 {y['d3'].mean() * 100:+.1f}% | BTC fwd3/7 {y['b3'].mean() * 100:+.1f}/{y['b7'].mean() * 100:+.1f} | basket fwd3/7/14 {y['a3'].mean() * 100:+.1f}/{y['a7'].mean() * 100:+.1f}/{y['a14'].mean() * 100:+.1f}, share a7>0 {float((y['a7'] > 0).mean()):.2f}"  # noqa: E501
    )
# all hours with d3 <= -8 (not just first): continuation rate
mask = (d3 <= -0.08) & (hours >= 1640995200)
idx = np.flatnonzero(mask[:-336])
print(
    f"\nhours with 3d drop <= -8% (2022+, overlapping): {len(idx)}; BTC fwd 72h mean {np.mean([fwd(btc, i, 72) for i in idx]) * 100:+.2f}%; P(BTC makes a new 3d low <= d3-5pts within 72h) ",  # noqa: E501
    end="",
)
cont = np.mean([np.nanmin(btc[i : i + 73] / btc[i - 72]) - 1 <= d3[i] - 0.05 for i in idx])
print(f"{cont:.2f}")

# case: Nov 2026
print(
    "\n=== 2026-11 slide: daily view (BTC close, 24h ret, distinct majors with 15x down 5m bars, total down-spike bars, basket) ==="  # noqa: E501
)
lo, hi = np.datetime64("2026-11-08"), np.datetime64("2026-11-24")
for i in range(n):
    d = np.datetime64(int(hours[i]), "s")
    if (
        d.astype("datetime64[D]") >= lo
        and d.astype("datetime64[D]") <= hi
        and d.astype("datetime64[h]").astype(int) % 24 == 23
    ):
        day_spikes = int(
            sum(((SP[s] >= hours[i] - 86400 + 3600) & (SP[s] <= hours[i] + 3600)).sum() for s in SP)
        )
        coins = int(
            sum(((SP[s] >= hours[i] - 86400 + 3600) & (SP[s] <= hours[i] + 3600)).any() for s in SP)
        )
        print(
            str(d.astype("datetime64[D]")),
            f"btc {btc[i]:.0f} 24h {fwd(btc, i - 24, 24) * 100 if False else (btc[i] / btc[i - 24] - 1) * 100:+.1f}% 3d {d3[i] * 100:+.1f}% worst1h(24h) {np.nanmin(ret1h[i - 23 : i + 1]) * 100:+.2f}% coins_w_spike {coins} spike_bars {day_spikes} basket24 {(basket[i] / basket[i - 24] - 1) * 100:+.1f}%",  # noqa: E501
        )
print("typical day baseline (2022+): mean distinct coins w/ 15x down bar per day", end=" ")
dd = []
for i in range(24 * 365 * 2, n - 24, 24):
    dd.append(sum(((SP[s] >= hours[i]) & (SP[s] < hours[i] + 86400)).any() for s in SP))
print(
    round(float(np.mean(dd)), 2),
    " p90",
    np.percentile(dd, 90),
    " p99",
    np.percentile(dd, 99),
    "max",
    max(dd),
)
# flush days: spike coins>=8 days, what does BTC do after
dd = np.array(
    [
        sum(((SP[s] >= hours[i]) & (SP[s] < hours[i] + 86400)).any() for s in SP)
        for i in range(0, n - 24 * 8, 24)
    ]
)
di = np.arange(0, n - 24 * 8, 24)
sel = di[(dd >= 8) & (hours[di] >= 1640995200)]
print(
    f"flush days (>=8 of 20 majors with 15x down 5m bar), 2022+: {len(sel)}; BTC fwd 3d/7d mean",
    f"{np.mean([fwd(btc, i + 24, 72) for i in sel]) * 100:+.1f}/{np.mean([fwd(btc, i + 24, 168) for i in sel]) * 100:+.1f}%; basket fwd 3d/7d",  # noqa: E501
    f"{np.mean([fwd(basket, i + 24, 72) for i in sel]) * 100:+.1f}/{np.mean([fwd(basket, i + 24, 168) for i in sel]) * 100:+.1f}%",  # noqa: E501
    " share of such days also within a 3d BTC drop<=-8%:",
    f"{np.mean([d3[i + 24] <= -0.08 for i in sel]):.2f}",
)
np.save("/tmp/w5_btc_hours.npy", hours)
np.save("/tmp/w5_btc.npy", btc)
E.write_parquet("/tmp/w5_episodes.parquet")
