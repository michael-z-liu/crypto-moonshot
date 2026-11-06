"""Quiet-coin snapshots (B3): every coin every 4 hours, how quiet it is and what happens next.

A snapshot is the 5m candle that closes at 00:00, 04:00, ... UTC. Features use candles up to and
including it; the trade would enter at the next candle's open. From 30 days after a coin's first
candle. On the gap-free 5m grid (missing candles carry the last close; after a series ends the last
close holds, so a delisted coin is valued at its last traded price).

Features (relative ones are ratios to the coin's own median over the previous 180 days of
snapshots, at least 30 days of history):
- rv1_rel, rv7_rel: realised volatility of 5m log returns over 1 / 7 days
- rng1_rel, rng7_rel: high-low range over 1 / 7 days, as a share of the close
- qv1_rel, qv7_rel: quote volume over 1 / 7 days
- q1, q7: quiet scores, the geometric mean of the three ratios for 1 / 7 days (1 = normal)
- flat1: share of 5m candles in the last day whose close equals the previous close
- days_since_spike: days since the last 5m candle with volume >= 10x its 24h median
- ret1, ret7, ret30: returns; dist_hi30: close / 30-day high - 1
- tbr_diff: taker-buy share of volume over 1 day minus over 7 days
- trades_rel, avgtrade_rel: trade count and average trade size over 1 day vs the 30-day median
- tick_rel: smallest non-zero 5m close change in the last day / close (a tick-size proxy)
- qv_med: median quote volume per 5m candle over the last day (USDT); age_days
Outcomes from the entry (next candle's open), as in build_long30:
- up_10 / up_20 / up_30 / up_50 / up_100: candles until the high first reaches the target, -1 if
  not within 30 days; dn_10 / dn_20 / dn_30 / dn_50: the same for the low
- c_1d / c_3d / c_7d / c_30d, lo_*, hi_*: close, lowest low and highest high within each window

Output: cache[_<universe>]/quiet.parquet. BTC context is joined on time.

Run: MOONSHOT_UNIVERSE=all uv run python research/volume_spike/build_quiet.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from build_events import full_grid
from build_long30 import LEVELS, MAX_BARS, first_hit, range_op, sparse
from common import BARS_PER_DAY, CACHE, load, universe

STEP = 48  # 4 hours of 5m candles
WARMUP = 30 * BARS_PER_DAY
SNAP_PER_DAY = BARS_PER_DAY // STEP
REL_WINDOW = 180 * SNAP_PER_DAY
REL_MIN = 30 * SNAP_PER_DAY
UP = {"up_10": 0.1, "up_20": 0.2, "up_30": 0.3, "up_50": 0.5, "up_100": 1.0}
DN = {"dn_10": 0.1, "dn_20": 0.2, "dn_30": 0.3, "dn_50": 0.5}
WINDOWS = {"1d": 1, "3d": 3, "7d": 7, "30d": 30}


def first_low(st_min: list[np.ndarray], start: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """First index j in [start, start + MAX_BARS) with low[j] <= thr, as an offset; -1 if none."""
    pos = start.copy()
    end = start + MAX_BARS
    for k in range(LEVELS - 1, -1, -1):
        step = 1 << k
        skip = (pos + step <= end) & (st_min[k][pos] > thr)
        pos = np.where(skip, pos + step, pos)
    hit = (pos < end) & (st_min[0][pos] <= thr)
    return np.where(hit, pos - start, -1).astype(np.int32)


def window_sum(cs: np.ndarray, i: np.ndarray, w: int) -> np.ndarray:
    """Sum of x[i - w + 1 .. i] from the cumulative sum cs (cs[k] = x[0] + ... + x[k])."""
    lo = i - w
    return cs[i] - np.where(lo >= 0, cs[np.maximum(lo, 0)], 0.0)


def rel(x: pl.Expr) -> pl.Expr:
    """Ratio to the median of the previous REL_WINDOW snapshots (past only)."""
    return x / x.shift(1).rolling_median(REL_WINDOW, min_samples=REL_MIN)


def rel30(x: pl.Expr) -> pl.Expr:
    """Ratio to the median of the previous 30 days of snapshots (past only)."""
    return x / x.shift(1).rolling_median(30 * SNAP_PER_DAY, min_samples=7 * SNAP_PER_DAY)


def process(symbol: str) -> pl.DataFrame | None:
    raw = load(symbol)
    if raw.height < WARMUP + 7 * BARS_PER_DAY:
        return None
    df = full_grid(raw)
    t = df["open_time"]
    n = df.height
    close = df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    opn = df["open"].to_numpy().astype(np.float64)
    high = np.where(np.isnan(high), close, high)
    low = np.where(np.isnan(low), close, low)
    opn = np.where(np.isnan(opn), close, opn)
    vol = df["volume"].fill_null(0.0).to_numpy()
    qv = df["quote_volume"].fill_null(0.0).to_numpy()
    trades = df["trade_count"].fill_null(0).to_numpy().astype(np.float64)
    tbb = df["taker_buy_base_volume"].fill_null(0.0).to_numpy()
    # candles that exist (padding after a delisting has zero volume and is dropped later by the
    # liquidity filter, qv_med >= 1k USDT)
    real = df["volume"].is_not_null().to_numpy()

    lr = np.zeros(n)
    lr[1:] = np.log(close[1:] / close[:-1])
    lr = np.nan_to_num(lr)
    flat = np.zeros(n)
    flat[1:] = close[1:] == close[:-1]
    chg = np.abs(np.diff(close, prepend=close[0]))
    chg = np.where(chg > 0, chg, np.inf)
    cs = {
        "r2": np.cumsum(lr**2), "qv": np.cumsum(qv), "v": np.cumsum(vol), "tbb": np.cumsum(tbb),
        "tr": np.cumsum(trades), "flat": np.cumsum(flat),
    }  # fmt: skip
    vmed = (
        pl.Series(vol).shift(1).rolling_median(BARS_PER_DAY, min_samples=BARS_PER_DAY // 2)
    ).to_numpy()
    spike = (vol >= 10 * vmed) & (vmed > 0)
    last_spike = np.maximum.accumulate(np.where(spike, np.arange(n), -1))
    qvmed = pl.Series(qv).rolling_median(BARS_PER_DAY, min_samples=BARS_PER_DAY // 2).to_numpy()
    tickmin = pl.Series(chg).rolling_min(BARS_PER_DAY, min_samples=12).to_numpy()

    end_min = (t.dt.hour().cast(pl.Int32) * 60 + t.dt.minute().cast(pl.Int32)).to_numpy()
    i = np.flatnonzero(((end_min + 5) % 240 == 0) & (np.arange(n) >= WARMUP) & real)
    i = i[i < n - 1]
    if len(i) == 0:
        return None
    pad = (1 << LEVELS) + MAX_BARS + 2
    last = close[-1]
    hp = np.concatenate([high, np.full(pad, last)])
    lp = np.concatenate([low, np.full(pad, last)])
    cp = np.concatenate([close, np.full(pad, last)])
    st_hi, st_lo = sparse(hp, np.maximum), sparse(lp, np.minimum)
    d1, d7, d30 = BARS_PER_DAY, 7 * BARS_PER_DAY, 30 * BARS_PER_DAY
    c = close[i]
    out: dict[str, np.ndarray] = {
        "open_time": t.to_numpy()[i],
        "rv1": np.sqrt(window_sum(cs["r2"], i, d1) / d1),
        "rv7": np.sqrt(window_sum(cs["r2"], i, d7) / d7),
        "rng1": (range_op(st_hi, i - d1 + 1, d1, np.maximum)
                 - range_op(st_lo, i - d1 + 1, d1, np.minimum)) / c,
        "rng7": (range_op(st_hi, i - d7 + 1, d7, np.maximum)
                 - range_op(st_lo, i - d7 + 1, d7, np.minimum)) / c,
        "qv1": window_sum(cs["qv"], i, d1),
        "qv7": window_sum(cs["qv"], i, d7) / 7,
        "flat1": window_sum(cs["flat"], i, d1) / d1,
        "days_since_spike": np.where(last_spike[i] >= 0, (i - last_spike[i]) / d1, 999.0),
        "ret1": c / close[i - d1] - 1,
        "ret7": c / close[i - d7] - 1,
        "ret30": c / close[i - d30 + 1] - 1,
        "dist_hi30": c / range_op(st_hi, i - d30 + 1, d30, np.maximum) - 1,
        "tbr_diff": window_sum(cs["tbb"], i, d1) / np.maximum(window_sum(cs["v"], i, d1), 1e-12)
        - window_sum(cs["tbb"], i, d7) / np.maximum(window_sum(cs["v"], i, d7), 1e-12),
        "trades1": window_sum(cs["tr"], i, d1),
        "avgtrade1": window_sum(cs["qv"], i, d1) / np.maximum(window_sum(cs["tr"], i, d1), 1),
        "tick_rel": tickmin[i] / c,
        "qv_med": qvmed[i],
        "age_days": i / d1,
    }  # fmt: skip
    e = i + 1
    entry = opn[e]
    ok = entry > 0
    for name, tp in UP.items():
        out[name] = first_hit(st_hi, e, entry * (1 + tp))
    for name, sl in DN.items():
        out[name] = first_low(st_lo, e, entry * (1 - sl))
    for name, days in WINDOWS.items():
        w = days * BARS_PER_DAY
        out[f"c_{name}"] = (cp[e + w - 1] / entry - 1).astype(np.float32)
        out[f"lo_{name}"] = (range_op(st_lo, e, w, np.minimum) / entry - 1).astype(np.float32)
        out[f"hi_{name}"] = (range_op(st_hi, e, w, np.maximum) / entry - 1).astype(np.float32)
    snap = (
        pl.DataFrame(out)
        .filter(pl.Series(ok))
        .with_columns(pl.col("open_time").dt.replace_time_zone("UTC"), symbol=pl.lit(symbol))
    )
    snap = snap.with_columns(
        rv1_rel=rel(pl.col("rv1")), rv7_rel=rel(pl.col("rv7")),
        rng1_rel=rel(pl.col("rng1")), rng7_rel=rel(pl.col("rng7")),
        qv1_rel=rel(pl.col("qv1")), qv7_rel=rel(pl.col("qv7")),
        trades_rel=rel30(pl.col("trades1")), avgtrade_rel=rel30(pl.col("avgtrade1")),
    )  # fmt: skip
    geo = lambda a, b, c: ((pl.col(a).log() + pl.col(b).log() + pl.col(c).log()) / 3).exp()  # noqa: E731
    return snap.with_columns(
        q1=geo("rv1_rel", "rng1_rel", "qv1_rel"), q7=geo("rv7_rel", "rng7_rel", "qv7_rel")
    )


def main() -> None:
    symbols = universe()
    parts = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for s, res in zip(symbols, pool.map(process, symbols), strict=True):
            if res is not None:
                parts.append(res)
            print(f"{s}: {0 if res is None else res.height}", file=sys.stderr, flush=True)
    snaps = pl.concat(parts, how="vertical_relaxed")
    btc = snaps.filter(pl.col("symbol") == "BTCUSDT").select(
        "open_time", btc_ret1="ret1", btc_ret7="ret7", btc_rv7_rel="rv7_rel"
    )
    snaps = snaps.join(btc, on="open_time", how="left").with_columns(
        signal_time=pl.col("open_time"), year=pl.col("open_time").dt.year()
    )
    snaps.write_parquet(CACHE / "quiet.parquet")
    print(snaps.height, "snapshots,", snaps["symbol"].n_unique(), "coins", flush=True)


if __name__ == "__main__":
    main()
