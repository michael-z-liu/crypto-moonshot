"""Richer sequences for the walk-forward CNN study (M9): 24h windows, coin, BTC, ETH and market.

Candidates are spikes on liquid coins (median >= 5k USDT per 5m candle). For each, the last SEQ
candles up to and including the spike candle, as channels:

0-3   coin open, high, low, close: log ratio to the spike candle's close
4     coin volume: log1p(volume / the spike's 24h median volume)
5     coin taker-buy share minus 0.5 (0 without volume)
6     coin trade count: log1p(trades / the spike's 24h median trade count)
7-8   BTC close (log ratio to BTC at the spike) and volume (log1p over its 24h median at the spike)
9-10  ETH, the same
11    market: cumulative equal-weight log return of the universe, relative to the spike candle
12    market: share of universe coins whose volume is >= 5x their own 24h median at that candle

Missing candles carry the previous close forward with zero volume. Outputs (in cache/):

- seq2.npy: float16 [events, 13, SEQ]
- seq2_meta.parquet: event row, symbol, time, scalars and targets (net returns at 15m..4h)
- market.parquet: the two market series on the global 5m grid

Run: uv run python research/volume_spike/build_seq2.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from build_events import full_grid
from common import BARS_PER_DAY, CACHE, load, universe
from evalx import COST, load_events

SEQ = 288
CHANNELS = 13
LIQ = 5_000
HORIZONS = {"y15m": 3, "y30m": 6, "y1h": 12, "y2h": 24, "y4h": 48}
FWD = 96  # forward bars kept for the path forecasts (8 hours)
T0 = datetime(2020, 1, 1, tzinfo=UTC)
STEP = timedelta(minutes=5)


def grid(symbol: str) -> tuple[int, dict[str, np.ndarray]]:
    """Offset on the global grid and the filled arrays of one symbol."""
    df = full_grid(load(symbol)).with_columns(
        pl.col("close").forward_fill(),
        pl.col("volume", "taker_buy_base_volume", "trade_count").fill_null(0),
    )
    df = df.with_columns(pl.col("open", "high", "low").fill_null(pl.col("close")))
    med = BARS_PER_DAY // 2
    df = df.with_columns(
        vol_base=pl.col("volume").shift(1).rolling_median(BARS_PER_DAY, min_samples=med),
        trades_base=pl.col("trade_count").cast(pl.Float64).shift(1)
        .rolling_median(BARS_PER_DAY, min_samples=med),
    )  # fmt: skip
    off = int((df["open_time"][0] - T0) / STEP)
    cols = ("open", "high", "low", "close", "volume", "taker_buy_base_volume", "trade_count",
            "vol_base", "trades_base")  # fmt: skip
    return off, {c: df[c].cast(pl.Float64).to_numpy() for c in cols}


def market_part(symbol: str) -> tuple[int, np.ndarray, np.ndarray]:
    off, a = grid(symbol)
    c = a["close"]
    r = np.zeros_like(c)
    r[1:] = np.log(c[1:] / c[:-1])
    valid = np.ones_like(c, dtype=bool)
    valid[:BARS_PER_DAY] = False  # skip the first day (no baseline yet)
    spike = (a["volume"] >= 5 * a["vol_base"]) & (a["vol_base"] > 0)
    return off, np.where(valid, r, np.nan), np.where(valid, spike, np.nan)


def build_market(symbols: list[str]) -> pl.DataFrame:
    n = int((datetime(2026, 12, 1, tzinfo=UTC) - T0) / STEP)
    rsum, rcnt, ssum = np.zeros(n), np.zeros(n), np.zeros(n)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for off, r, s in pool.map(market_part, symbols):
            k = slice(off, off + len(r))
            ok = ~np.isnan(r)
            rsum[k] += np.where(ok, r, 0)
            ssum[k] += np.where(ok, s, 0)
            rcnt[k] += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        alt = np.where(rcnt > 0, rsum / rcnt, 0.0)
        share = np.where(rcnt > 0, ssum / rcnt, 0.0)
    return pl.DataFrame({"i": np.arange(n), "alt_r": alt, "spike_share": share, "coins": rcnt})


def windows(arr: np.ndarray, end: np.ndarray) -> np.ndarray:
    return arr[end[:, None] - SEQ + 1 + np.arange(SEQ)[None, :]]


def coin_job(job: tuple[str, np.ndarray, np.ndarray, str, np.ndarray, np.ndarray]) -> int:
    """Coin channels; volume and trades are scaled by the spike detector's own 24h baselines."""
    symbol, gidx, rows, path, vbase, tbase = job
    off, a = grid(symbol)
    end = gidx - off
    assert end.min() >= SEQ - 1, symbol
    seq = np.load(path, mmap_mode="r+")
    c0 = a["close"][end][:, None]
    out = np.empty((len(end), 7, SEQ), dtype=np.float32)
    for k, c in enumerate(("open", "high", "low", "close")):
        out[:, k] = np.log(windows(a[c], end) / c0)
    v = windows(a["volume"], end)
    out[:, 4] = np.log1p(v / vbase[:, None])
    tb = windows(a["taker_buy_base_volume"], end)
    out[:, 5] = np.where(v > 0, tb / np.where(v > 0, v, 1.0), 0.5) - 0.5
    out[:, 6] = np.log1p(windows(a["trade_count"], end) / np.maximum(tbase, 1.0)[:, None])
    seq[rows, :7] = np.nan_to_num(np.clip(out, -20, 20)).astype(np.float16)
    seq.flush()
    return len(rows)


def fwd_job(job: tuple[str, np.ndarray, np.ndarray, str]) -> int:
    """(high, low, close) / entry - 1 for the FWD candles after the spike; NaN past the data."""
    symbol, gidx, rows, path = job
    off, a = grid(symbol)
    end = gidx - off
    n = len(a["close"])
    out = np.load(path, mmap_mode="r+")
    entry = a["open"][np.minimum(end + 1, n - 1)]
    k = end[:, None] + 1 + np.arange(FWD)[None, :]
    ok = k < n
    kk = np.minimum(k, n - 1)
    for j, c in enumerate(("high", "low", "close")):
        out[rows, j] = np.where(ok, a[c][kk] / entry[:, None] - 1, np.nan).astype(np.float32)
    out.flush()
    return len(rows)


def forward() -> None:
    meta = pl.read_parquet(CACHE / "seq2_meta.parquet").with_columns(
        gidx=((pl.col("open_time") - pl.lit(T0)) / pl.lit(STEP)).cast(pl.Int64)
    )
    path = CACHE / "fwd2.npy"
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(meta.height, 3, FWD))
    del arr
    gidx, syms = meta["gidx"].to_numpy(), meta["symbol"].to_numpy()
    jobs = [(s, gidx[syms == s], np.flatnonzero(syms == s), str(path)) for s in np.unique(syms)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        print("fwd rows", sum(pool.map(fwd_job, jobs)), file=sys.stderr)


def main() -> None:
    symbols = universe()
    print("market series", file=sys.stderr)
    market = build_market(symbols)
    market.write_parquet(CACHE / "market.parquet")

    ev, fwd = load_events()
    idx = np.flatnonzero((ev["is_spike"] & (ev["qv_base"] >= LIQ)).fill_null(False).to_numpy())
    sp = ev[idx].with_columns(
        event=pl.Series(idx),
        gidx=((pl.col("open_time") - pl.lit(T0)) / pl.lit(STEP)).cast(pl.Int64),
    )
    close = np.asarray(fwd[idx, 2, :], dtype=np.float64)
    meta = sp.select(
        "event",
        "symbol",
        "open_time",
        "gidx",
        "qv_base",
        "listed_days",
        "hour",
        "ret_4h",
        "ret_24h",
        "btc_vol_mult",
        "btc_ret_4h",
        "vol_mult",
    ).with_columns(**{k: pl.Series(close[:, b - 1] - COST) for k, b in HORIZONS.items()})
    path = CACHE / "seq2.npy"
    seq = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(sp.height, CHANNELS, SEQ)
    )
    del seq
    gidx = meta["gidx"].to_numpy()
    syms = meta["symbol"].to_numpy()
    vb = sp["vol_base"].to_numpy().astype(np.float64)
    tb = sp["trades_base"].fill_null(1.0).to_numpy().astype(np.float64)
    jobs = [(s, gidx[m], np.flatnonzero(m), str(path), vb[m], tb[m])
            for s in np.unique(syms) for m in [syms == s]]  # fmt: skip
    with ProcessPoolExecutor(max_workers=8) as pool:
        done = sum(pool.map(coin_job, jobs))
    print("coin rows", done, file=sys.stderr)

    seq = np.load(path, mmap_mode="r+")
    for ch, sym in ((7, "BTCUSDT"), (9, "ETHUSDT")):
        off, a = grid(sym)
        end = gidx - off
        for k in range(0, len(end), 50_000):
            e = end[k : k + 50_000]
            seq[k : k + 50_000, ch] = np.log(
                windows(a["close"], e) / a["close"][e][:, None]
            ).astype(np.float16)
            seq[k : k + 50_000, ch + 1] = np.log1p(
                windows(a["volume"], e) / a["vol_base"][e][:, None]
            ).astype(np.float16)  # fmt: skip
    cum = np.cumsum(market["alt_r"].to_numpy())
    share = market["spike_share"].to_numpy()
    for k in range(0, len(gidx), 50_000):
        e = gidx[k : k + 50_000]
        seq[k : k + 50_000, 11] = (windows(cum, e) - cum[e][:, None]).astype(np.float16)
        seq[k : k + 50_000, 12] = windows(share, e).astype(np.float16)
    seq.flush()
    meta.drop("gidx").write_parquet(CACHE / "seq2_meta.parquet")
    print(seq.shape, "nan:", int(np.isnan(seq[::101]).sum()))


if __name__ == "__main__":
    forward() if sys.argv[1:] == ["fwd"] else main()
