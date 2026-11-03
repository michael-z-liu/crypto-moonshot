"""Raw candle sequences before every spike, for the CNN (M7).

For each spike row of events.parquet (same order) it stores the last SEQ candles up to and
including the spike candle as channels:

0-3  open, high, low, close: log ratio to the spike candle's close
4    volume: log1p(volume / the spike's 24h median volume)
5    taker-buy share of volume, minus 0.5 (0 when the candle has no volume)
6    BTC close: log ratio to BTC's close at the spike candle
7    BTC volume: log1p(volume / BTC's 24h median volume at the spike candle)

Missing candles carry the previous close forward with zero volume. Output (in cache/):

- seq.npy: float16 [spikes, 8, SEQ]
- seq_meta.parquet: event row, symbol, open_time, split, target and the fields used to report

Run: uv run python research/volume_spike/build_seq.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from build_events import full_grid
from common import BARS_PER_DAY, CACHE, load
from evalx import COST, load_events

SEQ = 96  # 8 hours of 5m candles
CHANNELS = 8
HOLD = 12  # 1h, the target horizon


def grid_arrays(symbol: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Gap-free 5m grid of one symbol: open times (naive UTC, ns) and filled OHLCV arrays."""
    df = full_grid(load(symbol)).with_columns(
        pl.col("close").forward_fill(),
        pl.col("volume", "taker_buy_base_volume").fill_null(0.0),
    )
    df = df.with_columns(pl.col("open", "high", "low").fill_null(pl.col("close")))
    df = df.with_columns(
        vol_base=pl.col("volume")
        .shift(1)
        .rolling_median(BARS_PER_DAY, min_samples=BARS_PER_DAY // 2)
    )
    t = df["open_time"].dt.replace_time_zone(None).to_numpy()
    cols = ("open", "high", "low", "close", "volume", "taker_buy_base_volume", "vol_base")
    return t, {c: df[c].to_numpy().astype(np.float64) for c in cols}


def windows(arr: np.ndarray, end: np.ndarray) -> np.ndarray:
    return arr[end[:, None] - SEQ + 1 + np.arange(SEQ)[None, :]]


def process(job: tuple[str, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    symbol, times, vol_base = job
    t, a = grid_arrays(symbol)
    end = np.searchsorted(t, times)
    assert (t[end] == times).all(), symbol
    c0 = a["close"][end][:, None]
    out = np.empty((len(end), 6, SEQ), dtype=np.float32)
    for k, c in enumerate(("open", "high", "low", "close")):
        out[:, k] = np.log(windows(a[c], end) / c0)
    v = windows(a["volume"], end)
    out[:, 4] = np.log1p(v / vol_base[:, None])
    tb = windows(a["taker_buy_base_volume"], end)
    out[:, 5] = np.where(v > 0, tb / np.where(v > 0, v, 1.0), 0.5) - 0.5
    return end, out


def main() -> None:
    ev, fwd = load_events()
    idx = np.flatnonzero(ev["is_spike"].to_numpy())
    sp = ev[idx].with_columns(
        event=pl.Series(idx),
        y=pl.Series(np.asarray(fwd[idx, 2, HOLD - 1], dtype=np.float64)) - COST,
        naive=pl.col("open_time").dt.replace_time_zone(None),
    )
    meta = sp.select(
        "event", "symbol", "open_time", "split", "y", "qv_base", "ret_4h", "btc_vol_mult",
        "btc_ret_4h",
    )  # fmt: skip
    seq = np.lib.format.open_memmap(
        CACHE / "seq.npy", mode="w+", dtype=np.float16, shape=(sp.height, CHANNELS, SEQ)
    )
    pos = {s: np.flatnonzero(sp["symbol"].to_numpy() == s) for s in sp["symbol"].unique()}
    naive = sp["naive"].to_numpy()
    vb = sp["vol_base"].to_numpy().astype(np.float64)
    jobs = [(s, naive[p], vb[p]) for s, p in pos.items()]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for (symbol, _, _), (_, out) in zip(jobs, pool.map(process, jobs), strict=True):
            seq[pos[symbol], :6] = np.clip(out, -20, 20).astype(np.float16)
            print(f"{symbol}: {len(pos[symbol])}", file=sys.stderr)
    bt, b = grid_arrays("BTCUSDT")
    end = np.searchsorted(bt, naive)
    assert (bt[end] == naive).all()
    bc = windows(b["close"], end)
    seq[:, 6] = np.log(bc / b["close"][end][:, None]).astype(np.float16)
    seq[:, 7] = np.log1p(windows(b["volume"], end) / b["vol_base"][end][:, None]).astype(np.float16)
    seq.flush()
    meta.write_parquet(CACHE / "seq_meta.parquet")
    print(seq.shape, "nan:", int(np.isnan(seq[::97]).sum()))


if __name__ == "__main__":
    main()
