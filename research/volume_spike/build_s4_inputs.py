"""Input datasets for S4 of the CNN robustness study (docs/plans/cnn-robustness.md).

Every output is row-aligned with `seq2_meta.parquet` and written as a NEW local file in
`cache_<universe>/` (never through one of the shared symlinks). Re-running a builder rewrites its
own outputs only.

    cd research/volume_spike
    export MOONSHOT_UNIVERSE=all
    uv run --group research python build_s4_inputs.py ctx   # s4_ctx.parquet
    uv run --group research python build_s4_inputs.py h1    # seq_1h7d.npy + seq_1h7d_meta.parquet
    uv run --group research python build_s4_inputs.py m1    # seq_1m60.npy + seq_1m60_meta.parquet

All inputs are causal: they use data up to the close of the spike candle (open time t, close
t + 5m) and nothing later.

**ctx** (raw values; the runner transforms and z-scores them):

- `tick_rel`: from events.parquet (the row's `event`).
- `btc_ret_4h`, `btc_vol_mult`: from seq2_meta.
- `n_cand_60m`: distinct coins with a spike candidate (a seq2_meta row) whose open time is in
  [t - 55m, t], i.e. the 12 candles (60 minutes) ending with this one, across the universe.
- `n_crash_60m`: of those coins, the ones with at least one candidate in the window with
  `ret_4h <= -10%`.
- `ret4h_rank`: the row's rank of `ret_4h` among the candidates with the same open time, as a
  fraction: 0 = the deepest 4h drop, 1 = the highest (average rank for ties; 0.5 when alone).
- `spike_share`: market.parquet at the spike candle.

**h1** (`seq_1h7d.npy`, float16 [rows, 8, 168]): the last 7 days as 168 one-hour bars, built
from the 5m grid of `build_seq2.grid` (missing candles carry the close, zero volume). The bars
are anchored at the spike close, not at clock hours: the last bar is the 12 5m candles ending
with the spike candle (the 55 minutes before it plus the spike candle itself), and each earlier
bar steps back 12 candles, so every bar is a full hour and none looks past the spike close.
Channels:

0-3  coin open, high, low, close: log ratio to the spike candle's close
4    coin volume: log1p(hour volume / median hour volume of the 168 bars)
5    coin taker-buy share minus 0.5 (0 without volume)
6    BTC close: log ratio to BTC's close at the spike candle
7    BTC volume: log1p(hour volume / median of BTC's 168 bars)

Bars before the coin's first 5m candle (a coin listed less than 7 days before) repeat the first
full bar's prices with zero volume; `seq_1h7d_meta.parquet` has `hours` (full bars available).

**m1** (`seq_1m60.npy`, float16 [rows, 7, 60]): the last 60 one-minute candles ending at the
spike close, i.e. open times t - 55m .. t + 4m, from the 1m klines (each symbol's month files
are read once). Channels 0-3 coin OHLC as log ratio to the close of the t + 4m candle (the spike
close), 4 log1p(volume / median volume of the present minutes), 5 taker share - 0.5, 6 mask
(1 = that minute exists). Missing minutes are zeros with mask 0; a row without the t + 4m
candle (or without any 1m data) is all zeros. `seq_1m60_meta.parquet` has `minutes` (present).
"""

from __future__ import annotations

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from build_seq2 import STEP, T0, grid
from common import CACHE, DATA_DIR
from numpy.lib.stride_tricks import sliding_window_view

WORKERS = 4  # S1 shares the CPU
KLINES_1M = DATA_DIR / "klines" / "spot" / "interval=1m"

H1_STEPS = 168
H1_CH = 8
BAR = 12  # 5m candles per hour bar
M1_STEPS = 60
M1_CH = 7
MIN_US = 60_000_000


def local(name: str) -> Path:
    """A path for a new output in CACHE; refuses a symlink or anything outside CACHE."""
    p = CACHE / name
    if p.is_symlink() or p.resolve().parent != CACHE.resolve():
        msg = f"refusing to write {p}: not a local file in {CACHE}"
        raise RuntimeError(msg)
    return p


def read_meta() -> pl.DataFrame:
    return pl.read_parquet(CACHE / "seq2_meta.parquet").with_columns(
        gidx=((pl.col("open_time") - pl.lit(T0)) / pl.lit(STEP)).cast(pl.Int64)
    )


# ----------------------------------------------------------------------------- ctx


def coins_active(code: np.ndarray, gidx: np.ndarray, n: int) -> np.ndarray:
    """For each grid index g: the number of distinct coins with a row in [g - 11, g]."""
    count = np.zeros(n, dtype=np.int32)
    for c in np.unique(code):
        g = np.unique(gidx[code == c])
        diff = np.zeros(n + BAR + 1, dtype=np.int32)
        np.add.at(diff, g, 1)
        np.add.at(diff, g + BAR, -1)
        count += np.cumsum(diff)[:n] > 0
    return count


def build_ctx() -> None:
    t0 = time.time()
    meta = read_meta()
    gidx = meta["gidx"].to_numpy()
    _, code = np.unique(meta["symbol"].to_numpy(), return_inverse=True)
    n = int(gidx.max()) + 1
    ret4 = meta["ret_4h"].to_numpy()
    crash = np.nan_to_num(ret4, nan=0.0) <= -0.10
    n_cand = coins_active(code, gidx, n)[gidx]
    n_crash = coins_active(code[crash], gidx[crash], n)[gidx]
    rank = meta.select(
        r=(pl.col("ret_4h").rank("average").over("open_time") - 1)
        / (pl.col("ret_4h").count().over("open_time") - 1)
    )["r"]
    rank_np = rank.fill_nan(0.5).fill_null(0.5).to_numpy()
    tick = pl.read_parquet(CACHE / "events.parquet", columns=["tick_rel"])["tick_rel"].to_numpy()
    market = pl.read_parquet(CACHE / "market.parquet")
    assert (market["i"].to_numpy() == np.arange(market.height)).all()
    share = market["spike_share"].to_numpy()[gidx]
    out = meta.select("event", "symbol", "open_time", "btc_ret_4h", "btc_vol_mult").with_columns(
        tick_rel=pl.Series(tick[meta["event"].to_numpy()]),
        n_cand_60m=pl.Series(n_cand),
        n_crash_60m=pl.Series(n_crash),
        ret4h_rank=pl.Series(rank_np),
        spike_share=pl.Series(share),
    )
    out.write_parquet(local("s4_ctx.parquet"))
    print(out.describe(), flush=True)
    print(f"ctx: {out.height} rows in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- h1


def roll(a: np.ndarray, how: str) -> np.ndarray:
    """Aggregate of the BAR candles ending at each index (the first BAR-1 indexes are padded)."""
    if how == "sum":
        c = np.concatenate(([0.0], np.cumsum(a)))
        idx = np.arange(len(a)) + 1
        return c[idx] - c[np.maximum(idx - BAR, 0)]
    pad = np.concatenate((np.full(BAR - 1, a[0]), a))
    w = sliding_window_view(pad, BAR)
    return w.max(1) if how == "max" else w.min(1)


def hour_bars(end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bar indexes (the last 5m candle of each hour bar) for windows ending at `end`, clamped to
    the first full bar, and whether each bar is full."""
    idx = end[:, None] - BAR * np.arange(H1_STEPS - 1, -1, -1)[None, :]
    full = idx >= BAR - 1
    return np.maximum(idx, BAR - 1), full


def rel_volume(v: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """log1p(v / median of the window's present bars); the mean if that median is 0."""
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows (no data) are expected
        vv = np.where(ok, v, np.nan)
        base = np.nanmedian(vv, 1)
        mean = np.nanmean(vv, 1)
    base = np.where(base > 0, base, np.where(mean > 0, mean, 1.0))
    return np.where(ok, np.log1p(v / base[:, None]), 0.0)


def h1_job(job: tuple[str, np.ndarray, np.ndarray, str]) -> tuple[np.ndarray, np.ndarray]:
    """Writes the coin channels of `rows`; returns them and their full hour bars."""
    symbol, gidx, rows, path = job
    off, a = grid(symbol)
    end = gidx - off
    assert end.min() >= 0, symbol
    o, h, lo, c = a["open"], a["high"], a["low"], a["close"]
    hmax, lmin = roll(h, "max"), roll(lo, "min")
    vsum, tbsum = roll(a["volume"], "sum"), roll(a["taker_buy_base_volume"], "sum")
    ii, full = hour_bars(end)
    c0 = c[end][:, None]
    out = np.empty((len(end), H1_CH - 2, H1_STEPS), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:, 0] = np.log(o[ii - (BAR - 1)] / c0)
        out[:, 1] = np.log(hmax[ii] / c0)
        out[:, 2] = np.log(lmin[ii] / c0)
        out[:, 3] = np.log(c[ii] / c0)
        v = np.where(full, vsum[ii], 0.0)
        tb = np.where(full, tbsum[ii], 0.0)
        out[:, 4] = rel_volume(v, full)
        out[:, 5] = np.where(v > 0, tb / np.where(v > 0, v, 1.0), 0.5) - 0.5
    seq = np.load(path, mmap_mode="r+")
    seq[rows, : H1_CH - 2] = np.nan_to_num(np.clip(out, -20, 20)).astype(np.float16)
    seq.flush()
    return rows, full.sum(1).astype(np.int16)


def build_h1() -> None:
    t0 = time.time()
    meta = read_meta()
    path = local("seq_1h7d.npy")
    seq = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(meta.height, H1_CH, H1_STEPS)
    )
    del seq
    gidx, syms = meta["gidx"].to_numpy(), meta["symbol"].to_numpy()
    jobs = [
        (s, gidx[m], np.flatnonzero(m), str(path)) for s in np.unique(syms) for m in [syms == s]
    ]
    jobs.sort(key=lambda j: -len(j[2]))
    hours = np.zeros(meta.height, dtype=np.int16)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for k, (rows, h) in enumerate(pool.map(h1_job, jobs)):
            hours[rows] = h
            if k % 100 == 0:
                print(f"h1 {k}/{len(jobs)} symbols, {time.time() - t0:.0f}s", flush=True)
    # BTC channels, in chunks
    off, a = grid("BTCUSDT")
    c, vsum = a["close"], roll(a["volume"], "sum")
    seq = np.load(path, mmap_mode="r+")
    for k in range(0, len(gidx), 50_000):
        end = gidx[k : k + 50_000] - off
        ii, ok = hour_bars(end)
        seq[k : k + 50_000, 6] = np.clip(np.log(c[ii] / c[end][:, None]), -20, 20).astype(
            np.float16
        )
        v = np.where(ok, vsum[ii], 0.0)
        seq[k : k + 50_000, 7] = np.clip(rel_volume(v, ok), -20, 20).astype(np.float16)
    seq.flush()
    meta.select("event", "symbol", "open_time").with_columns(hours=pl.Series(hours)).write_parquet(
        local("seq_1h7d_meta.parquet")
    )
    sample = np.asarray(seq[::997], dtype=np.float32)
    full = int((hours == H1_STEPS).sum())
    print(
        f"h1: {meta.height} rows, {full} with 168 full bars ({full / meta.height:.2%}),"
        f" nan {int(np.isnan(sample).sum())}, {path.stat().st_size / 1e9:.2f} GB,"
        f" {time.time() - t0:.0f}s",
        flush=True,
    )


# ----------------------------------------------------------------------------- m1


def month_files(symbol: str, t_us: np.ndarray) -> list[Path]:
    d = KLINES_1M / f"symbol={symbol}"
    if not d.exists():
        return []
    months: set[str] = set()
    for off in (-55, 4):
        tt = (t_us + off * MIN_US).astype("datetime64[us]")
        months |= {str(m) for m in np.unique(tt.astype("datetime64[M]"))}
    return [d / f"{m}.parquet" for m in sorted(months) if (d / f"{m}.parquet").exists()]


def m1_job(job: tuple[str, np.ndarray, np.ndarray, str]) -> tuple[np.ndarray, np.ndarray]:
    """Writes `rows`; returns them and their present minutes."""
    symbol, t_us, rows, path = job
    files = month_files(symbol, t_us)
    out = np.zeros((len(rows), M1_CH, M1_STEPS), dtype=np.float32)
    cols = ["open_time", "open", "high", "low", "close", "volume", "taker_buy_base_volume"]
    df = pl.concat([pl.read_parquet(f, columns=cols) for f in files]) if files else None
    if df is not None and df.height:
        df = df.sort("open_time")
        ot = df["open_time"].dt.epoch("us").to_numpy().astype(np.int64)
        want = t_us[:, None] + (np.arange(-55, 5) * MIN_US)[None, :]
        pos = np.minimum(np.searchsorted(ot, want), len(ot) - 1)
        hit = ot[pos] == want
        ref = hit[:, -1]
        hit &= ref[:, None]
        c0 = df["close"].to_numpy()[pos[:, -1]][:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            for k, col in enumerate(("open", "high", "low", "close")):
                out[:, k] = np.where(hit, np.log(df[col].to_numpy()[pos] / c0), 0.0)
            v = np.where(hit, df["volume"].to_numpy()[pos], 0.0)
            tb = np.where(hit, df["taker_buy_base_volume"].to_numpy()[pos], 0.0)
            out[:, 4] = rel_volume(v, hit)
            out[:, 5] = np.where(hit & (v > 0), tb / np.where(v > 0, v, 1.0) - 0.5, 0.0)
        out[:, 6] = hit
    seq = np.load(path, mmap_mode="r+")
    seq[rows] = np.nan_to_num(np.clip(out, -20, 20)).astype(np.float16)
    seq.flush()
    return rows, out[:, 6].sum(1).astype(np.int16)


def build_m1() -> None:
    t0 = time.time()
    meta = read_meta()
    path = local("seq_1m60.npy")
    seq = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(meta.height, M1_CH, M1_STEPS)
    )
    del seq
    t_us = meta["open_time"].dt.epoch("us").to_numpy().astype(np.int64)
    syms = meta["symbol"].to_numpy()
    jobs = [
        (s, t_us[m], np.flatnonzero(m), str(path)) for s in np.unique(syms) for m in [syms == s]
    ]
    jobs.sort(key=lambda j: -len(j[2]))
    minutes = np.zeros(meta.height, dtype=np.int16)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for k, (rows, mins) in enumerate(pool.map(m1_job, jobs)):
            minutes[rows] = mins
            if k % 50 == 0:
                print(f"m1 {k}/{len(jobs)} symbols, {time.time() - t0:.0f}s", flush=True)
    m = meta.select("event", "symbol", "open_time").with_columns(minutes=pl.Series(minutes))
    m.write_parquet(local("seq_1m60_meta.parquet"))
    by_year = (
        m.group_by(year=pl.col("open_time").dt.year())
        .agg(
            rows=pl.len(),
            full=(pl.col("minutes") == M1_STEPS).mean(),
            partial=((pl.col("minutes") > 0) & (pl.col("minutes") < M1_STEPS)).mean(),
            none=(pl.col("minutes") == 0).mean(),
        )
        .sort("year")
    )
    print(by_year, flush=True)
    by_year.write_csv(local("seq_1m60_coverage.csv"))
    full, some = int((minutes == M1_STEPS).sum()), int((minutes > 0).sum())
    print(
        f"m1: {meta.height} rows, {full} full ({full / meta.height:.2%}), {some} with the"
        f" spike-close minute ({some / meta.height:.2%}), {path.stat().st_size / 1e9:.2f} GB,"
        f" {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    started = datetime.now(UTC)
    if cmd == "ctx":
        build_ctx()
    elif cmd == "h1":
        build_h1()
    elif cmd == "m1":
        build_m1()
    else:
        raise SystemExit(__doc__)
    print(f"{cmd} built {started:%H:%M}Z..{datetime.now(UTC):%H:%M}Z", flush=True)
