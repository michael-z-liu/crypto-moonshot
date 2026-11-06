"""Sequences for the quiet-coin CNN (B3 step 3): quiet snapshots, 7 days of 1h + 24h of 5m candles.

Candidates: snapshots from m19_quiet.load() (liquid, entry before 2026-10-01) in Q7 deciles 1-3.
The snapshot candle closes on a 4-hour boundary, so the 168 hours before it are whole hours.

seq_h, float16 [N, 10, 168], one step per hour, the last ending at the snapshot:
  0-3 open, high, low, close: log ratio to the snapshot close
  4   quote volume: log1p(hour / median hour of the 7 days)
  5   taker-buy share of volume minus 0.5 (0 without volume)
  6   trade count: log1p(hour / median hour of the 7 days)
  7   average trade size: log(hour / median of the 7 days), clipped to +-5 (0 without trades)
  8-9 BTC close (log ratio to BTC at the snapshot) and quote volume (as channel 4)
seq_m, float16 [N, 6, 288], the last 24h of 5m candles:
  0-3 open, high, low, close: log ratio to the snapshot close
  4   volume: log1p(candle / mean candle of the 24h)
  5   taker-buy share minus 0.5
Missing candles carry the last close with zero volume. Outputs: cache_all/qseq_h.npy,
qseq_m.npy, qseq_meta.parquet (same row order; snapshot features and outcomes).

Run: MOONSHOT_UNIVERSE=all uv run python research/volume_spike/build_quiet_seq.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import polars as pl
from build_events import full_grid
from common import CACHE, load

H, M = 168, 288
FIVE = np.timedelta64(5, "m")


def arrays(symbol: str) -> tuple[np.datetime64, dict[str, np.ndarray]]:
    df = full_grid(load(symbol))
    c = df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
    a = {"c": c}
    for k in ("open", "high", "low"):
        x = df[k].to_numpy().astype(np.float64)
        a[k[0]] = np.where(np.isnan(x), c, x)
    for k, col in (("v", "volume"), ("qv", "quote_volume"), ("tb", "taker_buy_base_volume")):
        a[k] = df[col].fill_null(0.0).to_numpy().astype(np.float64)
    a["n"] = df["trade_count"].fill_null(0).to_numpy().astype(np.float64)
    t0 = df["open_time"].dt.replace_time_zone(None).to_numpy()[0]
    return t0, a


@lru_cache(maxsize=1)
def btc() -> tuple[np.datetime64, dict[str, np.ndarray]]:
    return arrays("BTCUSDT")


def hourly(x: np.ndarray, idx: np.ndarray, how: str) -> np.ndarray:
    """[n, 168] hourly aggregates of a 5m array over the 2016 candles ending at idx."""
    k = idx[:, None] + np.arange(-H * 12 + 1, 1)[None, :]
    w = x[k].reshape(len(idx), H, 12)
    return {"first": w[:, :, 0], "last": w[:, :, -1], "max": w.max(2), "min": w.min(2),
            "sum": w.sum(2)}[how]  # fmt: skip


def rel_log1p(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=1, keepdims=True)
    med = np.where(med > 0, med, x.mean(axis=1, keepdims=True) + 1e-12)
    return np.log1p(x / med)


def job(args: tuple[str, np.ndarray, np.ndarray, str, str]) -> int:
    symbol, times, rows, path_h, path_m = args
    t0, a = arrays(symbol)
    i = ((times - t0) // FIVE).astype(np.int64)
    cs = a["c"][i][:, None]
    out_h = np.zeros((len(i), 10, H), dtype=np.float32)
    for ch, (k, how) in enumerate((("o", "first"), ("h", "max"), ("l", "min"), ("c", "last"))):
        out_h[:, ch] = np.log(hourly(a[k], i, how) / cs)
    qv, v, tb, n = (hourly(a[k], i, "sum") for k in ("qv", "v", "tb", "n"))
    out_h[:, 4] = rel_log1p(qv)
    out_h[:, 5] = np.where(v > 0, tb / np.maximum(v, 1e-12) - 0.5, 0.0)
    out_h[:, 6] = rel_log1p(n)
    avg = np.where(n > 0, qv / np.maximum(n, 1), np.nan)
    med = np.nanmedian(np.where(np.isnan(avg).all(1, keepdims=True), 1.0, avg), axis=1)
    out_h[:, 7] = np.clip(np.nan_to_num(np.log(avg / med[:, None]), nan=0.0), -5, 5)
    bt0, b = btc()
    j = ((times - bt0) // FIVE).astype(np.int64)
    ok = (j >= H * 12) & (j < len(b["c"]))
    jj = np.where(ok, j, H * 12)
    out_h[:, 8] = np.where(ok[:, None], np.log(hourly(b["c"], jj, "last") / b["c"][jj][:, None]), 0)
    out_h[:, 9] = np.where(ok[:, None], rel_log1p(hourly(b["qv"], jj, "sum")), 0)
    k = i[:, None] + np.arange(-M + 1, 1)[None, :]
    out_m = np.zeros((len(i), 6, M), dtype=np.float32)
    for ch, key in enumerate(("o", "h", "l", "c")):
        out_m[:, ch] = np.log(a[key][k] / cs)
    vm = a["v"][k]
    out_m[:, 4] = np.log1p(vm / (vm.mean(1, keepdims=True) + 1e-12))
    out_m[:, 5] = np.where(vm > 0, a["tb"][k] / np.maximum(vm, 1e-12) - 0.5, 0.0)
    sh = np.load(path_h, mmap_mode="r+")
    sm = np.load(path_m, mmap_mode="r+")
    sh[rows] = np.nan_to_num(out_h, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
    sm[rows] = np.nan_to_num(out_m, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
    sh.flush()
    sm.flush()
    return len(rows)


def main() -> None:
    from m19_quiet import load as load_quiet

    meta = load_quiet().filter(pl.col("q7_dec") <= 3).sort("symbol", "open_time")
    meta = meta.with_row_index("qrow")
    n = meta.height
    path_h, path_m = CACHE / "qseq_h.npy", CACHE / "qseq_m.npy"
    for p, shape in ((path_h, (n, 10, H)), (path_m, (n, 6, M))):
        arr = np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=shape)
        del arr
    jobs = []
    for (s,), g in meta.group_by("symbol", maintain_order=True):
        times = g["open_time"].dt.replace_time_zone(None).to_numpy()
        jobs.append((s, times, g["qrow"].to_numpy(), str(path_h), str(path_m)))
    with ProcessPoolExecutor(max_workers=8) as pool:
        done = sum(pool.map(job, jobs))
    meta.write_parquet(CACHE / "qseq_meta.parquet")
    sh = np.load(path_h, mmap_mode="r")
    print(
        n, "rows,", done, "written; sample abs max", float(np.abs(sh[:: max(1, n // 1000)]).max())
    )


if __name__ == "__main__":
    sys.exit(main())
