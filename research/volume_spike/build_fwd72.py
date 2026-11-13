"""B5: hourly 3-day forward paths for every row of seq2 (the liquid-spike dataset).

fwd72.npy, float32 [rows, 3, 72]: for each hour k = 1..72 after entry (the next 5m candle's
open), the hour's high, low and close, relative to the entry price (x / entry - 1). On the
gap-free 5m grid; after a series ends the last close holds (a delisted coin keeps its last
traded price). Same row order as seq2_meta.parquet.

Run: MOONSHOT_UNIVERSE=all uv run python research/volume_spike/build_fwd72.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from build_seq2 import STEP, T0, grid
from common import CACHE

HOURS = 72


def job(args: tuple[str, np.ndarray, np.ndarray, str]) -> int:
    symbol, gidx, rows, path = args
    off, a = grid(symbol)
    n = len(a["close"])
    pad = HOURS * 12 + 2
    last = a["close"][-1]
    hi, lo, cl = (np.concatenate([a[k], np.full(pad, last)]) for k in ("high", "low", "close"))
    e = gidx - off + 1
    entry = a["open"][np.minimum(e, n - 1)]
    k = e[:, None] + np.arange(HOURS * 12)[None, :]
    out = np.empty((len(e), 3, HOURS), dtype=np.float32)
    out[:, 0] = hi[k].reshape(len(e), HOURS, 12).max(2) / entry[:, None] - 1
    out[:, 1] = lo[k].reshape(len(e), HOURS, 12).min(2) / entry[:, None] - 1
    out[:, 2] = cl[k].reshape(len(e), HOURS, 12)[:, :, -1] / entry[:, None] - 1
    arr = np.load(path, mmap_mode="r+")
    arr[rows] = out
    arr.flush()
    return len(rows)


def main() -> None:
    meta = pl.read_parquet(CACHE / "seq2_meta.parquet").with_columns(
        gidx=((pl.col("open_time") - pl.lit(T0)) / pl.lit(STEP)).cast(pl.Int64)
    )
    path = CACHE / "fwd72.npy"
    arr = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(meta.height, 3, HOURS)
    )
    del arr
    gidx, syms = meta["gidx"].to_numpy(), meta["symbol"].to_numpy()
    jobs = [(s, gidx[syms == s], np.flatnonzero(syms == s), str(path)) for s in np.unique(syms)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        done = sum(pool.map(job, jobs))
    f = np.load(path, mmap_mode="r")
    print(done, "rows; NaN share", float(np.isnan(f[::97]).mean()), file=sys.stderr)


if __name__ == "__main__":
    main()
