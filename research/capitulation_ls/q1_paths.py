"""Q1 (illiquid spikes): spot price paths of the sub-5k spikes in L's calibration pool
(pct >= 0.95), with the volume-spike builders' own functions so the outcome is defined as for
every seq2 row: `build_seq2.fwd_job` (fwd2, 8h of 5m bars, entry = the next 5m open) and
`build_fwd72.job` (fwd72, 72 hourly bars, last close held after a series ends).

Writes OWN/q1/cand.parquet (the pool rows, in the order of the two arrays), OWN/q1/fwd2.npy and
OWN/q1/fwd72.npy. Run: uv run --group research python q1_paths.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import p1_lib  # noqa: F401  (sets the universe and the volume_spike import path)
import polars as pl
from build_fwd72 import job as fwd72_job
from build_seq2 import FWD, STEP, T0, fwd_job
from q1_build import Q1

POOL = 0.95


def main() -> None:
    d = pl.read_parquet(Q1 / "spot_spikes.parquet")
    c = d.filter(pl.col("pct").is_not_nan() & (pl.col("pct") >= POOL)).sort("open_time", "symbol")
    c = c.with_columns(gidx=((pl.col("open_time") - pl.lit(T0)) / pl.lit(STEP)).cast(pl.Int64))
    n = c.height
    print(n, "pool rows", file=sys.stderr)
    f2, f72 = Q1 / "fwd2.npy", Q1 / "fwd72.npy"
    a = np.lib.format.open_memmap(f2, mode="w+", dtype=np.float32, shape=(n, 3, FWD))
    b = np.lib.format.open_memmap(f72, mode="w+", dtype=np.float32, shape=(n, 3, 72))
    del a, b
    gidx, syms = c["gidx"].to_numpy(), c["symbol"].to_numpy()
    j2 = [(s, gidx[syms == s], np.flatnonzero(syms == s), str(f2)) for s in np.unique(syms)]
    j72 = [(s, gidx[syms == s], np.flatnonzero(syms == s), str(f72)) for s in np.unique(syms)]
    with ProcessPoolExecutor(8) as ex:
        assert sum(ex.map(fwd_job, j2)) == n
        assert sum(ex.map(fwd72_job, j72)) == n
    c.drop("gidx").write_parquet(Q1 / "cand.parquet")
    x = np.load(f2, mmap_mode="r")
    print("fwd2 NaN share", float(np.isnan(x[:, 2, -1]).mean()), file=sys.stderr)


if __name__ == "__main__":
    main()
