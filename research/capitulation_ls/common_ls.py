"""Shared loaders for the capitulation long/short search (docs/plans/capitulation-long-short.md).

Reads the volume-spike caches (universe "all"): seq2_meta (1.85M liquid 5m spikes, 2020-01 ..
2026-11-01), fwd2 (8h of 5m high/low/close relative to the entry = next 5m open) and fwd72
(72h of hourly high/low/close), plus the walk-forward CNN scores of the #10 study.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
VS = HERE.parent / "volume_spike"
CACHE_ALL = VS / "cache_all"
DATA_DIR = Path(os.environ.get("MOONSHOT_DATA_DIR", "/Users/mike/code/crypto-moonshot/data"))
OWN = DATA_DIR / "cache" / "capitulation_ls"
OWN.mkdir(parents=True, exist_ok=True)
RESULTS = HERE / "results"
I10 = Path(
    "/Users/mike/code/crypto-moonshot/.claude/worktrees/bridge-cse_015bssAUhKxTmkwHKHPLsH9P"
    "/research/volume_spike/cache_all"
)
SPOT_COST = 0.004  # round trip: fee 0.1% + slippage 0.05% + latency 0.05% per side
PERP_COST = 0.003  # round trip: taker 0.05% + slippage 0.05% + latency 0.05% per side


@cache
def meta() -> pl.DataFrame:
    m = pl.read_parquet(CACHE_ALL / "seq2_meta.parquet")
    return m.with_columns(
        year=pl.col("open_time").dt.year(), day=pl.col("open_time").dt.date()
    )


@cache
def fwd2() -> np.ndarray:
    return np.load(CACHE_ALL / "fwd2.npy", mmap_mode="r")


@cache
def fwd72() -> np.ndarray:
    return np.load(CACHE_ALL / "fwd72.npy", mmap_mode="r")


def scores(name: str, key: str = "pct") -> np.ndarray:
    """Walk-forward percentile scores (NaN outside the 2022-26 test folds)."""
    for d in (OWN / "scores", CACHE_ALL / "wf_i10", I10 / "wf_i10", I10 / "wf", CACHE_ALL / "wf"):
        p = d / f"{name}.npz"
        if p.exists():
            return np.load(p)[key]
    raise FileNotFoundError(name)


def first_per_event(mask: np.ndarray, lock_s: int = 86400) -> np.ndarray:
    """Rows of `mask`, keeping a coin's first spike and then nothing for `lock_s` seconds."""
    m = meta()
    t = m["open_time"].dt.epoch("s").to_numpy()
    sym = m["symbol"].to_numpy()
    idx = np.flatnonzero(mask)
    idx = idx[np.lexsort((t[idx], sym[idx]))]
    keep, last_sym, last_t = [], None, -1e18
    for i in idx:
        if sym[i] != last_sym or t[i] - last_t >= lock_s:
            keep.append(i)
            last_sym, last_t = sym[i], t[i]
    return np.array(sorted(keep), dtype=np.int64)
