"""Exit simulation and summary statistics over the cached event table."""

from __future__ import annotations

import numpy as np
import polars as pl
from common import CACHE, split_of

COST = 0.003  # round trip: 0.1% fee + 0.05% slippage, each side


def load_events() -> tuple[pl.DataFrame, np.ndarray]:
    events = pl.read_parquet(CACHE / "events.parquet").with_columns(split_of())
    fwd = np.load(CACHE / "fwd.npy", mmap_mode="r")
    return events, fwd


def triple_barrier(
    fwd: np.ndarray, tp: float | None, sl: float | None, bars: int, cost: float = COST
) -> np.ndarray:
    """Net return per event for a take-profit / stop-loss / time-stop exit.

    fwd is [n, 3, horizon] of (high, low, close) relative to entry. Barriers are checked bar by
    bar; if both are touched in the same bar the stop is assumed to come first. With no barrier
    hit, the trade closes at the close of bar `bars`.
    """
    hi = np.asarray(fwd[:, 0, :bars], dtype=np.float32)
    lo = np.asarray(fwd[:, 1, :bars], dtype=np.float32)
    n = hi.shape[0]
    never = bars + 1
    t_tp = np.full(n, never)
    t_sl = np.full(n, never)
    if tp is not None:
        hit = hi >= tp
        t_tp = np.where(hit.any(1), hit.argmax(1), never)
    if sl is not None:
        hit = lo <= -sl
        t_sl = np.where(hit.any(1), hit.argmax(1), never)
    out = np.asarray(fwd[:, 2, bars - 1], dtype=np.float64).copy()
    stopped = (t_sl <= t_tp) & (t_sl < never)
    took = (t_tp < t_sl) & (t_tp < never)
    if sl is not None:
        out[stopped] = -sl
    if tp is not None:
        out[took] = tp
    return out - cost


def summary(ret: np.ndarray) -> dict[str, float]:
    n = len(ret)
    if n == 0:
        return {"n": 0, "mean_bp": np.nan, "t": np.nan, "win": np.nan}
    m = float(ret.mean())
    s = float(ret.std(ddof=1)) if n > 1 else np.nan
    return {
        "n": n,
        "mean_bp": m * 1e4,
        "t": m / s * np.sqrt(n) if s and s > 0 else np.nan,
        "win": float((ret > 0).mean()),
    }
