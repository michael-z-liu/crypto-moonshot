"""Forward paths over 24 hours (288 bars) for the crash subset of the event table.

Output (in cache/): long.parquet (the selected event rows) and long_fwd.npy, float32
[rows, 3, LONG] of (high, low, close) / entry - 1 for candles t+1 .. t+LONG.

Run: uv run python research/volume_spike/build_long.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from build_events import full_grid
from common import BARS_PER_DAY, CACHE, load

LONG = BARS_PER_DAY


def crash_subset() -> pl.DataFrame:
    events = pl.read_parquet(CACHE / "events.parquet")
    return events.filter(
        (pl.col("is_spike") | pl.col("is_random"))
        & ((pl.col("ret_4h") <= -0.05) | (pl.col("ret_24h") <= -0.08))
    )


def paths(args: tuple[str, list]) -> tuple[list, np.ndarray]:
    symbol, times = args
    df = full_grid(load(symbol))
    t = df["open_time"].dt.replace_time_zone(None).to_numpy()
    want = pl.Series(times).dt.replace_time_zone(None).to_numpy()
    pos = np.searchsorted(t, want)
    assert (t[pos] == want).all()
    hi, lo, cl, op = (df[c].to_numpy() for c in ("high", "low", "close", "open"))
    n = len(t)
    out = np.full((len(pos), 3, LONG), np.nan, dtype=np.float32)
    for j, i in enumerate(pos):
        if i + 1 + LONG > n:
            continue
        entry = op[i + 1]
        w = slice(i + 1, i + 1 + LONG)
        out[j] = np.stack([hi[w], lo[w], cl[w]]) / entry - 1
    return times, out


def main() -> None:
    sub = crash_subset().sort("symbol", "open_time")
    groups = sub.group_by("symbol", maintain_order=True).agg(pl.col("open_time"))
    jobs = [(s, list(ts)) for s, ts in groups.iter_rows()]
    with ProcessPoolExecutor(max_workers=8) as pool:
        arrays = [a for _, a in pool.map(paths, jobs)]
    fwd = np.concatenate(arrays)
    ok = ~np.isnan(fwd).any(axis=(1, 2))
    sub.filter(pl.Series(ok)).write_parquet(CACHE / "long.parquet")
    np.save(CACHE / "long_fwd.npy", fwd[ok])
    print(f"{ok.sum()} of {len(ok)} rows kept")


if __name__ == "__main__":
    main()
