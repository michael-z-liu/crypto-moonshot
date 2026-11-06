"""30-day outcomes for every event row (B2, long shots): when +20/+30/+50/+100% is first reached.

For each row of events.parquet (spikes and random bars), entry is the open of the next 5m candle.
On the gap-free 5m grid (a missing candle carries the last close; after a series ends the last
close holds, so a delisted coin is valued at its last traded price) it stores:

- hit_20 / hit_30 / hit_50 / hit_100: candles from entry until the high first reaches the target
  (0 = the entry candle itself), or -1 if not within 30 days
- c_1d / c_3d / c_7d / c_30d: close at the end of each window, relative to entry
- lo_1d / lo_3d / lo_7d / lo_30d: lowest low within each window, relative to entry
- hi_30d: highest high within 30 days

Output: cache[_<universe>]/long30.parquet (one row per event row, column `row`).

Run: MOONSHOT_UNIVERSE=all uv run python research/volume_spike/build_long30.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from build_events import full_grid
from common import BARS_PER_DAY, CACHE, load

TARGETS = {"hit_20": 0.2, "hit_30": 0.3, "hit_50": 0.5, "hit_100": 1.0}
WINDOWS = {"1d": 1, "3d": 3, "7d": 7, "30d": 30}
MAX_BARS = 30 * BARS_PER_DAY
LEVELS = 14  # 2**13 = 8192 < 8640 <= 2**14


def sparse(a: np.ndarray, op: np.ufunc) -> list[np.ndarray]:
    """st[k][j] = op over a[j : j + 2**k] (a is padded, so every block fits)."""
    st = [a]
    for k in range(1, LEVELS + 1):
        prev, h = st[-1], 1 << (k - 1)
        nxt = prev.copy()
        nxt[: len(prev) - h] = op(prev[: len(prev) - h], prev[h:])
        st.append(nxt)
    return st


def range_op(st: list[np.ndarray], start: np.ndarray, length: int, op: np.ufunc) -> np.ndarray:
    k = int(np.floor(np.log2(length)))
    return op(st[k][start], st[k][start + length - (1 << k)])


def first_hit(st_max: list[np.ndarray], start: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """First index j in [start, start + MAX_BARS) with high[j] >= thr, as an offset; -1 if none."""
    pos = start.copy()
    end = start + MAX_BARS
    for k in range(LEVELS - 1, -1, -1):
        step = 1 << k
        skip = (pos + step <= end) & (st_max[k][pos] < thr)
        pos = np.where(skip, pos + step, pos)
    hit = (pos < end) & (st_max[0][pos] >= thr)
    return np.where(hit, pos - start, -1).astype(np.int32)


def process(job: tuple[str, np.ndarray, np.ndarray]) -> pl.DataFrame | None:
    symbol, rows, times = job
    df = full_grid(load(symbol))
    close = df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    opn = df["open"].to_numpy().astype(np.float64)
    high = np.where(np.isnan(high), close, high)
    low = np.where(np.isnan(low), close, low)
    pad = (1 << LEVELS) + MAX_BARS + 2
    last = close[-1]
    high = np.concatenate([high, np.full(pad, last)])
    low = np.concatenate([low, np.full(pad, last)])
    close_p = np.concatenate([close, np.full(pad, last)])
    t0 = df["open_time"].dt.replace_time_zone(None).to_numpy()[0]
    i = ((times - t0) // np.timedelta64(5, "m")).astype(np.int64)
    e = i + 1  # entry candle
    entry = opn[np.minimum(e, len(opn) - 1)]
    ok = (e < len(opn)) & ~np.isnan(entry) & (entry > 0)
    if not ok.any():
        return None
    rows, e, entry = rows[ok], e[ok], entry[ok]
    st_hi = sparse(high, np.maximum)
    st_lo = sparse(low, np.minimum)
    out: dict[str, np.ndarray] = {"row": rows}
    for name, tp in TARGETS.items():
        out[name] = first_hit(st_hi, e, entry * (1 + tp))
    for name, days in WINDOWS.items():
        n = days * BARS_PER_DAY
        out[f"c_{name}"] = (close_p[e + n - 1] / entry - 1).astype(np.float32)
        out[f"lo_{name}"] = (range_op(st_lo, e, n, np.minimum) / entry - 1).astype(np.float32)
    out["hi_30d"] = (range_op(st_hi, e, MAX_BARS, np.maximum) / entry - 1).astype(np.float32)
    return pl.DataFrame(out)


def main() -> None:
    ev = pl.read_parquet(CACHE / "events.parquet", columns=["row", "symbol", "open_time"])
    jobs = [
        (s, g["row"].to_numpy(), g["open_time"].dt.replace_time_zone(None).to_numpy())
        for (s,), g in ev.group_by("symbol", maintain_order=True)
    ]
    parts = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for (s, _rows, _), res in zip(jobs, pool.map(process, jobs), strict=True):
            if res is not None:
                parts.append(res)
            print(f"{s}: {0 if res is None else res.height} rows", file=sys.stderr, flush=True)
    out = pl.concat(parts).sort("row")
    out.write_parquet(CACHE / "long30.parquet")
    print(out.height, "rows of", ev.height)


if __name__ == "__main__":
    main()
