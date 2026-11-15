"""M1: labels and trade results of every panel entry, from the 5m candles.

    uv run --group research python build_labels.py [--workers 8]

Reads `cache/panel.parquet` and writes the labels of discovery and gap entries (D before
2024-01-01) to `cache/labels.parquet`, and those of every entry with D >= 2024-01-01 only to
`cache/sealed/labels.parquet`. It prints row counts only, never an outcome.

Definitions (the protocol is in `docs/plans/rebound-factors.md`), on the 5m candles opening in
the window `[D, D + 30 days)`; levels are relative to `P`, the panel's reference price:

- `hit10` / `hit20`: the first candle whose high reaches `1.30 P` comes before the first whose
  low reaches `0.90 P` / `0.80 P`; when one candle reaches both, the stop counts first.
  `hit_ns`: some candle's high reaches `1.30 P`.
- `peak7`, `trough7`, `peak30`, `trough30`: highest high / lowest low over the first 7 and 30
  days, as a fraction of `P` minus one. `h_target`, `h_stop10`, `h_stop20`: hours from `D` to the
  open of the first candle reaching the level (null if none).
- Trade per stop variant `10`, `20`, `ns`: market buy at the open of the first candle in the
  window; a resting limit sell at `1.30 P` fills at exactly that price, no slippage (also on a
  gap above it); the stop is a market sell when a low reaches the stop price, filled at the stop
  price or at the candle's open if it opened below (a gap, or the entry candle opening below
  it); otherwise a market sell at the close of the last candle in the window. Fee 0.1% per
  side, 0.1% slippage on the market buy, stop and time exits:
  `ret = exit (1 - slip if market) (1 - fee) / (entry (1 + slip) (1 + fee)) - 1`.
- Missing 5m candles are simply absent. If the candles stop more than a day before the window
  end (a delisting), the entry is kept, the trade exits at the last close in the window, the
  reason is `series_end` instead of `time` and `stopped_early` is set. An entry with no candle
  in the window has reason `no_entry` and null outcomes.

`label_entry` is a pure function over one coin's arrays; it reads only candles inside the window,
which `check_panel.py` tests on synthetic candles and checks against a slow loop on raw candles.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from common import (
    DAY,
    F64,
    FEE,
    FIVE,
    HOUR,
    I64,
    LABELS,
    PANEL,
    SEALED_LABELS,
    SEALED_START_US,
    SLIPPAGE,
    STOPS,
    TARGET,
    WINDOW_7_US,
    WINDOW_US,
    read_5m,
)

type Value = float | int | bool | str | None

LABEL_SCHEMA: dict[str, pl.DataType] = {
    "n_candles": pl.Int32(),
    "stopped_early": pl.Boolean(),
    "entry_t": pl.Int64(),
    "entry_open": pl.Float64(),
    "hit10": pl.Boolean(),
    "hit20": pl.Boolean(),
    "hit_ns": pl.Boolean(),
    "peak7": pl.Float64(),
    "trough7": pl.Float64(),
    "peak30": pl.Float64(),
    "trough30": pl.Float64(),
    "h_target": pl.Float64(),
    "h_stop10": pl.Float64(),
    "h_stop20": pl.Float64(),
    **{
        f"{k}_{v}": dt
        for v in STOPS
        for k, dt in (
            ("reason", pl.String()),
            ("exit_t", pl.Int64()),
            ("exit_price", pl.Float64()),
            ("ret", pl.Float64()),
        )
    },
}
KEY_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String(),
    "d_us": pl.Int64(),
    "year": pl.Int32(),
    "split": pl.String(),
}


def _first(mask: npt.NDArray[np.bool_]) -> int:
    """Index of the first true element, or len(mask) if none."""
    if len(mask) == 0:
        return 0
    i = int(mask.argmax())
    return i if mask[i] else len(mask)


def label_entry(t: I64, o: F64, h: F64, low: F64, c: F64, d_us: int, p: float) -> dict[str, Value]:
    """Labels and trade results of the entry at `D = d_us` with reference price `p`.

    `t`, `o`, `h`, `low`, `c`: one coin's 5m candles (open time in µs and OHLC), sorted by `t`.
    Only the candles opening in `[d_us, d_us + 30 days)` are read.
    """
    lo, hi = (int(x) for x in np.searchsorted(t, [d_us, d_us + WINDOW_US], side="left"))
    n = hi - lo
    out: dict[str, Value] = dict.fromkeys(LABEL_SCHEMA)
    out["n_candles"] = n
    if n == 0:
        for v in STOPS:
            out[f"reason_{v}"] = "no_entry"
        return out
    tw, ow, hw, lw = t[lo:hi], o[lo:hi], h[lo:hi], low[lo:hi]
    target = (1 + TARGET) * p
    i_t = _first(hw >= target)
    i_stop = {v: (n if s is None else _first(lw <= (1 - s) * p)) for v, s in STOPS.items()}
    n7 = int(np.searchsorted(tw, d_us + WINDOW_7_US, side="left"))
    last_t, last_close = int(tw[-1]), float(c[hi - 1])
    stopped_early = last_t + FIVE < d_us + WINDOW_US - DAY
    entry = float(ow[0])

    def hours(i: int) -> float | None:
        return (int(tw[i]) - d_us) / HOUR if i < n else None

    out.update(
        stopped_early=stopped_early,
        entry_t=int(tw[0]),
        entry_open=entry,
        hit10=i_t < i_stop["10"],
        hit20=i_t < i_stop["20"],
        hit_ns=i_t < n,
        peak7=float(hw[:n7].max()) / p - 1 if n7 else None,  # null: no candle in 7 days
        trough7=float(lw[:n7].min()) / p - 1 if n7 else None,
        peak30=float(hw.max()) / p - 1,
        trough30=float(lw.min()) / p - 1,
        h_target=hours(i_t),
        h_stop10=hours(i_stop["10"]),
        h_stop20=hours(i_stop["20"]),
    )
    buy = entry * (1 + SLIPPAGE) * (1 + FEE)
    for v, s in STOPS.items():
        i_s = i_stop[v]
        if i_t < i_s:
            reason, exit_t, price, sell = "target", int(tw[i_t]), target, target * (1 - FEE)
        elif i_s < n:
            assert s is not None
            price = min((1 - s) * p, float(ow[i_s]))
            reason, exit_t = "stop", int(tw[i_s])
            sell = price * (1 - SLIPPAGE) * (1 - FEE)
        else:
            reason = "series_end" if stopped_early else "time"
            exit_t, price = last_t + FIVE, last_close
            sell = price * (1 - SLIPPAGE) * (1 - FEE)
        out[f"reason_{v}"] = reason
        out[f"exit_t_{v}"] = exit_t
        out[f"exit_price_{v}"] = price
        out[f"ret_{v}"] = sell / buy - 1
    return out


def coin_labels(job: tuple[str, list[int], list[float]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(discovery and gap labels, sealed labels) of one coin's panel entries."""
    symbol, d_list, p_list = job
    c5 = read_5m(symbol)
    t = c5["t"].to_numpy().astype(np.int64)
    o, h, lo, c = (c5[k].to_numpy().astype(np.float64) for k in ("open", "high", "low", "close"))
    rows = [label_entry(t, o, h, lo, c, d, p) for d, p in zip(d_list, p_list, strict=True)]
    df = pl.DataFrame(rows, schema=LABEL_SCHEMA, orient="row")
    df = df.with_columns(symbol=pl.lit(symbol), d_us=pl.Series(d_list, dtype=pl.Int64))
    sealed = pl.col("d_us") >= SEALED_START_US
    return df.filter(~sealed), df.filter(sealed)


def write_atomic(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    t0 = time.perf_counter()
    panel = pl.read_parquet(PANEL)
    jobs = [
        (str(g["symbol"][0]), g["d_us"].to_list(), g["P"].to_list())
        for g in panel.sort("d_us").partition_by("symbol", maintain_order=True)
    ]
    jobs.sort(key=lambda j: -len(j[1]))  # longest first, for a balanced pool
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        parts = list(pool.map(coin_labels, jobs))
    keys = panel.select(list(KEY_SCHEMA))
    cols = [*KEY_SCHEMA, *LABEL_SCHEMA]
    discovery = pl.concat([a for a, _ in parts]).join(keys, on=["symbol", "d_us"], how="left")
    sealed = pl.concat([b for _, b in parts]).join(keys, on=["symbol", "d_us"], how="left")
    discovery = discovery.select(cols).sort("d_us", "symbol")
    sealed = sealed.select(cols).sort("d_us", "symbol")
    write_atomic(discovery, LABELS)
    write_atomic(sealed, SEALED_LABELS)
    secs = time.perf_counter() - t0
    print(f"{len(jobs)} coins, panel {panel.height} rows, built in {secs:.0f}s")
    print(f"labels.parquet: {discovery.height} rows (D before 2024)")
    print(f"sealed/labels.parquet: {sealed.height} rows (D from 2024; outcomes not shown)")


if __name__ == "__main__":
    main()
