"""Run a per-coin build step in parallel processes, one part file per coin, and combine.

Each step writes `cache/parts/<table>/<SYMBOL>.parquet` for every table it makes (atomically, so
a cut-off run resumes with the coins still missing), then concatenates the parts into
`cache/<table>.parquet`.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from common import CACHE, DATA_END, universe

END_US = int(DATA_END.timestamp()) * 1_000_000
PARTS = CACHE / "parts"
BARS = CACHE / "bars"

type Step = Callable[[str], dict[str, pl.DataFrame]]


def part_path(table: str, symbol: str) -> Path:
    return PARTS / table / f"{symbol}.parquet"


def write_atomic(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def read_part(table: str, symbol: str) -> pl.DataFrame:
    return pl.read_parquet(part_path(table, symbol))


def _work(step: Step, tables: tuple[str, ...], symbol: str) -> tuple[str, dict[str, int], float]:
    t0 = time.perf_counter()
    out = step(symbol)
    # The first table is written last: its presence marks the coin as done.
    for table in reversed(tables):
        write_atomic(out[table], part_path(table, symbol))
    return symbol, {k: v.height for k, v in out.items()}, time.perf_counter() - t0


def run(
    step: Step,
    tables: tuple[str, ...],
    description: str,
    *,
    combine: bool = True,
    show_counts: bool = True,
) -> None:
    """Build every coin not yet done, then combine. `show_counts=False` keeps row counts out of
    the log for tables whose size depends on 2025-26 outcomes (fills found, labels made)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--symbols", nargs="*", help="only these coins (default: the universe)")
    parser.add_argument("--force", action="store_true", help="rebuild coins already done")
    parser.add_argument("--no-combine", action="store_true", help="skip the combined tables")
    args = parser.parse_args()
    symbols = args.symbols or universe()
    todo = [s for s in symbols if args.force or not part_path(tables[0], s).exists()]
    print(f"{len(symbols)} coins, {len(todo)} to build, {args.workers} workers", flush=True)
    t0 = time.perf_counter()
    totals = dict.fromkeys(tables, 0)
    if todo:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
            futures = [pool.submit(_work, step, tables, s) for s in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                symbol, counts, secs = fut.result()
                for k in tables:
                    totals[k] += counts.get(k, 0)
                rows = " ".join(f"{k}={v}" for k, v in counts.items()) if show_counts else ""
                elapsed = time.perf_counter() - t0
                print(f"[{i}/{len(todo)}] {symbol} {rows} {secs:.1f}s ({elapsed:.0f}s)", flush=True)
    print(f"built in {time.perf_counter() - t0:.0f}s" + (f": {totals}" if show_counts else ""))
    if not combine or args.no_combine or args.symbols:
        return
    for table in tables:
        t1 = time.perf_counter()
        files = [part_path(table, s) for s in symbols]
        missing = [f for f in files if not f.exists()]
        if missing:
            print(f"{table}: {len(missing)} parts missing, not combined", flush=True)
            continue
        df = pl.concat([pl.read_parquet(f) for f in files], how="vertical")
        write_atomic(df, CACHE / f"{table}.parquet")
        size = f"{df.height} rows, " if show_counts else ""
        print(f"{table}.parquet: {size}{time.perf_counter() - t1:.0f}s", flush=True)
