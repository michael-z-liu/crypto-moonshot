"""Run a `SignalRule` over a universe of stored series: in parallel, and cached.

`candidates` computes every symbol's candidate rows in a process pool (each worker reads one
symbol's whole series from the store) and joins the rule's cross-symbol context by open time.
`compute_signals` applies the rule's `select` and caches the resulting signal frame as Parquet
under `<cache_dir>/signals/`, keyed by the rule's `key`, the symbols, the data end (`until`),
the data root and a signature of the month files read (names, sizes, modification times): a
change to any of them recomputes, anything else reuses the file. A JSON file with the key sits
next to each cached frame.

Candidates are not cached: on the `all` universe they are ~10M volume spikes (~0.5 GB), while a
rule's signals are a few thousand rows and recomputing candidates takes ~20 s.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import polars as pl

from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.models import KLINE_SCHEMA, Interval, as_utc
from crypto_moonshot.strategy.base import SignalRule, Signals

CACHE_VERSION = 1
"""Bump when the signal computation changes in a way the keys do not capture."""

type Progress = Callable[[str, int], None]
"""Called with each symbol and its number of candidate rows as it completes."""


@dataclass(frozen=True, slots=True)
class SignalRun:
    """A rule's signals and how they were obtained: from the cache or computed, how long it
    took, and the cache file (if caching)."""

    signals: Signals
    cached: bool
    seconds: float
    path: Path | None


def _symbol_candidates(
    rule: SignalRule, root: Path, symbol: str, until: datetime | None
) -> pl.DataFrame:
    candles = KlineStore(root).read_series(symbol, rule.interval, end=until)
    return _named(rule.candidates(candles), symbol)


def _named(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    return frame.select(pl.lit(symbol, dtype=pl.String).alias("symbol"), pl.exclude("symbol"))


def data_signature(
    root: Path, symbols: Sequence[str], interval: Interval, until: datetime | None
) -> str:
    """Hash of the month files a run reads: names, sizes and modification times."""
    store = KlineStore(root)
    digest = hashlib.sha256()
    for symbol in sorted(symbols):
        digest.update(symbol.encode())
        for month in store.list_months(symbol, interval):
            if until is not None and month >= until:
                continue
            stat = store.month_path(symbol, interval, month).stat()
            digest.update(f"{month:%Y-%m}:{stat.st_size}:{stat.st_mtime_ns};".encode())
    return digest.hexdigest()


def cache_key(
    rule: SignalRule, symbols: Sequence[str], root: Path, until: datetime | None
) -> dict[str, object]:
    """Everything a cached signal frame depends on."""
    every = sorted({*symbols, *rule.context_symbols})
    return {
        "version": CACHE_VERSION,
        "name": rule.name,
        "rule": rule.key(),
        "symbols": sorted(symbols),
        "until": None if until is None else as_utc(until).isoformat(),
        "root": str(root.resolve()),
        "data": data_signature(root, every, rule.interval, until),
    }


def candidates(
    rule: SignalRule,
    symbols: Sequence[str],
    root: Path,
    *,
    until: datetime | None = None,
    workers: int | None = None,
    progress: Progress | None = None,
) -> pl.DataFrame:
    """Every symbol's candidates with the rule's context joined on, by symbol then time.

    `until`: the data end (exclusive): later candles are not read. `workers`: processes
    (default: CPU count, at most 8; 1 runs in this process).
    """
    until = None if until is None else as_utc(until)
    parts = _compute(rule, symbols, root, until, workers, progress)
    if not parts:
        parts = [_named(rule.candidates(pl.DataFrame(schema=KLINE_SCHEMA)), "")]
    table = pl.concat(parts, how="vertical_relaxed")
    if rule.context_symbols:
        store = KlineStore(root)
        series = {s: store.read_series(s, rule.interval, end=until) for s in rule.context_symbols}
        context = rule.context(series)
        if context is not None:
            table = table.join(context, on="open_time", how="left", maintain_order="left")
    return table


def _compute(
    rule: SignalRule,
    symbols: Sequence[str],
    root: Path,
    until: datetime | None,
    workers: int | None,
    progress: Progress | None,
) -> list[pl.DataFrame]:
    workers = workers or min(8, os.cpu_count() or 1)
    ordered = sorted(symbols)
    parts: list[pl.DataFrame] = []
    if workers <= 1:
        for symbol in ordered:
            part = _symbol_candidates(rule, root, symbol, until)
            if progress is not None:
                progress(symbol, part.height)
            parts.append(part)
        return parts
    # spawn, not fork: polars' thread pool is not fork-safe. Callers' scripts need the usual
    # `if __name__ == "__main__":` guard.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        n = len(ordered)
        results = pool.map(_symbol_candidates, [rule] * n, [root] * n, ordered, [until] * n)
        for symbol, part in zip(ordered, results, strict=True):
            if progress is not None:
                progress(symbol, part.height)
            parts.append(part)
    return parts


def compute_signals(
    rule: SignalRule,
    symbols: Sequence[str],
    root: Path,
    *,
    until: datetime | None = None,
    workers: int | None = None,
    cache_dir: Path | None = None,
    progress: Progress | None = None,
) -> SignalRun:
    """The rule's signals over `symbols`, from the cache under `cache_dir` when present (no
    caching when None). See `candidates` for the other arguments."""
    started = time.perf_counter()
    path: Path | None = None
    text = ""
    if cache_dir is not None:
        text = json.dumps(cache_key(rule, symbols, root, until), indent=2, sort_keys=True)
        digest = hashlib.sha256(text.encode()).hexdigest()[:24]
        path = cache_dir / "signals" / f"{rule.name}-{digest}.parquet"
        if path.exists():
            frame = pl.read_parquet(path)
            signals = Signals(frame, rule.exit, rule.interval)
            return SignalRun(signals, True, time.perf_counter() - started, path)
    table = candidates(rule, symbols, root, until=until, workers=workers, progress=progress)
    frame = rule.select(table)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.name}.tmp"
        frame.write_parquet(tmp)
        os.replace(tmp, path)
        path.with_suffix(".json").write_text(text + "\n", "utf-8")
    signals = Signals(frame, rule.exit, rule.interval)
    return SignalRun(signals, False, time.perf_counter() - started, path)
