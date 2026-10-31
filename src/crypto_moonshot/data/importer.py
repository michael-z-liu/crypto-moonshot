"""Import klines from another data folder into a store, after structural checks.

The source tree is only read. Every `*.parquet` under `<source>/klines/spot/` is either imported or
skipped with its reasons; there is no upgrade logic, so a file that is not valid for the current
schema is left for the next `data sync` to download again. The data itself is trusted.

- Destination month absent: the source file is written (`imported`).
- Destination month valid: candles are merged, existing candles win (`merged`, or `unchanged`
  when nothing was new).
- Destination month invalid: replaced by the valid source file (`replaced`).

`_meta.json` of a source series is merged in when it is valid for the current schema version:
gaps are unioned and the destination's listing is kept if set. The function does no printing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl

from crypto_moonshot.data.store import (
    META_FILENAME,
    KlineStore,
    SeriesMeta,
    add_gaps,
    check_month_file,
    load_meta,
    parse_month_path,
)
from crypto_moonshot.models import Interval, as_utc

_INTERVAL_ORDER = {interval: index for index, interval in enumerate(Interval)}
_SPOT_PARTS = 3
"""A file sits at `interval=<i>/symbol=<S>/<YYYY-MM>.parquet` directly under `klines/spot`."""


class ImportSourceError(Exception):
    """The source folder cannot be imported from (a usage error)."""


class ImportAction(StrEnum):
    IMPORTED = "imported"
    MERGED = "merged"
    REPLACED = "replaced"
    UNCHANGED = "unchanged"
    SKIPPED = "skipped"


class MetaAction(StrEnum):
    NONE = "none"
    """The source series has no `_meta.json`."""
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    SKIPPED = "skipped"
    """The source `_meta.json` is not valid for the current schema version."""


@dataclass(frozen=True)
class FileImport:
    """What happened to one source file."""

    path: str
    """Path relative to the source root, with `/` separators."""
    symbol: str | None
    interval: Interval | None
    month: datetime | None
    """`None` for all three when the path does not follow the layout."""
    action: ImportAction
    candles_added: int
    problems: tuple[str, ...]
    """Why the file was skipped (empty unless `action` is `skipped`)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "interval": None if self.interval is None else self.interval.value,
            "month": None if self.month is None else f"{self.month:%Y-%m}",
            "action": self.action.value,
            "candles_added": self.candles_added,
            "problems": list(self.problems),
        }


@dataclass(frozen=True)
class SeriesImport:
    """Per-series aggregate of the file results and the `_meta.json` outcome."""

    symbol: str
    interval: Interval
    imported: int
    merged: int
    replaced: int
    unchanged: int
    skipped: int
    candles_added: int
    meta: MetaAction
    meta_problems: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval.value,
            "imported": self.imported,
            "merged": self.merged,
            "replaced": self.replaced,
            "unchanged": self.unchanged,
            "skipped": self.skipped,
            "candles_added": self.candles_added,
            "meta": self.meta.value,
            "meta_problems": list(self.meta_problems),
        }


@dataclass(frozen=True)
class ImportReport:
    source: Path
    destination: Path
    dry_run: bool
    files: tuple[FileImport, ...]
    series: tuple[SeriesImport, ...]

    def count(self, action: ImportAction) -> int:
        return sum(1 for item in self.files if item.action is action)

    @property
    def candles_added(self) -> int:
        return sum(item.candles_added for item in self.files)

    def totals(self) -> dict[str, int]:
        return {
            "files": len(self.files),
            **{action.value: self.count(action) for action in ImportAction},
            "candles_added": self.candles_added,
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict: datetimes as ISO strings or `YYYY-MM`, enums as their values."""
        return {
            "source": str(self.source),
            "destination": str(self.destination),
            "dry_run": self.dry_run,
            "totals": self.totals(),
            "series": [item.to_dict() for item in self.series],
            "files": [item.to_dict() for item in self.files],
        }


# --------------------------------------------------------------------------------------
# Source validation
# --------------------------------------------------------------------------------------


def check_source(source_root: Path, destination_root: Path) -> None:
    """Raise `ImportSourceError` with a clear message when `source_root` is not importable."""
    if not source_root.exists():
        msg = f"{source_root} does not exist"
        raise ImportSourceError(msg)
    if not source_root.is_dir():
        msg = f"{source_root} is not a folder"
        raise ImportSourceError(msg)
    if source_root.resolve() == destination_root.resolve():
        msg = f"{source_root} is the destination data folder; pick another folder to import from"
        raise ImportSourceError(msg)
    if not (source_root / "klines").is_dir():
        msg = f"{source_root} has no klines/ folder, so it is not a data folder"
        resolved = source_root.resolve()
        if resolved.name == "klines":
            msg += f"; did you mean {resolved.parent}?"
        elif resolved.name == "spot" and resolved.parent.name == "klines":
            msg += f"; did you mean {resolved.parent.parent}?"
        raise ImportSourceError(msg)


# --------------------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------------------


def _source_files(source_root: Path) -> list[Path]:
    base = source_root / "klines" / "spot"
    return sorted(
        path
        for path in base.rglob("*.parquet")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(base).parts)
    )


def _skipped(
    relative: str,
    parsed: tuple[str, Interval, datetime] | None,
    problems: list[str],
) -> FileImport:
    symbol, interval, month = parsed if parsed is not None else (None, None, None)
    return FileImport(relative, symbol, interval, month, ImportAction.SKIPPED, 0, tuple(problems))


def _import_file(
    source_root: Path, path: Path, dest: KlineStore, *, now: datetime, dry_run: bool
) -> FileImport:
    relative = path.relative_to(source_root).as_posix()
    try:
        parsed = parse_month_path(path)
    except ValueError as exc:
        return _skipped(relative, None, [f"bad path: {exc}"])
    if len(path.relative_to(source_root / "klines" / "spot").parts) != _SPOT_PARTS:
        problems = [
            "bad path: not interval=<interval>/symbol=<SYMBOL>/<YYYY-MM>.parquet in klines/spot"
        ]
        return _skipped(relative, None, problems)
    problems = check_month_file(path, now=now)
    if problems:
        return _skipped(relative, parsed, problems)
    symbol, interval, month = parsed
    try:
        frame = pl.read_parquet(path)
    except Exception as exc:  # the file changed or broke since it was checked
        return _skipped(relative, parsed, [f"unreadable parquet file: {type(exc).__name__}"])

    def result(action: ImportAction, added: int) -> FileImport:
        return FileImport(relative, symbol, interval, month, action, added, ())

    target = dest.month_path(symbol, interval, month)
    if not target.exists():
        if not dry_run:
            dest.write_month(symbol, interval, month, frame)
        return result(ImportAction.IMPORTED, frame.height)
    if check_month_file(target, now=now):
        if not dry_run:
            dest.write_month(symbol, interval, month, frame)
        return result(ImportAction.REPLACED, frame.height)
    existing = dest.read_month(symbol, interval, month)["open_time"]
    added = int((~frame["open_time"].is_in(existing.implode())).sum())
    if added and not dry_run:
        added = dest.merge_month(symbol, interval, month, frame)
    return result(ImportAction.MERGED if added else ImportAction.UNCHANGED, added)


def _import_meta(
    source_root: Path, symbol: str, interval: Interval, dest: KlineStore, *, dry_run: bool
) -> tuple[MetaAction, tuple[str, ...]]:
    source_dir = source_root / "klines" / "spot" / f"interval={interval.value}" / f"symbol={symbol}"
    if not (source_dir / META_FILENAME).is_file():
        return MetaAction.NONE, ()
    incoming = load_meta(source_dir)
    if incoming is None:
        return MetaAction.SKIPPED, (f"{META_FILENAME} is not valid for the current schema version",)
    current = dest.read_meta(symbol, interval)
    merged = SeriesMeta(
        listing=current.listing if current.listing is not None else incoming.listing,
        gaps=add_gaps(current.gaps, incoming.gaps),
    )
    if (merged.listing, merged.gaps) == (current.listing, current.gaps):
        return MetaAction.UNCHANGED, ()
    if not dry_run:
        dest.write_meta(symbol, interval, merged)
    return MetaAction.UPDATED, ()


def import_store(
    source_root: Path, dest: KlineStore, *, now: datetime, dry_run: bool = False
) -> ImportReport:
    """Import every valid month file under `source_root` into `dest`; skip the rest.

    `now` decides which candles count as closed. With `dry_run` nothing is written, and the
    report shows exactly what a real run would do. Raises `ImportSourceError` for an unusable
    `source_root`.
    """
    now = as_utc(now)
    check_source(source_root, dest.root)
    files = tuple(
        _import_file(source_root, path, dest, now=now, dry_run=dry_run)
        for path in _source_files(source_root)
    )
    grouped: dict[tuple[str, Interval], list[FileImport]] = {}
    for item in files:
        if item.symbol is not None and item.interval is not None:
            grouped.setdefault((item.symbol, item.interval), []).append(item)
    series: list[SeriesImport] = []
    for (symbol, interval), items in sorted(
        grouped.items(), key=lambda entry: (entry[0][0], _INTERVAL_ORDER[entry[0][1]])
    ):
        usable = any(item.action is not ImportAction.SKIPPED for item in items)
        meta, meta_problems = (
            _import_meta(source_root, symbol, interval, dest, dry_run=dry_run)
            if usable
            else (MetaAction.NONE, ())
        )
        series.append(
            SeriesImport(
                symbol=symbol,
                interval=interval,
                imported=sum(item.action is ImportAction.IMPORTED for item in items),
                merged=sum(item.action is ImportAction.MERGED for item in items),
                replaced=sum(item.action is ImportAction.REPLACED for item in items),
                unchanged=sum(item.action is ImportAction.UNCHANGED for item in items),
                skipped=sum(item.action is ImportAction.SKIPPED for item in items),
                candles_added=sum(item.candles_added for item in items),
                meta=meta,
                meta_problems=meta_problems,
            )
        )
    return ImportReport(source_root, dest.root, dry_run, files, tuple(series))
