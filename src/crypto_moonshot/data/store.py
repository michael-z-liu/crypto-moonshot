"""Local Parquet store for klines, per-series metadata and missing-range planning.

Layout (see `docs/data-format.md`)::

    <root>/klines/spot/interval=<interval>/symbol=<SYMBOL>/<YYYY-MM>.parquet
    <root>/klines/spot/interval=<interval>/symbol=<SYMBOL>/_meta.json

All writes go to a hidden temporary file in the same directory followed by `os.replace`, so
readers see either the old file or the new one. The planning functions (`missing_ranges`,
`split_by_month`, `merge_ranges`) are pure and do no I/O.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from crypto_moonshot.models import (
    EPOCH,
    KLINE_SCHEMA,
    SCHEMA_VERSION,
    Interval,
    as_utc,
    month_start,
    next_month,
)

SCHEMA_VERSION_KEY = "moonshot.schema_version"
"""Parquet key-value metadata key holding the schema version."""
META_FILENAME = "_meta.json"

_SYMBOL_RE = re.compile(r"^[A-Z0-9]+$")
_MONTH_FILE_RE = re.compile(r"^(\d{4})-(\d{2})\.parquet$")
_INTERVAL_ORDER = {interval: index for index, interval in enumerate(Interval)}

type Range = tuple[datetime, datetime]
"""A half-open `[start, end)` range of candle open times."""


class InvalidFrameError(ValueError):
    """A frame offered to the store does not satisfy the kline schema or invariants."""


class InvalidMonthFileError(Exception):
    """A month file on disk is unreadable or violates the format."""


# --------------------------------------------------------------------------------------
# Paths and validation helpers
# --------------------------------------------------------------------------------------


def _check_symbol(symbol: str) -> None:
    if not _SYMBOL_RE.fullmatch(symbol):
        msg = f"invalid symbol {symbol!r}: expected upper-case letters and digits"
        raise ValueError(msg)


def _require_month(month: datetime) -> datetime:
    month = as_utc(month)
    if month != month_start(month):
        msg = f"{month.isoformat()} is not the start of a month"
        raise ValueError(msg)
    return month


def _month_name(month: datetime) -> str:
    return f"{month.year:04d}-{month.month:02d}"


def parse_month_path(path: Path) -> tuple[str, Interval, datetime]:
    """Infer `(symbol, interval, month start)` from `.../interval=<i>/symbol=<S>/<YYYY-MM>.parquet`.

    Raises `ValueError` with a short reason when the path does not follow the layout.
    """
    if len(path.parts) < 3:
        msg = "path is too short for interval=<interval>/symbol=<SYMBOL>/<YYYY-MM>.parquet"
        raise ValueError(msg)
    interval_dir, symbol_dir, name = path.parts[-3:]
    if not interval_dir.startswith("interval="):
        msg = f"folder {interval_dir!r} is not interval=<interval>"
        raise ValueError(msg)
    try:
        interval = Interval(interval_dir.removeprefix("interval="))
    except ValueError:
        msg = f"unknown interval in folder {interval_dir!r}"
        raise ValueError(msg) from None
    if not symbol_dir.startswith("symbol="):
        msg = f"folder {symbol_dir!r} is not symbol=<SYMBOL>"
        raise ValueError(msg)
    symbol = symbol_dir.removeprefix("symbol=")
    if not _SYMBOL_RE.fullmatch(symbol):
        msg = f"invalid symbol {symbol!r} in folder name"
        raise ValueError(msg)
    match = _MONTH_FILE_RE.fullmatch(name)
    if match is None:
        msg = f"file name {name!r} is not YYYY-MM.parquet"
        raise ValueError(msg)
    year, mon = int(match.group(1)), int(match.group(2))
    if not 1 <= mon <= 12 or year < 1:
        msg = f"invalid month in file name {name!r}"
        raise ValueError(msg)
    return symbol, interval, datetime(year, mon, 1, tzinfo=UTC)


def _anchor_us(interval: Interval) -> int:
    """Microseconds since the epoch of an arbitrary aligned open time of a fixed interval."""
    return _to_us(interval.floor(EPOCH))


def _length_us(interval: Interval) -> int:
    return interval.length // timedelta(microseconds=1)


def _to_us(dt: datetime) -> int:
    return (dt - EPOCH) // timedelta(microseconds=1)


def _from_us(us: int) -> datetime:
    return EPOCH + timedelta(microseconds=us)


def _schema_problems(schema: Mapping[str, pl.DataType]) -> list[str]:
    """Reasons `schema` is not exactly `KLINE_SCHEMA` (empty list when it is)."""
    problems: list[str] = []
    missing = [name for name in KLINE_SCHEMA if name not in schema]
    extra = [name for name in schema if name not in KLINE_SCHEMA]
    if missing:
        problems.append(f"missing columns: {', '.join(missing)}")
    if extra:
        problems.append(f"unexpected columns: {', '.join(extra)}")
    for name, expected in KLINE_SCHEMA.items():
        actual = schema.get(name)
        if actual is None or actual == expected:
            continue
        if name == "open_time":
            problems.append(
                f"open_time must be Datetime(us, UTC) (timezone-aware UTC), got {actual}"
            )
        else:
            problems.append(f"column {name!r} has dtype {actual}, expected {expected}")
    if not problems and list(schema) != list(KLINE_SCHEMA):
        problems.append(f"columns out of order: {', '.join(schema)}")
    return problems


def _open_time_problems(
    open_time: pl.Series, interval: Interval, month: datetime, *, check_order: bool
) -> list[str]:
    """Reasons an `open_time` column is invalid for a month file (empty list when valid)."""
    if open_time.len() == 0:
        return ["file has no candles"]
    problems: list[str] = []
    nulls = open_time.null_count()
    if nulls:
        problems.append(f"{nulls} null open_time values")
    values = open_time.drop_nulls()
    if values.len() == 0:
        return problems
    end = next_month(month)
    outside = int(((values < month) | (values >= end)).sum())
    if outside:
        problems.append(f"{outside} open_time values outside month {_month_name(month)}")
    if interval.is_monthly:
        misaligned = int((values != month).sum())
    else:
        offset = values.dt.epoch("us") - _anchor_us(interval)
        misaligned = int((offset % _length_us(interval) != 0).sum())
    if misaligned:
        problems.append(f"{misaligned} open_time values not aligned to {interval.value}")
    duplicates = int(values.is_duplicated().sum())
    if duplicates:
        problems.append(f"{duplicates} duplicate open_time values")
    if check_order and not open_time.is_sorted():
        problems.append("open_time is not sorted ascending")
    return problems


def check_month_file(path: Path, *, now: datetime) -> list[str]:
    """Structural check of one month file; an empty list means it is valid.

    Interval, symbol and month are inferred from the path. Unreadable or corrupt files give a
    problem, never an exception. A candle must be closed by `now` (`open_time + interval <= now`).
    """
    now = as_utc(now)
    try:
        _symbol, interval, month = parse_month_path(path)
    except ValueError as exc:
        return [f"bad path: {exc}"]
    try:
        metadata = pl.read_parquet_metadata(path)
        schema = pl.read_parquet_schema(path)
    except Exception as exc:  # corrupt, truncated or not a parquet file
        return [f"unreadable parquet file: {_short(exc)}"]
    problems: list[str] = []
    version = metadata.get(SCHEMA_VERSION_KEY)
    if version is None:
        problems.append(f"missing {SCHEMA_VERSION_KEY} metadata")
    elif version != str(SCHEMA_VERSION):
        problems.append(f"schema version {version!r}, expected {str(SCHEMA_VERSION)!r}")
    schema_problems = _schema_problems(schema)
    if schema_problems:
        return [*problems, *schema_problems]
    try:
        open_time = pl.read_parquet(path, columns=["open_time"]).to_series()
    except Exception as exc:
        return [*problems, f"unreadable parquet file: {_short(exc)}"]
    data_problems = _open_time_problems(open_time, interval, month, check_order=True)
    problems.extend(data_problems)
    latest = open_time.drop_nulls().max()
    if isinstance(latest, datetime) and interval.next(latest) > now:
        problems.append(
            f"candle opening {latest.isoformat()} is not closed yet as of {now.isoformat()}"
        )
    return problems


def _short(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text[:160]}"


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Write via a hidden temp file in the same folder, then `os.replace` it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------------------
# Ranges (pure)
# --------------------------------------------------------------------------------------


def merge_ranges(ranges: Iterable[Range]) -> list[Range]:
    """Sort ranges, drop empty ones and merge overlapping or adjacent ones."""
    merged: list[Range] = []
    for start, end in sorted(ranges):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _subtract(ranges: list[Range], gaps: list[Range]) -> list[Range]:
    """`ranges` minus `gaps`; both sorted and non-overlapping."""
    out: list[Range] = []
    first = 0
    for start, end in ranges:
        cursor = start
        while first < len(gaps) and gaps[first][1] <= cursor:
            first += 1
        index = first
        while index < len(gaps) and gaps[index][0] < end:
            gap_start, gap_end = gaps[index]
            if gap_start > cursor:
                out.append((cursor, gap_start))
            cursor = max(cursor, gap_end)
            if cursor >= end:
                break
            index += 1
        if cursor < end:
            out.append((cursor, end))
    return out


def _fixed_holes(interval: Interval, lo: datetime, hi: datetime, stored: pl.Series) -> list[Range]:
    """Aligned open-time ranges in `[lo, hi)` (both aligned) not covered by `stored`."""
    length = _length_us(interval)
    lo_us, hi_us = _to_us(lo), _to_us(hi)
    times = stored.dt.epoch("us")
    times = times.filter(
        (times >= lo_us) & (times < hi_us) & ((times - _anchor_us(interval)) % length == 0)
    )
    times = times.unique().sort()
    if times.len() == 0:
        return [(lo, hi)]
    holes: list[Range] = []
    first = int(times[0])
    if first > lo_us:
        holes.append((lo, _from_us(first)))
    frame = (
        pl.DataFrame({"t": times})
        .with_columns(
            hole_start=pl.col("t") + length,
            hole_end=pl.col("t").shift(-1, fill_value=hi_us),
        )
        .filter(pl.col("hole_end") > pl.col("hole_start"))
        .select("hole_start", "hole_end")
    )
    holes.extend((_from_us(a), _from_us(b)) for a, b in frame.iter_rows())
    return holes


def _monthly_holes(
    interval: Interval, start: datetime, end: datetime, stored: pl.Series, gaps: list[Range]
) -> list[Range]:
    """`1M` has few open times, so enumerate them instead of doing interval arithmetic."""
    have = set(stored.to_list())
    holes: list[Range] = []
    gap_index = 0
    for open_time in interval.open_times(start, end):
        while gap_index < len(gaps) and gaps[gap_index][1] <= open_time:
            gap_index += 1
        in_gap = gap_index < len(gaps) and gaps[gap_index][0] <= open_time
        if open_time in have or in_gap:
            continue
        following = interval.next(open_time)
        if holes and holes[-1][1] == open_time:
            holes[-1] = (holes[-1][0], following)
        else:
            holes.append((open_time, following))
    return holes


def missing_ranges(
    interval: Interval,
    start: datetime,
    end: datetime,
    stored_open_times: pl.Series,
    gaps: Iterable[Range] = (),
) -> list[Range]:
    """Half-open ranges of aligned open times in `[start, end)` that are neither stored nor gaps.

    Returns maximal contiguous ranges, sorted. Each range starts at a missing open time and ends
    just after the last missing open time of the run. For fixed-length intervals holes are
    computed from the sorted stored times, never by enumerating every expected open time.
    """
    start, end = as_utc(start), as_utc(end)
    lo, hi = interval.ceil(start), interval.ceil(end)
    if lo >= hi:
        return []
    merged_gaps = merge_ranges(gaps)
    if interval.is_monthly:
        return _monthly_holes(interval, lo, hi, stored_open_times, merged_gaps)
    holes = _fixed_holes(interval, lo, hi, stored_open_times)
    pieces = [
        (interval.ceil(piece_start), interval.ceil(piece_end))
        for piece_start, piece_end in _subtract(holes, merged_gaps)
    ]
    return merge_ranges(pieces)


def split_by_month(ranges: Iterable[Range]) -> dict[datetime, list[Range]]:
    """Split ranges at month boundaries, keyed by month start in chronological order."""
    result: dict[datetime, list[Range]] = {}
    for start, end in ranges:
        cursor = as_utc(start)
        end = as_utc(end)
        while cursor < end:
            month = month_start(cursor)
            piece_end = min(end, next_month(month))
            result.setdefault(month, []).append((cursor, piece_end))
            cursor = piece_end
    return dict(sorted(result.items()))


# --------------------------------------------------------------------------------------
# Series metadata
# --------------------------------------------------------------------------------------


@dataclass
class SeriesMeta:
    """Contents of `_meta.json`: the listing time and known Binance gaps of one series."""

    schema_version: int = SCHEMA_VERSION
    listing: datetime | None = None
    """First open time Binance has for this symbol and interval, if looked up."""
    gaps: list[Range] = field(default_factory=list)
    """Half-open `[start, end)` open-time ranges with no candles on Binance."""

    def with_gaps(self, new: Iterable[Range]) -> SeriesMeta:
        """Copy with `new` gaps added (overlapping and adjacent ranges merged)."""
        return SeriesMeta(self.schema_version, self.listing, add_gaps(self.gaps, new))


def add_gaps(existing: Iterable[Range], new: Iterable[Range]) -> list[Range]:
    """Union of two gap lists, sorted, with overlapping and adjacent ranges merged."""
    return merge_ranges([*existing, *new])


def _iso(dt: datetime) -> str:
    return as_utc(dt).isoformat()


def _parse_dt(value: object) -> datetime:
    if not isinstance(value, str):
        msg = f"expected an ISO-8601 string, got {value!r}"
        raise TypeError(msg)
    return as_utc(datetime.fromisoformat(value))


def _meta_from_json(raw: object) -> SeriesMeta | None:
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        return None
    listing_raw = raw.get("listing")
    listing = None if listing_raw is None else _parse_dt(listing_raw)
    gaps_raw = raw.get("gaps", [])
    if not isinstance(gaps_raw, list):
        return None
    gaps: list[Range] = []
    for item in gaps_raw:
        if not isinstance(item, list) or len(item) != 2:
            return None
        gap = (_parse_dt(item[0]), _parse_dt(item[1]))
        if gap[0] >= gap[1]:
            return None
        gaps.append(gap)
    return SeriesMeta(SCHEMA_VERSION, listing, merge_ranges(gaps))


def read_meta(series_dir: Path) -> SeriesMeta:
    """Read `<series_dir>/_meta.json`; a missing, corrupt or other-version file gives defaults."""
    try:
        meta = _meta_from_json(json.loads((series_dir / META_FILENAME).read_text("utf-8")))
    except (OSError, ValueError, TypeError):
        return SeriesMeta()
    return meta if meta is not None else SeriesMeta()


def write_meta(series_dir: Path, meta: SeriesMeta) -> None:
    """Atomically write `<series_dir>/_meta.json` in the documented format."""
    document = {
        "schema_version": SCHEMA_VERSION,
        "listing": None if meta.listing is None else _iso(meta.listing),
        "gaps": [[_iso(start), _iso(end)] for start, end in merge_ranges(meta.gaps)],
    }
    text = json.dumps(document, indent=2) + "\n"

    def write(tmp: Path) -> None:
        tmp.write_text(text, "utf-8")

    _atomic_write(series_dir / META_FILENAME, write)


# --------------------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------------------


class KlineStore:
    """Monthly Parquet files of klines under a data root."""

    def __init__(self, root: Path) -> None:
        self.root = root

    # paths

    def series_dir(self, symbol: str, interval: Interval) -> Path:
        _check_symbol(symbol)
        return (
            self.root
            / "klines"
            / "spot"
            / f"interval={Interval(interval).value}"
            / f"symbol={symbol}"
        )

    def month_path(self, symbol: str, interval: Interval, month: datetime) -> Path:
        month = _require_month(month)
        return self.series_dir(symbol, interval) / f"{_month_name(month)}.parquet"

    def list_series(self) -> list[tuple[str, Interval]]:
        """Every `(symbol, interval)` that has a series folder, sorted by symbol then interval."""
        base = self.root / "klines" / "spot"
        found: list[tuple[str, Interval]] = []
        for interval_dir in base.glob("interval=*"):
            try:
                interval = Interval(interval_dir.name.removeprefix("interval="))
            except ValueError:
                continue
            for symbol_dir in interval_dir.glob("symbol=*"):
                symbol = symbol_dir.name.removeprefix("symbol=")
                if symbol_dir.is_dir() and _SYMBOL_RE.fullmatch(symbol):
                    found.append((symbol, interval))
        return sorted(found, key=lambda item: (item[0], _INTERVAL_ORDER[item[1]]))

    def list_months(self, symbol: str, interval: Interval) -> list[datetime]:
        """Start of every month that has a file (valid or not), sorted."""
        months: list[datetime] = []
        for path in self.series_dir(symbol, interval).glob("*.parquet"):
            match = _MONTH_FILE_RE.fullmatch(path.name)
            if match is None or not 1 <= int(match.group(2)) <= 12:
                continue
            months.append(datetime(int(match.group(1)), int(match.group(2)), 1, tzinfo=UTC))
        return sorted(months)

    # metadata

    def read_meta(self, symbol: str, interval: Interval) -> SeriesMeta:
        return read_meta(self.series_dir(symbol, interval))

    def write_meta(self, symbol: str, interval: Interval, meta: SeriesMeta) -> None:
        write_meta(self.series_dir(symbol, interval), meta)

    # reading

    def read_month(self, symbol: str, interval: Interval, month: datetime) -> pl.DataFrame:
        """Read one month file. Raises `FileNotFoundError` or `InvalidMonthFileError`."""
        path = self.month_path(symbol, interval, month)
        try:
            version = pl.read_parquet_metadata(path).get(SCHEMA_VERSION_KEY)
            frame = pl.read_parquet(path)
        except FileNotFoundError:
            raise
        except Exception as exc:
            msg = f"{path}: unreadable parquet file: {_short(exc)}"
            raise InvalidMonthFileError(msg) from exc
        problems = _schema_problems(frame.schema)
        if version != str(SCHEMA_VERSION):
            problems.insert(0, f"schema version {version!r}, expected {str(SCHEMA_VERSION)!r}")
        if problems:
            msg = f"{path}: {'; '.join(problems)}"
            raise InvalidMonthFileError(msg)
        return frame

    def stored_open_times(
        self, symbol: str, interval: Interval, *, months: Iterable[datetime] | None = None
    ) -> pl.Series:
        """Sorted `open_time` values across month files, reading only that column.

        `months` limits which month files are read (default: all). Pass the valid months to
        treat invalid files as missing. Raises `InvalidMonthFileError` for an unreadable file.
        """
        wanted = self.list_months(symbol, interval) if months is None else sorted(months)
        parts: list[pl.Series] = []
        for month in wanted:
            path = self.month_path(symbol, interval, month)
            try:
                column = pl.read_parquet(path, columns=["open_time"]).to_series()
            except Exception as exc:
                msg = f"{path}: unreadable parquet file: {_short(exc)}"
                raise InvalidMonthFileError(msg) from exc
            if column.dtype != KLINE_SCHEMA["open_time"]:
                msg = f"{path}: open_time has dtype {column.dtype}"
                raise InvalidMonthFileError(msg)
            parts.append(column)
        if not parts:
            return pl.Series("open_time", [], dtype=KLINE_SCHEMA["open_time"])
        return pl.concat(parts).sort()

    def invalid_months(
        self, symbol: str, interval: Interval, *, now: datetime
    ) -> dict[datetime, list[str]]:
        """Month files that fail `check_month_file`, with their problems."""
        invalid: dict[datetime, list[str]] = {}
        for month in self.list_months(symbol, interval):
            problems = check_month_file(self.month_path(symbol, interval, month), now=now)
            if problems:
                invalid[month] = problems
        return invalid

    # writing

    def _prepare(
        self, interval: Interval, month: datetime, frame: pl.DataFrame, *, unique: bool
    ) -> pl.DataFrame:
        """Validate a frame for a month and return it sorted (and de-duplicated if not `unique`)."""
        problems = _schema_problems(frame.schema)
        if problems:
            msg = f"frame does not match the kline schema: {'; '.join(problems)}"
            raise InvalidFrameError(msg)
        frame = frame.sort("open_time")
        if unique:
            problems = _open_time_problems(frame["open_time"], interval, month, check_order=False)
        else:
            frame = frame.unique(subset="open_time", keep="first", maintain_order=True)
            problems = (
                _open_time_problems(frame["open_time"], interval, month, check_order=False)
                if frame.height
                else []
            )
        if problems:
            msg = f"invalid candles for {_month_name(month)}: {'; '.join(problems)}"
            raise InvalidFrameError(msg)
        return frame

    def _write_prepared(
        self, symbol: str, interval: Interval, month: datetime, frame: pl.DataFrame
    ) -> None:
        path = self.month_path(symbol, interval, month)
        metadata = {SCHEMA_VERSION_KEY: str(SCHEMA_VERSION)}
        _atomic_write(path, lambda tmp: frame.write_parquet(tmp, metadata=metadata))

    def write_month(
        self, symbol: str, interval: Interval, month: datetime, frame: pl.DataFrame
    ) -> None:
        """Validate and atomically write (replacing) one month file.

        The frame must have exactly `KLINE_SCHEMA`; its open times must be unique, aligned and
        inside `month`. Rows are sorted by `open_time`. Raises `InvalidFrameError` otherwise.
        """
        month = _require_month(month)
        prepared = self._prepare(interval, month, frame, unique=True)
        self._write_prepared(symbol, interval, month, prepared)

    def merge_month(
        self, symbol: str, interval: Interval, month: datetime, frame: pl.DataFrame
    ) -> int:
        """Merge candles into the month file; existing candles win. Returns candles added.

        Duplicates within `frame` are collapsed (first wins). The file is only rewritten when
        something was added. Raises `InvalidMonthFileError` if the existing file is invalid
        (callers should overwrite such months with `write_month`).
        """
        month = _require_month(month)
        new = self._prepare(interval, month, frame, unique=False)
        path = self.month_path(symbol, interval, month)
        if not path.exists():
            if new.height == 0:
                return 0
            self._write_prepared(symbol, interval, month, new)
            return new.height
        existing = self.read_month(symbol, interval, month)
        merged = (
            pl.concat([existing, new])
            .unique(subset="open_time", keep="first", maintain_order=True)
            .sort("open_time")
        )
        added = merged.height - existing.height
        if added:
            self._write_prepared(symbol, interval, month, merged)
        return added
