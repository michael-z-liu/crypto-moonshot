"""Parallel kline sync: fetch only the candles a series is missing and store them by month.

`sync_klines` plans every (symbol, interval) series, then runs one unit of work per month that has
missing candles. Units of all series run concurrently under one global limit; the source (a
`BinanceClient`) paces the requests themselves. Progress is reported as small frozen event
dataclasses to an optional sink, so this module never renders or prints anything.

Failure handling:

- A non-fatal error in one series (an unknown symbol, a bad response, an unexpected exception)
  marks that series `failed`, cancels its remaining months and leaves the other series running.
- `RateLimitAbortError` or `IpBannedError` cancels every series. Unfinished series are `aborted`
  and `SyncSummary.aborted` says why.
- Cancelling the caller's task (Ctrl-C) cancels every unit and re-raises `CancelledError`. A month
  write already under way is allowed to finish first, so no temporary files are left behind.

Months already written stay intact in every case: each write is atomic.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal, Protocol

import polars as pl

from crypto_moonshot.data.binance import (
    BinanceError,
    ClientStats,
    IpBannedError,
    RateLimitAbortError,
    RateLimitHit,
    RateLimitHook,
    empty_klines,
)
from crypto_moonshot.data.store import (
    KlineStore,
    Range,
    SeriesMeta,
    missing_ranges,
    split_by_month,
)
from crypto_moonshot.models import Interval, as_utc, month_start

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 8
MIN_GAP_AGE = timedelta(minutes=5)
"""A hole must have closed at least this long ago (and two candles ago) to be recorded as a gap."""


# --------------------------------------------------------------------------------------
# Source protocol
# --------------------------------------------------------------------------------------


class KlineSource(Protocol):
    """What sync needs from a kline source. `BinanceClient` implements it; tests use fakes."""

    on_rate_limit: RateLimitHook | None
    """Sync installs its own hook for the duration of a run (chaining any existing one)."""

    @property
    def stats(self) -> ClientStats: ...

    async def fetch_klines(
        self,
        symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
        *,
        on_page: Callable[[int], None] | None = None,
    ) -> pl.DataFrame: ...

    async def first_open_time(self, symbol: str, interval: Interval) -> datetime | None: ...


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


class SeriesStatus(StrEnum):
    """Outcome of one series."""

    OK = "ok"
    """Every planned month was fetched and stored."""
    UP_TO_DATE = "up_to_date"
    """Nothing was missing; no candles were requested."""
    NO_DATA = "no_data"
    """Binance has no candles at all for this symbol and interval."""
    FAILED = "failed"
    """This series hit an error; `error` says which. Other series were not affected."""
    ABORTED = "aborted"
    """The whole run was aborted (rate limit or IP ban) before this series finished."""


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else as_utc(dt).isoformat()


@dataclass(frozen=True)
class SeriesResult:
    """What sync did to one (symbol, interval) series."""

    symbol: str
    interval: Interval
    status: SeriesStatus
    months_to_fetch: int = 0
    """Months that had missing candles when the series was planned."""
    candles_to_fetch: int = 0
    """Missing candles when the series was planned (some may turn out to be gaps)."""
    candles_added: int = 0
    months_written: int = 0
    gap_candles_found: int = 0
    """Candles newly recorded as gaps in `_meta.json` (Binance has none for them)."""
    first_open_time: datetime | None = None
    """Earliest stored candle after the sync (None if none, or unknown)."""
    last_open_time: datetime | None = None
    """Latest stored candle after the sync (None if none, or unknown)."""
    error: str | None = None
    """Why the series failed or was aborted."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict: datetimes as ISO-8601 strings, enums as their values."""
        return {
            "symbol": self.symbol,
            "interval": self.interval.value,
            "status": self.status.value,
            "months_to_fetch": self.months_to_fetch,
            "candles_to_fetch": self.candles_to_fetch,
            "candles_added": self.candles_added,
            "months_written": self.months_written,
            "gap_candles_found": self.gap_candles_found,
            "first_open_time": _iso(self.first_open_time),
            "last_open_time": _iso(self.last_open_time),
            "error": self.error,
        }


@dataclass(frozen=True)
class SyncSummary:
    """Result of one `sync_klines` run."""

    series: tuple[SeriesResult, ...]
    start: datetime
    end: datetime
    """Requested range end (the caller's `end`, or `now`); forming candles are never fetched."""
    now: datetime
    duration_s: float
    requests: int
    """Requests the source made during this run (from `source.stats`)."""
    rate_limit_hits: int
    max_used_weight: int
    aborted: str | None = None
    """Why the run was aborted (rate-limit abort or IP ban), or None if it was not."""

    @property
    def outcome(self) -> Literal["ok", "failed", "aborted"]:
        """`aborted` if the run was aborted, else `failed` if any series failed, else `ok`."""
        if self.aborted is not None:
            return "aborted"
        if any(result.status is SeriesStatus.FAILED for result in self.series):
            return "failed"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict: datetimes as ISO-8601 strings, enums as their values."""
        return {
            "outcome": self.outcome,
            "aborted": self.aborted,
            "start": _iso(self.start),
            "end": _iso(self.end),
            "now": _iso(self.now),
            "duration_s": round(self.duration_s, 3),
            "requests": self.requests,
            "rate_limit_hits": self.rate_limit_hits,
            "max_used_weight": self.max_used_weight,
            "candles_added": sum(result.candles_added for result in self.series),
            "series": [result.to_dict() for result in self.series],
        }


# --------------------------------------------------------------------------------------
# Progress events
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncStarted:
    """First event of a run."""

    series: tuple[tuple[str, Interval], ...]
    start: datetime
    end: datetime
    now: datetime
    concurrency: int


@dataclass(frozen=True)
class SeriesPlanned:
    """A series has been planned. Every series that is planned gets this before its months run.

    Progress for the series is complete when `candles_fetched + candles_missing` summed over its
    `MonthCompleted` events reaches `candles_to_fetch`.
    """

    symbol: str
    interval: Interval
    listing: datetime
    start: datetime
    """Start of the wanted range (the later of the requested start and the listing)."""
    end: datetime
    """End of the wanted range (exclusive): never past the still-forming candle."""
    candles_stored: int
    """Valid stored candles already inside the wanted range."""
    candles_to_fetch: int
    months: tuple[datetime, ...]
    """Start of each month that will be fetched, in order."""

    @property
    def months_to_fetch(self) -> int:
        return len(self.months)


@dataclass(frozen=True)
class MonthStarted:
    """A month unit got a concurrency slot and starts fetching."""

    symbol: str
    interval: Interval
    month: datetime


@dataclass(frozen=True)
class PageFetched:
    """One page of candles arrived for a month unit."""

    symbol: str
    interval: Interval
    month: datetime
    candles: int


@dataclass(frozen=True)
class MonthCompleted:
    """A month unit finished: fetched, stored (if anything came back) and gaps recorded."""

    symbol: str
    interval: Interval
    month: datetime
    candles_fetched: int
    candles_missing: int
    """Wanted candles Binance did not return (recorded gaps plus recent holes left for later)."""
    candles_added: int
    gap_candles: int
    """Of `candles_missing`, those recorded as gaps in `_meta.json`."""
    written: bool
    """Whether the month file was (re)written."""
    replaced_invalid: bool
    """Whether an invalid month file was overwritten."""
    months_done: int
    """Months of this series completed so far, including this one."""
    months_total: int


@dataclass(frozen=True)
class SeriesFinished:
    """A series is done (any status). Always the last event for that series."""

    result: SeriesResult


@dataclass(frozen=True)
class RateLimited:
    """Binance answered HTTP 429. Every request now waits `retry_after` seconds."""

    hits: int
    max_hits: int
    retry_after: float
    aborting: bool
    """True when this hit aborts the whole run."""


@dataclass(frozen=True)
class SyncFinished:
    """Last event of a run that was not cancelled."""

    summary: SyncSummary


type SyncEvent = (
    SyncStarted
    | SeriesPlanned
    | MonthStarted
    | PageFetched
    | MonthCompleted
    | SeriesFinished
    | RateLimited
    | SyncFinished
)
type SyncSink = Callable[[SyncEvent], None]
type _Job = Callable[[], Awaitable[None]]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


async def _to_thread_to_completion[T](func: Callable[[], T]) -> T:
    """Run blocking `func` in a thread. If cancelled, wait for it to finish, then re-raise.

    Used for writes: the atomic write either completes or cleans up its temp file, and the
    caller does not return before that happens.
    """
    future = asyncio.ensure_future(asyncio.to_thread(func))
    cancelled = False
    while not future.done():
        try:
            await asyncio.wait([future])  # never cancels `future` and never raises its error
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        error = future.exception()  # retrieved so asyncio does not warn about it
        if error is not None:
            logger.debug("write failed while the sync was being cancelled", exc_info=error)
        raise asyncio.CancelledError
    return future.result()


def _previous_month(month: datetime) -> datetime:
    return month_start(month - timedelta(days=1))


def gap_limit(interval: Interval, now: datetime) -> datetime:
    """Latest gap end that may be recorded at `now`.

    A hole that ends within `max(2 x interval, 5 minutes)` of `now` may just be Binance not having
    published the candles yet, so it is left for the next run instead of recorded as a gap.
    """
    now = as_utc(now)
    if interval.is_monthly:
        return _previous_month(_previous_month(month_start(now)))
    return interval.floor(now - max(2 * interval.length, MIN_GAP_AGE))


def _count(interval: Interval, ranges: Sequence[Range]) -> int:
    return sum(interval.count_open_times(start, end) for start, end in ranges)


def _describe(exc: BaseException) -> str:
    if isinstance(exc, BinanceError):
        return str(exc)
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _round_robin[T](groups: Sequence[Sequence[T]]) -> Iterator[T]:
    """First item of every group, then the second of every group, and so on."""
    longest = max((len(group) for group in groups), default=0)
    for index in range(longest):
        for group in groups:
            if index < len(group):
                yield group[index]


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------


@dataclass
class _Unit:
    month: datetime
    ranges: list[Range]


@dataclass
class _Series:
    symbol: str
    interval: Interval
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    meta: SeriesMeta = field(default_factory=SeriesMeta)
    invalid_months: frozenset[datetime] = frozenset()
    units: list[_Unit] = field(default_factory=list)
    planned: bool = False
    no_data: bool = False
    candles_to_fetch: int = 0
    units_done: int = 0
    candles_added: int = 0
    months_written: int = 0
    gap_candles: int = 0
    first: datetime | None = None
    last: datetime | None = None
    error: str | None = None
    tasks: set[asyncio.Task[None]] = field(default_factory=set)
    open_tasks: int = 0
    result: SeriesResult | None = None

    @property
    def label(self) -> str:
        return f"{self.symbol} {self.interval.value}"

    def extend_bounds(self, times: pl.Series) -> None:
        if times.len() == 0:
            return
        low, high = times.min(), times.max()
        if isinstance(low, datetime):
            self.first = low if self.first is None else min(self.first, low)
        if isinstance(high, datetime):
            self.last = high if self.last is None else max(self.last, high)


@dataclass(frozen=True)
class _LocalState:
    meta: SeriesMeta
    invalid_months: frozenset[datetime]
    stored: pl.Series


class _SyncRun:
    def __init__(
        self,
        source: KlineSource,
        store: KlineStore,
        series: Sequence[tuple[str, Interval]],
        *,
        start: datetime,
        end: datetime,
        now: datetime,
        concurrency: int,
        sink: SyncSink | None,
    ) -> None:
        self._source = source
        self._store = store
        self._start = start
        self._end = end
        self._now = now
        self._concurrency = concurrency
        self._sink = sink
        self._semaphore = asyncio.Semaphore(concurrency)
        self._series = [_Series(symbol, Interval(interval)) for symbol, interval in series]
        self._tasks: set[asyncio.Task[None]] = set()
        self._protected: set[asyncio.Task[Any]] = set()
        self._aborted: str | None = None
        self._cancelling = False
        self._running_units = False

    # -- events -----------------------------------------------------------------------------

    def _emit(self, event: SyncEvent) -> None:
        if self._sink is None:
            return
        try:
            self._sink(event)
        except Exception:
            logger.exception("sync progress sink raised on %s; ignoring it", type(event).__name__)

    def _rate_limit_hook(self, previous: RateLimitHook | None) -> RateLimitHook:
        def hook(hit: RateLimitHit) -> None:
            logger.debug("rate limited: %s", hit)
            self._emit(RateLimited(hit.hits, hit.max_hits, hit.retry_after, hit.aborting))
            if previous is not None:
                previous(hit)

        return hook

    # -- run --------------------------------------------------------------------------------

    async def run(self) -> SyncSummary:
        started = time.perf_counter()
        stats = self._source.stats
        requests_before, hits_before = stats.requests, stats.rate_limit_hits
        previous_hook = self._source.on_rate_limit
        self._source.on_rate_limit = self._rate_limit_hook(previous_hook)
        try:
            self._emit(
                SyncStarted(
                    tuple((s.symbol, s.interval) for s in self._series),
                    self._start,
                    self._end,
                    self._now,
                    self._concurrency,
                )
            )
            await self._run_tasks([(s, functools.partial(self._plan, s)) for s in self._series])
            for series in self._series:
                if series.error is not None or not series.units or self._aborted is not None:
                    self._finish(series)
            if self._aborted is None:
                runnable = [s for s in self._series if s.result is None]
                self._running_units = True
                jobs = _round_robin(
                    [
                        [(s, functools.partial(self._fetch_month, s, unit)) for unit in s.units]
                        for s in runnable
                    ]
                )
                await self._run_tasks(list(jobs))
            for series in self._series:  # anything left was cut short by an abort
                self._finish(series)
        finally:
            self._source.on_rate_limit = previous_hook

        stats = self._source.stats
        summary = SyncSummary(
            series=tuple(s.result for s in self._series if s.result is not None),
            start=self._start,
            end=self._end,
            now=self._now,
            duration_s=time.perf_counter() - started,
            requests=stats.requests - requests_before,
            rate_limit_hits=stats.rate_limit_hits - hits_before,
            max_used_weight=stats.max_used_weight,
            aborted=self._aborted,
        )
        self._emit(SyncFinished(summary))
        return summary

    async def _run_tasks(self, jobs: Sequence[tuple[_Series, _Job]]) -> None:
        """Run each job as a task and wait for all; if cancelled, cancel them all first."""
        tasks: list[asyncio.Task[None]] = []
        for series, job in jobs:
            task = asyncio.create_task(self._guard(series, job))
            tasks.append(task)
            self._tasks.add(task)
            series.tasks.add(task)
            series.open_tasks += 1
            task.add_done_callback(functools.partial(self._task_done, series))
        if not tasks:
            return
        try:
            await asyncio.wait(tasks)
        except asyncio.CancelledError:
            self._cancelling = True
            for task in tasks:
                task.cancel()
            await asyncio.wait(tasks)
            raise

    def _task_done(self, series: _Series, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        series.tasks.discard(task)
        series.open_tasks -= 1
        if self._running_units and series.open_tasks == 0 and not self._cancelling:
            self._finish(series)

    async def _guard(self, series: _Series, job: _Job) -> None:
        """Run a job, turning its exceptions into a series failure or a run abort."""
        try:
            await job()
        except asyncio.CancelledError:
            raise
        except (RateLimitAbortError, IpBannedError) as exc:
            self._abort(series, exc)
        except Exception as exc:
            self._fail(series, exc)

    def _cancel(self, tasks: set[asyncio.Task[None]]) -> None:
        current = asyncio.current_task()
        for task in tasks:
            if task is not current and task not in self._protected and not task.done():
                task.cancel()

    def _fail(self, series: _Series, exc: Exception) -> None:
        if series.error is None:
            series.error = _describe(exc)
            logger.debug("%s failed: %s", series.label, series.error, exc_info=exc)
        self._cancel(set(series.tasks))

    def _abort(self, series: _Series, exc: Exception) -> None:
        if self._aborted is None:
            self._aborted = _describe(exc)
            logger.debug("aborting sync after %s: %s", series.label, self._aborted)
        self._cancel(set(self._tasks))

    def _finish(self, series: _Series) -> None:
        """Freeze the series' result and emit `SeriesFinished` (once)."""
        if series.result is not None:
            return
        error = series.error
        if error is not None:
            status = SeriesStatus.FAILED
        elif series.no_data:
            status = SeriesStatus.NO_DATA
        elif series.planned and not series.units:
            status = SeriesStatus.UP_TO_DATE
        elif series.planned and series.units_done == len(series.units):
            status = SeriesStatus.OK
        else:
            status = SeriesStatus.ABORTED
            error = self._aborted
        series.result = SeriesResult(
            symbol=series.symbol,
            interval=series.interval,
            status=status,
            months_to_fetch=len(series.units),
            candles_to_fetch=series.candles_to_fetch,
            candles_added=series.candles_added,
            months_written=series.months_written,
            gap_candles_found=series.gap_candles,
            first_open_time=series.first,
            last_open_time=series.last,
            error=error,
        )
        self._emit(SeriesFinished(series.result))

    # -- planning ---------------------------------------------------------------------------

    def _read_local(self, symbol: str, interval: Interval) -> _LocalState:
        meta = self._store.read_meta(symbol, interval)
        invalid = frozenset(self._store.invalid_months(symbol, interval, now=self._now))
        valid = [m for m in self._store.list_months(symbol, interval) if m not in invalid]
        stored = self._store.stored_open_times(symbol, interval, months=valid)
        return _LocalState(meta, invalid, stored)

    async def _plan(self, series: _Series) -> None:
        async with self._semaphore:
            symbol, interval = series.symbol, series.interval
            local = await asyncio.to_thread(self._read_local, symbol, interval)
            series.meta, series.invalid_months = local.meta, local.invalid_months
            series.extend_bounds(local.stored)
            if local.invalid_months:
                logger.debug(
                    "%s: invalid month files will be refetched: %s",
                    series.label,
                    ", ".join(m.strftime("%Y-%m") for m in sorted(local.invalid_months)),
                )

            listing = series.meta.listing
            if listing is None:
                listing = await self._source.first_open_time(symbol, interval)
                if listing is None:
                    logger.debug("%s: Binance has no candles", series.label)
                    series.no_data = True
                    series.planned = True
                    return
                listing = as_utc(listing)
                meta = dataclasses.replace(series.meta, listing=listing)
                await _to_thread_to_completion(
                    lambda: self._store.write_meta(symbol, interval, meta)
                )
                series.meta = meta

            lo = max(self._start, listing)
            hi = min(self._end, interval.floor(self._now))
            gaps = list(series.meta.gaps)
            ranges = (
                await asyncio.to_thread(missing_ranges, interval, lo, hi, local.stored, gaps)
                if lo < hi
                else []
            )
            by_month = split_by_month(ranges)
            series.units = [_Unit(month, pieces) for month, pieces in by_month.items()]
            series.candles_to_fetch = _count(interval, ranges)
            series.planned = True
            stored = local.stored
            in_range = stored.filter((stored >= lo) & (stored < hi)).len() if lo < hi else 0
            logger.debug(
                "%s: %d candles stored, %d missing in %d months",
                series.label,
                in_range,
                series.candles_to_fetch,
                len(series.units),
            )
            self._emit(
                SeriesPlanned(
                    symbol=symbol,
                    interval=interval,
                    listing=listing,
                    start=lo,
                    end=max(lo, hi),
                    candles_stored=in_range,
                    candles_to_fetch=series.candles_to_fetch,
                    months=tuple(unit.month for unit in series.units),
                )
            )

    # -- month units ------------------------------------------------------------------------

    def _recordable(self, interval: Interval, holes: Sequence[Range]) -> list[Range]:
        limit = gap_limit(interval, self._now)
        return [(start, min(end, limit)) for start, end in holes if start < limit]

    async def _fetch_month(self, series: _Series, unit: _Unit) -> None:
        async with self._semaphore:
            symbol, interval, month = series.symbol, series.interval, unit.month
            self._emit(MonthStarted(symbol, interval, month))

            def on_page(candles: int) -> None:
                self._emit(PageFetched(symbol, interval, month, candles))

            frames: list[pl.DataFrame] = []
            missing = 0
            gaps: list[Range] = []
            for start, end in unit.ranges:
                frame = await self._source.fetch_klines(
                    symbol, interval, start, end, on_page=on_page
                )
                frames.append(frame)
                holes = missing_ranges(interval, start, end, frame["open_time"])
                missing += _count(interval, holes)
                gaps.extend(self._recordable(interval, holes))
            fetched = pl.concat(frames) if frames else empty_klines()

            # From here on the unit only writes. A series failure or a run abort lets it finish,
            # so its result counts what is on disk; only the caller's cancellation interrupts it
            # (and even then the write in progress completes, see _to_thread_to_completion).
            task = asyncio.current_task()
            if task is not None:
                self._protected.add(task)
            try:
                await self._store_month(series, unit, fetched, missing, gaps)
            finally:
                self._protected.discard(task)

    async def _store_month(
        self,
        series: _Series,
        unit: _Unit,
        fetched: pl.DataFrame,
        missing: int,
        gaps: list[Range],
    ) -> None:
        symbol, interval, month = series.symbol, series.interval, unit.month
        replace = month in series.invalid_months
        added = 0
        written = False
        if fetched.height:
            if replace:
                await _to_thread_to_completion(
                    lambda: self._store.write_month(symbol, interval, month, fetched)
                )
                added, written = fetched.height, True
            else:
                added = await _to_thread_to_completion(
                    lambda: self._store.merge_month(symbol, interval, month, fetched)
                )
                written = added > 0

        gap_candles = 0
        if gaps:
            async with series.lock:
                meta = series.meta.with_gaps(gaps)
                await _to_thread_to_completion(
                    lambda: self._store.write_meta(symbol, interval, meta)
                )
                series.meta = meta
            gap_candles = _count(interval, gaps)

        series.units_done += 1
        series.candles_added += added
        series.months_written += int(written)
        series.gap_candles += gap_candles
        if written:
            series.extend_bounds(fetched["open_time"])
        logger.debug(
            "%s %s: fetched %d, added %d, %d missing (%d recorded as gaps)%s",
            series.label,
            month.strftime("%Y-%m"),
            fetched.height,
            added,
            missing,
            gap_candles,
            ", replaced invalid file" if replace and written else "",
        )
        self._emit(
            MonthCompleted(
                symbol=symbol,
                interval=interval,
                month=month,
                candles_fetched=fetched.height,
                candles_missing=missing,
                candles_added=added,
                gap_candles=gap_candles,
                written=written,
                replaced_invalid=replace and written,
                months_done=series.units_done,
                months_total=len(series.units),
            )
        )


def _unique_series(series: Sequence[tuple[str, Interval]]) -> list[tuple[str, Interval]]:
    seen: dict[tuple[str, Interval], None] = {}
    for symbol, interval in series:
        seen.setdefault((symbol, Interval(interval)), None)
    return list(seen)


async def sync_klines(
    source: KlineSource,
    store: KlineStore,
    series: Sequence[tuple[str, Interval]],
    *,
    start: datetime,
    end: datetime | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    sink: SyncSink | None = None,
    now: datetime | None = None,
) -> SyncSummary:
    """Fetch the candles each (symbol, interval) series is missing and store them.

    For each series the wanted range is `[max(start, listing), min(end or now, floor(now)))`:
    nothing before the symbol's listing, and never the candle that is still forming. Stored
    candles and recorded gaps are skipped; invalid month files count as missing and are
    overwritten (only the part inside the wanted range is refetched). One unit of work per month
    with missing candles; units of all series share one limit of `concurrency` at a time.

    Args:
        source: Where candles come from (one `BinanceClient` per run).
        store: Where candles go.
        series: `(symbol, interval)` pairs; duplicates are ignored.
        start: Earliest open time wanted (timezone-aware).
        end: Open times before this are wanted (default: `now`).
        concurrency: Months (and series plans) in flight at once.
        sink: Receives progress events on the event-loop thread. Its exceptions are logged and
            ignored.
        now: The current time (default: the system clock), fixed for the whole run.

    Returns:
        A summary with one result per series, in the order given. Check `aborted`: a rate-limit
        abort or IP ban does not raise, it ends the run with `aborted` set.

    Raises:
        asyncio.CancelledError: The caller's task was cancelled; months already written stay.
        ValueError: Naive datetimes or `concurrency < 1`.
    """
    if concurrency < 1:
        msg = "concurrency must be at least 1"
        raise ValueError(msg)
    now = as_utc(now) if now is not None else datetime.now(UTC)
    start = as_utc(start)
    end = as_utc(end) if end is not None else now
    run = _SyncRun(
        source,
        store,
        _unique_series(series),
        start=start,
        end=min(end, now),
        now=now,
        concurrency=concurrency,
        sink=sink,
    )
    return await run.run()
