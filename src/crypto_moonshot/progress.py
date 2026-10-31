"""Progress renderers for `moonshot data sync`: each one is a `SyncSink` that draws sync events.

- `BarsProgress` (rich): one live row per series plus an overall row, for a terminal.
- `LinesProgress`: plain text with no escape codes, one line per event plus a periodic snapshot,
  for agents and log files. The snapshot keeps coming while sync waits (for example during a
  rate-limit pause), so a reader can tell the run is alive.
- `none`: no renderer at all (`make_renderer` returns None).

Everything here goes to stderr by default; stdout is reserved for the summary.

Both renderers share `ProgressTracker`, which folds the event stream into per-series state:
candles done / total, months, rate and ETA. Candles done advance per `PageFetched` and are
reconciled to the exact figures on `MonthCompleted`, so a finished series always ends at 100%.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, TextIO

from rich.console import Console, RenderableType
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TaskID
from rich.table import Column
from rich.text import Text

from crypto_moonshot.data.sync import (
    MonthCompleted,
    MonthStarted,
    PageFetched,
    RateLimited,
    SeriesFinished,
    SeriesPlanned,
    SeriesResult,
    SeriesStatus,
    SyncEvent,
    SyncFinished,
    SyncStarted,
    SyncSummary,
)
from crypto_moonshot.models import Interval

type Clock = Callable[[], float]
type WallClock = Callable[[], datetime]

DEFAULT_SNAPSHOT_INTERVAL_S = 5.0
_REASON_WIDTH = 44
_BARS_REASON_WIDTH = 120
"""A long failure reason wraps inside the status cell instead of being cut."""


class ProgressMode(StrEnum):
    """How `moonshot data sync` shows progress."""

    AUTO = "auto"
    """`bars` when stderr is a terminal, else `lines`."""
    BARS = "bars"
    LINES = "lines"
    NONE = "none"


class ProgressRenderer(Protocol):
    """A `SyncSink` that also needs closing (to stop a live display or ticker)."""

    def __call__(self, event: SyncEvent, /) -> None: ...

    def close(self) -> None:
        """Stop drawing. Safe to call more than once, and after the run finished."""
        ...


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def format_count(value: int) -> str:
    return f"{value:,}"


def format_duration(seconds: float | None) -> str:
    """`H:MM:SS`, or `?` when unknown."""
    if seconds is None:
        return "?"
    total = max(0, round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def format_rate(rate: float | None) -> str:
    return "-" if rate is None else f"{rate:,.0f} c/s"


def short_reason(text: str | None, width: int = _REASON_WIDTH) -> str:
    """One-line, length-limited version of an error message."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def _percent(done: int, total: int) -> int:
    return 100 if total <= 0 else min(100, int(100 * done / total))


# --------------------------------------------------------------------------------------
# State tracking
# --------------------------------------------------------------------------------------


class Phase(StrEnum):
    PLANNING = "planning"
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    FINISHED = "finished"


@dataclass
class SeriesProgress:
    """Progress of one (symbol, interval) series, built from its events."""

    symbol: str
    interval: Interval
    phase: Phase = Phase.PLANNING
    planned: bool = False
    """Whether `SeriesPlanned` arrived (a series that fails while planning never gets it)."""
    stored: int = 0
    total: int = 0
    """Candles to fetch, from `SeriesPlanned`."""
    settled: int = 0
    """Candles accounted for by completed months (fetched plus known to be missing)."""
    paged: int = 0
    """Candles received so far for months that have not completed."""
    months_done: int = 0
    months_total: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    result: SeriesResult | None = None
    _paged_by_month: dict[datetime, int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.symbol} {self.interval.value}"

    @property
    def done(self) -> int:
        return min(self.total, self.settled + self.paged)

    def rate(self, now: float) -> float | None:
        """Average candles per second since the first month started (frozen when finished)."""
        if self.started_at is None or self.done == 0:
            return None
        elapsed = (self.finished_at if self.finished_at is not None else now) - self.started_at
        return self.done / elapsed if elapsed > 0 else None

    def eta(self, now: float) -> float | None:
        if self.phase is Phase.FINISHED:
            return None
        rate = self.rate(now)
        if rate is None or rate <= 0:
            return None
        return (self.total - self.done) / rate

    def add_page(self, month: datetime, candles: int) -> None:
        self.paged += candles
        self._paged_by_month[month] = self._paged_by_month.get(month, 0) + candles

    def complete_month(self, event: MonthCompleted) -> None:
        self.paged -= self._paged_by_month.pop(event.month, 0)
        self.settled += event.candles_fetched + event.candles_missing
        self.months_done = event.months_done

    def finish(self, result: SeriesResult, now: float) -> None:
        self.result = result
        self.phase = Phase.FINISHED
        self.finished_at = now
        if result.status is SeriesStatus.OK:
            self.settled, self.paged, self._paged_by_month = self.total, 0, {}
            self.months_done = self.months_total


@dataclass(frozen=True)
class Totals:
    """All series added together."""

    series_total: int
    series_done: int
    planning: int
    done: int
    total: int
    months_done: int
    months_total: int
    rate: float | None
    eta: float | None


class ProgressTracker:
    """Folds sync events into `SeriesProgress` rows and overall figures."""

    def __init__(self, clock: Clock = time.monotonic) -> None:
        self.clock = clock
        self.series: dict[tuple[str, Interval], SeriesProgress] = {}
        self.started: SyncStarted | None = None
        self.summary: SyncSummary | None = None
        self.rate_limit_hits = 0
        self._paused_until: float | None = None
        self._first_fetch_at: float | None = None
        self._finished_at: float | None = None

    def row(self, symbol: str, interval: Interval) -> SeriesProgress:
        key = (symbol, Interval(interval))
        if key not in self.series:
            self.series[key] = SeriesProgress(symbol, key[1])
        return self.series[key]

    def handle(self, event: SyncEvent) -> SeriesProgress | None:
        """Update state for `event`; return the series it concerns, if any."""
        now = self.clock()
        match event:
            case SyncStarted():
                self.started = event
                for symbol, interval in event.series:
                    self.row(symbol, interval)
            case SeriesPlanned():
                row = self.row(event.symbol, event.interval)
                row.planned = True
                row.stored = event.candles_stored
                row.total = event.candles_to_fetch
                row.months_total = event.months_to_fetch
                row.phase = Phase.QUEUED if event.months_to_fetch else row.phase
                return row
            case MonthStarted():
                row = self.row(event.symbol, event.interval)
                row.phase = Phase.DOWNLOADING
                if row.started_at is None:
                    row.started_at = now
                if self._first_fetch_at is None:
                    self._first_fetch_at = now
                return row
            case PageFetched():
                row = self.row(event.symbol, event.interval)
                row.add_page(event.month, event.candles)
                return row
            case MonthCompleted():
                row = self.row(event.symbol, event.interval)
                row.complete_month(event)
                return row
            case SeriesFinished():
                row = self.row(event.result.symbol, event.result.interval)
                row.finish(event.result, now)
                return row
            case RateLimited():
                self.rate_limit_hits = event.hits
                self._paused_until = None if event.aborting else now + event.retry_after
            case SyncFinished():
                self.summary = event.summary
                self._finished_at = now
                self._paused_until = None
        return None

    def pause_left(self) -> float | None:
        """Seconds left of a rate-limit pause, or None when not paused."""
        if self._paused_until is None:
            return None
        left = self._paused_until - self.clock()
        return left if left > 0 else None

    def totals(self) -> Totals:
        now = self.clock()
        rows = list(self.series.values())
        done = sum(row.done for row in rows)
        total = sum(row.total for row in rows)
        rate: float | None = None
        if self._first_fetch_at is not None and done:
            end = self._finished_at if self._finished_at is not None else now
            elapsed = end - self._first_fetch_at
            rate = done / elapsed if elapsed > 0 else None
        planning = sum(1 for row in rows if row.phase is Phase.PLANNING)
        eta = (total - done) / rate if rate and self._finished_at is None else None
        return Totals(
            series_total=len(rows),
            series_done=sum(1 for row in rows if row.phase is Phase.FINISHED),
            planning=planning,
            done=done,
            total=total,
            months_done=sum(row.months_done for row in rows),
            months_total=sum(row.months_total for row in rows),
            rate=rate,
            eta=eta,
        )


def status_text(row: SeriesProgress, reason_width: int = _REASON_WIDTH) -> tuple[str, str]:
    """Short status of a series and a rich style for it."""
    result = row.result
    if result is None:
        return {
            Phase.PLANNING: ("planning...", "dim"),
            Phase.QUEUED: ("queued", "dim"),
            Phase.DOWNLOADING: ("downloading", "cyan"),
            Phase.FINISHED: ("finished", ""),
        }[row.phase]
    match result.status:
        case SeriesStatus.OK:
            return "done \N{CHECK MARK}", "bold green"
        case SeriesStatus.UP_TO_DATE:
            return "up to date", "green"
        case SeriesStatus.NO_DATA:
            return "no data on Binance", "yellow"
        case SeriesStatus.FAILED:
            return f"failed \N{BALLOT X} {short_reason(result.error, reason_width)}", "bold red"
        case SeriesStatus.ABORTED:
            return "aborted", "yellow"


# --------------------------------------------------------------------------------------
# Lines renderer
# --------------------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


class LinesProgress:
    """Plain-text progress: one timestamped line per event, plus periodic snapshots.

    Lines look like `12:00:05 BTCUSDT 1h  23,400/59,000 candles (40%)  28/70 months  1,850 c/s
    ETA 0:00:19`. No escape codes, ASCII only. Snapshots come from an asyncio task started on
    `SyncStarted` (when an event loop is running); `snapshot()` can also be called directly.

    Args:
        stream: Where to write (default: `sys.stderr`, looked up at each write).
        snapshot_interval: Seconds between snapshots.
        clock: Monotonic clock for rates and ETAs.
        wall_clock: UTC time for the line prefix.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        snapshot_interval: float = DEFAULT_SNAPSHOT_INTERVAL_S,
        clock: Clock = time.monotonic,
        wall_clock: WallClock = _utc_now,
    ) -> None:
        self._stream = stream
        self._snapshot_interval = snapshot_interval
        self._wall_clock = wall_clock
        self.tracker = ProgressTracker(clock)
        self._ticker: asyncio.Task[None] | None = None

    # -- output -----------------------------------------------------------------------------

    def _write(self, text: str) -> None:
        stream = self._stream if self._stream is not None else sys.stderr
        stamp = self._wall_clock().astimezone(UTC).strftime("%H:%M:%S")
        stream.write(f"{stamp} {text}\n")
        stream.flush()

    def _width(self) -> int:
        return max((len(row.label) for row in self.tracker.series.values()), default=0)

    # -- events -----------------------------------------------------------------------------

    def __call__(self, event: SyncEvent, /) -> None:
        self.tracker.handle(event)
        match event:
            case SyncStarted():
                self._write(self._started_line(event))
                self._start_ticker()
            case SeriesPlanned():
                self._write(self._planned_line(event))
            case SeriesFinished():
                self._write(self._finished_line(event.result))
            case RateLimited():
                self._write(self._rate_limited_line(event))
            case SyncFinished():
                self._stop_ticker()
                self._write(self._summary_line(event.summary))
            case _:
                pass

    def _started_line(self, event: SyncStarted) -> str:
        count = len(event.series)
        return (
            f"sync started: {count} series, {event.start:%Y-%m-%d %H:%M} to "
            f"{event.end:%Y-%m-%d %H:%M} UTC, concurrency {event.concurrency}"
        )

    def _planned_line(self, event: SeriesPlanned) -> str:
        label = f"{event.symbol} {event.interval.value}".ljust(self._width())
        stored = format_count(event.candles_stored)
        if not event.candles_to_fetch:
            return f"{label}  planned: nothing to fetch ({stored} candles stored)"
        return (
            f"{label}  planned: {stored} candles stored, "
            f"{format_count(event.candles_to_fetch)} to fetch in {event.months_to_fetch} months"
        )

    def _finished_line(self, result: SeriesResult) -> str:
        label = f"{result.symbol} {result.interval.value}".ljust(self._width())
        match result.status:
            case SeriesStatus.OK:
                text = f"done: added {format_count(result.candles_added)} candles"
                text += f", {result.months_written} months written"
                if result.gap_candles_found:
                    text += f", {format_count(result.gap_candles_found)} gap candles recorded"
            case SeriesStatus.UP_TO_DATE:
                text = "up to date: nothing to fetch"
            case SeriesStatus.NO_DATA:
                text = "no data: Binance has no candles for this symbol and interval"
            case SeriesStatus.FAILED:
                text = f"FAILED: {short_reason(result.error, 200)}"
                if result.candles_added:
                    text += f" (added {format_count(result.candles_added)} candles before failing)"
            case SeriesStatus.ABORTED:
                text = f"ABORTED: {short_reason(result.error, 200)}"
                text += f" (added {format_count(result.candles_added)} candles)"
        return f"{label}  {text}"

    @staticmethod
    def _rate_limited_line(event: RateLimited) -> str:
        head = f"WARNING rate limited by Binance (hit {event.hits}/{event.max_hits})"
        if event.aborting:
            return f"{head}, aborting the run"
        return f"{head}, waiting {event.retry_after:g}s"

    @staticmethod
    def _summary_line(summary: SyncSummary) -> str:
        added = sum(result.candles_added for result in summary.series)
        return (
            f"sync finished: {summary.outcome}, {format_count(added)} candles added, "
            f"{summary.requests} requests, {summary.duration_s:.1f}s"
        )

    # -- snapshots --------------------------------------------------------------------------

    def snapshot(self) -> None:
        """Write one line per unfinished series that has work, plus an overall line."""
        tracker = self.tracker
        now = tracker.clock()
        width = self._width()
        for row in tracker.series.values():
            if row.phase in (Phase.QUEUED, Phase.DOWNLOADING):
                self._write(self._series_snapshot(row, now, width))
        self._write(self._overall_snapshot(width))

    @staticmethod
    def _series_snapshot(row: SeriesProgress, now: float, width: int) -> str:
        return (
            f"{row.label.ljust(width)}  "
            f"{format_count(row.done)}/{format_count(row.total)} candles "
            f"({_percent(row.done, row.total)}%)  "
            f"{row.months_done}/{row.months_total} months  "
            f"{format_rate(row.rate(now))}  ETA {format_duration(row.eta(now))}"
        )

    def _overall_snapshot(self, width: int) -> str:
        totals = self.tracker.totals()
        text = (
            f"{'TOTAL'.ljust(width)}  "
            f"{totals.series_done}/{totals.series_total} series done  "
            f"{format_count(totals.done)}/{format_count(totals.total)} candles "
            f"({_percent(totals.done, totals.total)}%)  "
            f"{totals.months_done}/{totals.months_total} months  "
            f"{format_rate(totals.rate)}  ETA {format_duration(totals.eta)}"
        )
        if totals.planning:
            text += f"  (planning {totals.planning} series)"
        pause = self.tracker.pause_left()
        if pause is not None:
            text += f"  (paused by rate limit, about {pause:.0f}s left)"
        return text

    def _start_ticker(self) -> None:
        if self._ticker is not None or self._snapshot_interval <= 0:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no event loop: the caller can still call snapshot()
        self._ticker = loop.create_task(self._tick())

    def _stop_ticker(self) -> None:
        if self._ticker is not None:
            self._ticker.cancel()
            self._ticker = None

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(self._snapshot_interval)
            self.snapshot()

    def close(self) -> None:
        self._stop_ticker()


# --------------------------------------------------------------------------------------
# Bars renderer
# --------------------------------------------------------------------------------------


@dataclass
class _OverallView:
    """Stands in for a `SeriesProgress` on the bold overall row."""

    tracker: ProgressTracker

    @property
    def label(self) -> str:
        return "TOTAL"


type _View = SeriesProgress | _OverallView


class _Cell(ProgressColumn):
    """A column whose text is computed from the row's live state at every refresh."""

    def __init__(
        self,
        render: Callable[[_View, float], Text],
        clock: Clock,
        *,
        justify: str = "left",
        min_width: int | None = None,
        wrap: bool = False,
    ) -> None:
        super().__init__(
            table_column=Column(
                no_wrap=not wrap,
                overflow="fold" if wrap else "ellipsis",
                justify=justify,  # type: ignore[arg-type]
                min_width=min_width,
            )
        )
        self._render = render
        self._clock = clock

    def render(self, task: Task) -> RenderableType:
        view: _View = task.fields["view"]
        text = self._render(view, self._clock())
        if isinstance(view, _OverallView):
            text.stylize("bold")
        return text


class BarsProgress:
    """Live rich progress rows: one per series plus a bold overall row.

    Each row shows the label, a bar, candles done/total, months, candles/s, ETA and a status.
    Rate-limit warnings print above the bars. Finished rows stay on screen.

    Args:
        console: Where to draw (default: a console on stderr).
        clock: Monotonic clock for rates and ETAs.
        refresh_per_second: Redraw rate; rates and ETAs refresh even when no event arrives.
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        clock: Clock = time.monotonic,
        refresh_per_second: float = 10,
    ) -> None:
        self.console = console if console is not None else Console(stderr=True)
        self.tracker = ProgressTracker(clock)
        self._tasks: dict[tuple[str, Interval], TaskID] = {}
        self._overall: TaskID | None = None
        self._running = False
        self._progress = Progress(
            _Cell(self._label_text, clock),
            BarColumn(bar_width=20, table_column=Column(no_wrap=True)),
            _Cell(self._candles_text, clock, justify="right", min_width=17),
            _Cell(self._months_text, clock, justify="right", min_width=10),
            _Cell(self._rate_text, clock, justify="right", min_width=11),
            _Cell(self._eta_text, clock, justify="right", min_width=11),
            _Cell(self._status_text, clock, wrap=True),
            console=self.console,
            transient=False,
            refresh_per_second=refresh_per_second,
        )

    # -- cell text --------------------------------------------------------------------------

    def _label_text(self, view: _View, now: float) -> Text:
        return Text(view.label)

    def _candles_text(self, view: _View, now: float) -> Text:
        if isinstance(view, _OverallView):
            totals = view.tracker.totals()
            return Text(f"{format_count(totals.done)}/{format_count(totals.total)}")
        if view.phase is Phase.FINISHED and not view.planned:
            return Text("-")
        if view.planned and view.total == 0:
            return Text(f"{format_count(view.stored)} stored")
        return Text(f"{format_count(view.done)}/{format_count(view.total)}")

    def _months_text(self, view: _View, now: float) -> Text:
        if isinstance(view, _OverallView):
            totals = view.tracker.totals()
            return Text(f"{totals.months_done}/{totals.months_total} mo")
        return Text(f"{view.months_done}/{view.months_total} mo")

    def _rate_text(self, view: _View, now: float) -> Text:
        if isinstance(view, _OverallView):
            return Text(format_rate(view.tracker.totals().rate))
        return Text(format_rate(view.rate(now)))

    def _eta_text(self, view: _View, now: float) -> Text:
        if isinstance(view, _OverallView):
            totals = view.tracker.totals()
            return (
                Text(f"ETA {format_duration(totals.eta)}") if totals.eta is not None else Text("")
            )
        if view.phase is Phase.FINISHED:
            return Text("")
        return Text(f"ETA {format_duration(view.eta(now))}")

    def _status_text(self, view: _View, now: float) -> Text:
        if isinstance(view, _OverallView):
            totals = view.tracker.totals()
            text = f"{totals.series_done}/{totals.series_total} series"
            pause = view.tracker.pause_left()
            if pause is not None:
                text += f", rate-limit pause {pause:.0f}s"
            return Text(text)
        text, style = status_text(view, _BARS_REASON_WIDTH)
        return Text(text, style=style)

    # -- events -----------------------------------------------------------------------------

    def __call__(self, event: SyncEvent, /) -> None:
        row = self.tracker.handle(event)
        if isinstance(event, SyncStarted):
            self._start(event)
        elif isinstance(event, RateLimited):
            self.console.print(Text(self._warning(event), style="bold yellow"))
        elif isinstance(event, SyncFinished):
            self._update_all()
            self.close()
            return
        if row is not None:
            self._update_row(row)
        self._update_overall()

    @staticmethod
    def _warning(event: RateLimited) -> str:
        head = f"\N{WARNING SIGN} rate limited by Binance (hit {event.hits}/{event.max_hits})"
        if event.aborting:
            return f"{head}, aborting the run"
        return f"{head}, waiting {event.retry_after:g}s"

    def _start(self, event: SyncStarted) -> None:
        if self._running:
            return
        self._progress.start()
        self._running = True
        for symbol, interval in event.series:
            row = self.tracker.row(symbol, interval)
            self._tasks[(symbol, interval)] = self._progress.add_task(
                row.label, total=1, completed=0, view=row
            )
        self._overall = self._progress.add_task(
            "TOTAL", total=1, completed=0, view=_OverallView(self.tracker)
        )

    def _update_row(self, row: SeriesProgress) -> None:
        task_id = self._tasks.get((row.symbol, row.interval))
        if task_id is None:
            return
        if row.total > 0:
            self._progress.update(task_id, total=row.total, completed=row.done)
        elif row.planned:
            self._progress.update(task_id, total=1, completed=1)  # nothing to fetch: full bar

    def _update_overall(self) -> None:
        if self._overall is None:
            return
        totals = self.tracker.totals()
        self._progress.update(
            self._overall,
            total=max(totals.total, 1),
            completed=totals.done or int(not totals.total),
        )

    def _update_all(self) -> None:
        for row in self.tracker.series.values():
            self._update_row(row)
        self._update_overall()

    def close(self) -> None:
        if self._running:
            self._running = False
            self._progress.stop()


def make_renderer(
    mode: ProgressMode, *, console: Console | None = None, stderr_is_tty: bool | None = None
) -> ProgressRenderer | None:
    """Renderer for `mode` (None for `none`). `auto` follows `stderr_is_tty` (default: detect)."""
    if mode is ProgressMode.AUTO:
        tty = sys.stderr.isatty() if stderr_is_tty is None else stderr_is_tty
        mode = ProgressMode.BARS if tty else ProgressMode.LINES
    if mode is ProgressMode.BARS:
        return BarsProgress(console)
    if mode is ProgressMode.LINES:
        return LinesProgress()
    return None
