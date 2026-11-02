import asyncio
import io
import re
from datetime import UTC, datetime, timedelta

import pytest
from rich.console import Console

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
from crypto_moonshot.progress import (
    BarsProgress,
    LinesProgress,
    Phase,
    ProgressMode,
    format_duration,
    make_renderer,
    short_reason,
)

H1 = Interval.H1
START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 3, 1, tzinfo=UTC)
JAN = datetime(2024, 1, 1, tzinfo=UTC)
FEB = datetime(2024, 2, 1, tzinfo=UTC)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class FakeClock:
    """Monotonic seconds and a UTC wall clock that move together."""

    def __init__(self) -> None:
        self.seconds = 100.0

    def __call__(self) -> float:
        return self.seconds

    def wall(self) -> datetime:
        return datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC) + timedelta(seconds=self.seconds - 100.0)

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


def planned(symbol: str, to_fetch: int, months: tuple[datetime, ...]) -> SeriesPlanned:
    return SeriesPlanned(
        symbol=symbol,
        interval=H1,
        listing=START,
        start=START,
        end=END,
        candles_stored=100,
        candles_to_fetch=to_fetch,
        months=months,
    )


def month_done(
    symbol: str, month: datetime, fetched: int, done: int, total: int, missing: int = 0
) -> MonthCompleted:
    return MonthCompleted(
        symbol=symbol,
        interval=H1,
        month=month,
        candles_fetched=fetched,
        candles_missing=missing,
        candles_added=fetched,
        gap_candles=missing,
        written=True,
        replaced_invalid=False,
        months_done=done,
        months_total=total,
    )


BTC_RESULT = SeriesResult("BTCUSDT", H1, SeriesStatus.OK, 2, 1000, 995, 2, 5)
ETH_RESULT = SeriesResult("ETHUSDT", H1, SeriesStatus.UP_TO_DATE)
BAD_RESULT = SeriesResult(
    "BADUSDT", H1, SeriesStatus.FAILED, error="HTTP 400, code -1121: Invalid symbol."
)
SUMMARY = SyncSummary(
    series=(BTC_RESULT, ETH_RESULT, BAD_RESULT),
    start=START,
    end=END,
    now=END,
    duration_s=12.3,
    requests=9,
    rate_limit_hits=1,
    max_used_weight=40,
)

SERIES = (("BTCUSDT", H1), ("ETHUSDT", H1), ("BADUSDT", H1))


def started() -> SyncStarted:
    return SyncStarted(SERIES, START, END, END, 4)


def script_until_midway() -> list[SyncEvent]:
    """Three series: BAD fails while planning, ETH is up to date, BTC is mid-download."""
    return [
        started(),
        SeriesFinished(BAD_RESULT),
        planned("ETHUSDT", 0, ()),
        SeriesFinished(ETH_RESULT),
        planned("BTCUSDT", 1000, (JAN, FEB)),
        MonthStarted("BTCUSDT", H1, JAN),
        MonthStarted("BTCUSDT", H1, FEB),
        PageFetched("BTCUSDT", H1, JAN, 300),
    ]


def script_to_end() -> list[SyncEvent]:
    return [
        *script_until_midway(),
        RateLimited(1, 3, 30.0, False),
        PageFetched("BTCUSDT", H1, JAN, 300),
        month_done("BTCUSDT", JAN, 600, 1, 2),
        PageFetched("BTCUSDT", H1, FEB, 395),
        month_done("BTCUSDT", FEB, 395, 2, 2, missing=5),
        SeriesFinished(BTC_RESULT),
        SyncFinished(SUMMARY),
    ]


# -- formatting ------------------------------------------------------------------------------


def test_format_duration() -> None:
    assert format_duration(19) == "0:00:19"
    assert format_duration(3725.4) == "1:02:05"
    assert format_duration(None) == "?"


def test_short_reason_flattens_and_truncates() -> None:
    assert short_reason("a\n  b") == "a b"
    assert len(short_reason("x" * 100)) == 44
    assert short_reason("x" * 100).endswith("...")
    assert short_reason(None) == ""


def test_make_renderer_picks_by_mode_and_tty() -> None:
    assert make_renderer(ProgressMode.NONE) is None
    assert isinstance(make_renderer(ProgressMode.AUTO, stderr_is_tty=True), BarsProgress)
    assert isinstance(make_renderer(ProgressMode.AUTO, stderr_is_tty=False), LinesProgress)
    assert isinstance(make_renderer(ProgressMode.LINES, stderr_is_tty=True), LinesProgress)


# -- lines -----------------------------------------------------------------------------------


def lines_renderer(clock: FakeClock, out: io.StringIO, interval: float = 5.0) -> LinesProgress:
    return LinesProgress(out, snapshot_interval=interval, clock=clock, wall_clock=clock.wall)


def test_lines_event_lines() -> None:
    clock = FakeClock()
    out = io.StringIO()
    renderer = lines_renderer(clock, out)
    for event in script_to_end():
        renderer(event)

    lines = out.getvalue().splitlines()
    assert all(re.match(r"^\d\d:\d\d:\d\d \S", line) for line in lines)
    assert "\x1b" not in out.getvalue()
    body = [line[9:] for line in lines]
    assert body == [
        "sync started: 3 series, 2024-01-01 00:00 to 2024-03-01 00:00 UTC, concurrency 4",
        "BADUSDT 1h  FAILED: HTTP 400, code -1121: Invalid symbol.",
        "ETHUSDT 1h  planned: nothing to fetch (100 candles stored)",
        "ETHUSDT 1h  up to date: nothing to fetch",
        "BTCUSDT 1h  planned: 100 candles stored, 1,000 to fetch in 2 months",
        "WARNING rate limited by Binance (hit 1/3), waiting 30s",
        "BTCUSDT 1h  done: added 995 candles, 2 months written, 5 gap candles recorded",
        "sync finished: failed, 995 candles added, 9 requests, 12.3s",
    ]


def test_lines_rate_limit_abort_and_aborted_series() -> None:
    out = io.StringIO()
    renderer = lines_renderer(FakeClock(), out)
    renderer(started())
    renderer(RateLimited(3, 3, 0.0, True))
    renderer(SeriesFinished(SeriesResult("BTCUSDT", H1, SeriesStatus.ABORTED, error="too many")))
    text = out.getvalue()
    assert "WARNING rate limited by Binance (hit 3/3), aborting the run" in text
    assert "BTCUSDT 1h  ABORTED: too many (added 0 candles)" in text


def test_lines_snapshot_format() -> None:
    clock = FakeClock()
    out = io.StringIO()
    renderer = lines_renderer(clock, out)
    for event in script_until_midway()[:-1]:
        renderer(event)
    clock.advance(10)
    renderer(PageFetched("BTCUSDT", H1, JAN, 300))
    out.seek(0)
    out.truncate()

    renderer.snapshot()

    btc, total = out.getvalue().splitlines()
    assert btc.endswith("BTCUSDT 1h  300/1,000 candles (30%)  0/2 months  30 c/s  ETA 0:00:23")
    assert total.endswith(
        "TOTAL       2/3 series done  300/1,000 candles (30%)  0/2 months  30 c/s  ETA 0:00:23"
    )
    assert re.match(r"^\d\d:\d\d:\d\d ", btc)


def test_lines_snapshot_shows_the_rate_limit_pause() -> None:
    clock = FakeClock()
    out = io.StringIO()
    renderer = lines_renderer(clock, out)
    for event in script_until_midway():
        renderer(event)
    renderer(RateLimited(1, 3, 30.0, False))
    clock.advance(10)
    out.seek(0)
    out.truncate()

    renderer.snapshot()

    assert "(paused by rate limit, about 20s left)" in out.getvalue().splitlines()[-1]


def test_lines_snapshot_before_planning_has_only_the_overall_line() -> None:
    out = io.StringIO()
    renderer = lines_renderer(FakeClock(), out)
    renderer(started())
    out.seek(0)
    out.truncate()

    renderer.snapshot()

    (line,) = out.getvalue().splitlines()
    assert "0/3 series done  0/0 candles (100%)" in line
    assert line.endswith("(planning 3 series)")


@pytest.mark.anyio
async def test_lines_ticker_keeps_printing_snapshots_while_idle() -> None:
    clock = FakeClock()
    out = io.StringIO()
    renderer = lines_renderer(clock, out, interval=0.01)
    for event in script_until_midway():
        renderer(event)
    out.seek(0)
    out.truncate()

    await asyncio.sleep(0.15)  # nothing happens: a long wait
    renderer.close()
    count = out.getvalue().count("TOTAL")
    assert count >= 3
    await asyncio.sleep(0.05)
    assert out.getvalue().count("TOTAL") == count  # closed: no more snapshots


@pytest.mark.anyio
async def test_lines_ticker_stops_when_the_run_finishes() -> None:
    out = io.StringIO()
    renderer = lines_renderer(FakeClock(), out, interval=0.01)
    for event in script_to_end():
        renderer(event)
    await asyncio.sleep(0.05)
    assert "TOTAL" not in out.getvalue()


# -- tracker via renderers ---------------------------------------------------------------------


def test_progress_advances_per_page_and_reconciles_on_month_completed() -> None:
    renderer = lines_renderer(FakeClock(), io.StringIO())
    for event in script_until_midway():
        renderer(event)
    btc = renderer.tracker.row("BTCUSDT", H1)
    assert (btc.phase, btc.done, btc.total) == (Phase.DOWNLOADING, 300, 1000)

    renderer(PageFetched("BTCUSDT", H1, JAN, 300))
    assert btc.done == 600
    renderer(month_done("BTCUSDT", JAN, 600, 1, 2))
    assert (btc.done, btc.months_done) == (600, 1)
    renderer(PageFetched("BTCUSDT", H1, FEB, 395))
    renderer(month_done("BTCUSDT", FEB, 395, 2, 2, missing=5))
    assert btc.done == 1000  # fetched + missing: the bar reaches 100%
    renderer(SeriesFinished(BTC_RESULT))
    assert btc.phase is Phase.FINISHED
    assert btc.eta(renderer.tracker.clock()) is None


# -- bars ------------------------------------------------------------------------------------


def bars(clock: FakeClock, width: int = 120) -> tuple[BarsProgress, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, width=width)
    return BarsProgress(console, clock=clock, refresh_per_second=50), buffer


def plain(buffer: io.StringIO) -> str:
    return ANSI.sub("", buffer.getvalue())


@pytest.mark.parametrize("width", [80, 120, 160])
def test_bars_show_every_series_the_warning_and_the_overall_row(width: int) -> None:
    clock = FakeClock()
    renderer, buffer = bars(clock, width)
    events = script_to_end()
    for index, event in enumerate(events):
        renderer(event)
        clock.advance(2)
        if index == len(events) - 2:  # before SyncFinished
            assert renderer.tracker.row("BTCUSDT", H1).phase is Phase.FINISHED

    text = plain(buffer)
    for label in ("BTCUSDT 1h", "ETHUSDT 1h", "BADUSDT 1h", "TOTAL"):
        assert label in text
    assert "\N{WARNING SIGN} rate limited by Binance (hit 1/3), waiting 30s" in text
    assert "done \N{CHECK MARK}" in text
    assert "up to date" in text
    assert "failed \N{BALLOT X}" in text
    assert "symbol." in text  # the reason wraps inside the status cell at narrow widths
    assert "995/1,000" not in text  # BTC's settled figures are fetched + missing = 1,000
    assert "1,000/1,000" in text
    assert "2/2 mo" in text
    assert "3/3 series" in text


def test_bars_mid_run_rows_show_progress_and_downloading() -> None:
    clock = FakeClock()
    renderer, buffer = bars(clock)
    for event in script_until_midway():
        renderer(event)
        clock.advance(1)
    renderer.close()  # as on Ctrl-C: stops the display, keeps the rows

    text = plain(buffer)
    assert "300/1,000" in text
    assert "0/2 mo" in text
    assert "downloading" in text
    assert "ETA 0:" in text
    renderer.close()  # idempotent


def test_bars_ignore_events_before_started_and_close_without_start() -> None:
    renderer, buffer = bars(FakeClock())
    renderer(planned("BTCUSDT", 10, (JAN,)))
    renderer.close()
    assert plain(buffer) == ""
