"""`moonshot data`: download klines into the local store and inspect what is stored.

Exit codes of `data sync`: 0 ok, 1 some series failed, 2 usage error, 3 aborted on rate limits or
an IP ban, 130 interrupted. Progress goes to stderr, the summary (or `--json`) to stdout.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler

from crypto_moonshot.cli_output import print_status, print_sync_summary, stdout_is_tty
from crypto_moonshot.config import Settings
from crypto_moonshot.data.binance import BinanceClient
from crypto_moonshot.data.status import store_report
from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.data.sync import DEFAULT_CONCURRENCY, KlineSource, SyncSummary, sync_klines
from crypto_moonshot.models import Interval
from crypto_moonshot.progress import (
    BarsProgress,
    ProgressMode,
    ProgressRenderer,
    make_renderer,
)

data_app = typer.Typer(no_args_is_help=True, help="Download and inspect Binance kline data.")

DEFAULT_START = "2020-01-01"
DEFAULT_INTERVAL = Interval.M5
EXIT_FAILED = 1
EXIT_ABORTED = 3
EXIT_INTERRUPTED = 130

_SYMBOL_RE = re.compile(r"[A-Z0-9]+")
_LOGGER_NAME = "crypto_moonshot"
_EXIT_CODES = {"ok": 0, "failed": EXIT_FAILED, "aborted": EXIT_ABORTED}


# --------------------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------------------


def make_source(settings: Settings, max_rate_limit_hits: int) -> KlineSource:
    """Create the run's kline source. Tests replace this with a fake; nothing else builds one."""
    return BinanceClient.from_settings(settings, max_rate_limit_hits=max_rate_limit_hits)


async def close_source(source: KlineSource) -> None:
    """Release a source made by `make_source` (anything with an `aclose` coroutine)."""
    aclose = getattr(source, "aclose", None)
    if aclose is not None:
        await aclose()


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def parse_symbols(values: list[str]) -> list[str]:
    """Upper-case and validate symbols, dropping duplicates (order kept)."""
    symbols: dict[str, None] = {}
    for raw in values:
        symbol = raw.upper()
        if not _SYMBOL_RE.fullmatch(symbol):
            msg = f"invalid symbol {raw!r}: use letters and digits only, for example BTCUSDT"
            raise typer.BadParameter(msg, param_hint="SYMBOL")
        symbols[symbol] = None
    return list(symbols)


def parse_intervals(values: list[str] | None, default: list[Interval]) -> list[Interval]:
    """Validate intervals exactly (`1m` is a minute, `1M` a month), dropping duplicates."""
    if not values:
        return default
    intervals: dict[Interval, None] = {}
    for raw in values:
        try:
            intervals[Interval(raw)] = None
        except ValueError:
            choices = ", ".join(interval.value for interval in Interval)
            msg = f"invalid interval {raw!r}: choose from {choices} (case matters)"
            raise typer.BadParameter(msg, param_hint="--interval") from None
    return list(intervals)


def parse_when(text: str, *, option: str) -> datetime:
    """`YYYY-MM-DD` or an ISO-8601 datetime; UTC when no offset is given."""
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        msg = f"invalid date {text!r}: use YYYY-MM-DD or an ISO datetime such as 2024-05-01T12:00"
        raise typer.BadParameter(msg, param_hint=option) from None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _resolve_settings(data_dir: Path | None) -> Settings:
    settings = Settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    return settings


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------


@contextlib.contextmanager
def _logging(verbose: bool, console: Console | None) -> Iterator[None]:
    """Route `crypto_moonshot` logs to stderr (DEBUG with -v, else WARNING); undo on exit.

    With a rich console (bars mode) logs go through it so they print above the bars.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    handler: logging.Handler
    if console is not None:
        handler = RichHandler(console=console, show_path=False, markup=False)
    else:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.propagate = False
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


# --------------------------------------------------------------------------------------
# data sync
# --------------------------------------------------------------------------------------


async def _run_sync(
    settings: Settings,
    series: list[tuple[str, Interval]],
    *,
    start: datetime,
    end: datetime,
    concurrency: int,
    max_rate_limit_hits: int,
    renderer: ProgressRenderer | None,
) -> SyncSummary:
    source = make_source(settings, max_rate_limit_hits)
    try:
        return await sync_klines(
            source,
            KlineStore(settings.data_dir),
            series,
            start=start,
            end=end,
            concurrency=concurrency,
            sink=renderer,
        )
    finally:
        if renderer is not None:
            renderer.close()
        await close_source(source)


@data_app.command()
def sync(
    symbols: Annotated[
        list[str],
        typer.Argument(metavar="SYMBOL...", help="Binance symbols, e.g. BTCUSDT ETHUSDT."),
    ],
    interval: Annotated[
        list[str] | None,
        typer.Option(
            "--interval",
            "-i",
            help="Candle interval; repeat for several (case matters: 1m minute, 1M month). "
            f"Default {DEFAULT_INTERVAL.value}.",
        ),
    ] = None,
    start: Annotated[
        str, typer.Option(help="First day wanted: YYYY-MM-DD or ISO datetime (UTC by default).")
    ] = DEFAULT_START,
    end: Annotated[
        str | None, typer.Option(help="End of the range, same formats. Default: now.")
    ] = None,
    concurrency: Annotated[
        int, typer.Option(min=1, help="Month downloads running at once, across all series.")
    ] = DEFAULT_CONCURRENCY,
    max_rate_limit_hits: Annotated[
        int, typer.Option(min=1, help="Abort the run after this many HTTP 429 responses.")
    ] = 3,
    data_dir: Annotated[
        Path | None,
        typer.Option(help="Data folder (default: $MOONSHOT_DATA_DIR, else ./data)."),
    ] = None,
    progress: Annotated[
        ProgressMode,
        typer.Option(
            help="Progress on stderr: bars (live, for terminals), lines (plain text for logs and "
            "agents), none. auto picks bars on a terminal, else lines."
        ),
    ] = ProgressMode.AUTO,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the summary as one JSON object on stdout.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Debug logging on stderr.")
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="No progress output.")] = False,
) -> None:
    """Download klines for SYMBOL... into the local store, fetching only what is missing.

    Re-running resumes: finished months are never fetched again.

    Exit codes: 0 ok, 1 some series failed, 2 usage error, 3 aborted (rate limits or IP ban),
    130 interrupted.
    """
    wanted_symbols = parse_symbols(symbols)
    intervals = parse_intervals(interval, [DEFAULT_INTERVAL])
    start_dt = parse_when(start, option="--start")
    end_dt = parse_when(end, option="--end") if end is not None else datetime.now(UTC)
    if start_dt >= end_dt:
        msg = f"--start ({start_dt:%Y-%m-%d %H:%M}) must be before --end ({end_dt:%Y-%m-%d %H:%M})"
        raise typer.BadParameter(msg, param_hint="--start")
    if verbose and quiet:
        msg = "--verbose and --quiet cannot be used together"
        raise typer.BadParameter(msg, param_hint="--verbose")

    settings = _resolve_settings(data_dir)
    series = [(symbol, interval_) for symbol in wanted_symbols for interval_ in intervals]
    renderer = make_renderer(ProgressMode.NONE if quiet else progress)
    console = renderer.console if isinstance(renderer, BarsProgress) else None

    with _logging(verbose, console):
        try:
            summary = asyncio.run(
                _run_sync(
                    settings,
                    series,
                    start=start_dt,
                    end=end_dt,
                    concurrency=concurrency,
                    max_rate_limit_hits=max_rate_limit_hits,
                    renderer=renderer,
                )
            )
        except KeyboardInterrupt:
            if renderer is not None:
                renderer.close()
            typer.echo(
                "Interrupted. Months already written are kept; run the same command to resume.",
                err=True,
            )
            raise typer.Exit(EXIT_INTERRUPTED) from None

    if json_output:
        typer.echo(json.dumps(summary.to_dict()))
    else:
        print_sync_summary(summary)
    code = _EXIT_CODES[summary.outcome]
    if code:
        raise typer.Exit(code)


# --------------------------------------------------------------------------------------
# data status
# --------------------------------------------------------------------------------------


@data_app.command()
def status(
    symbols: Annotated[
        list[str] | None,
        typer.Argument(metavar="[SYMBOL]...", help="Only these symbols (default: all stored)."),
    ] = None,
    interval: Annotated[
        list[str] | None,
        typer.Option("--interval", "-i", help="Only these intervals; repeat for several."),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(help="Data folder (default: $MOONSHOT_DATA_DIR, else ./data)."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the report as a JSON list on stdout.")
    ] = False,
) -> None:
    """Show what is stored: range, candle counts, gaps and invalid files. Reads disk only."""
    wanted_symbols = parse_symbols(symbols) if symbols else []
    intervals = parse_intervals(interval, [])
    settings = _resolve_settings(data_dir)
    reports = store_report(KlineStore(settings.data_dir), wanted_symbols, intervals)
    if json_output:
        typer.echo(json.dumps([report.to_dict() for report in reports]))
    elif not reports:
        if wanted_symbols or intervals:
            typer.echo(
                f"No stored data matches those symbols and intervals in {settings.data_dir}."
            )
        else:
            typer.echo(
                f"No data stored in {settings.data_dir} yet. "
                "Download some with: moonshot data sync BTCUSDT -i 1h"
            )
    else:
        print_status(reports, tty=stdout_is_tty())
