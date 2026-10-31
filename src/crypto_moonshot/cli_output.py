"""Human-readable output of `moonshot data`: a rich table in a terminal, aligned text otherwise.

Everything here writes to stdout. Plain output is ASCII only and has no escape codes, so an agent
or a log file can read it.
"""

from __future__ import annotations

import sys
from collections.abc import Collection, Sequence
from datetime import datetime

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from crypto_moonshot.data.binance import IP_WEIGHT_LIMIT_1M
from crypto_moonshot.data.importer import ImportAction, ImportReport, MetaAction
from crypto_moonshot.data.status import SeriesReport
from crypto_moonshot.data.sync import SeriesStatus, SyncSummary
from crypto_moonshot.progress import format_count, short_reason

_STATUS_STYLES = {
    SeriesStatus.OK: "green",
    SeriesStatus.UP_TO_DATE: "green",
    SeriesStatus.NO_DATA: "yellow",
    SeriesStatus.FAILED: "bold red",
    SeriesStatus.ABORTED: "yellow",
}


def stdout_is_tty() -> bool:
    return sys.stdout.isatty()


def format_time(dt: datetime | None) -> str:
    return "-" if dt is None else dt.strftime("%Y-%m-%d %H:%M")


def format_age(seconds: float | None) -> str:
    """Compact age such as `5m`, `3h 20m` or `12d 4h`."""
    if seconds is None:
        return "-"
    total = max(0, int(seconds))
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m"
    return f"{total}s"


def print_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    tty: bool,
    right: Collection[int] = (),
    styles: Sequence[str] = (),
) -> None:
    """Print rows under headers. `right` lists right-aligned columns; `styles` is per row (rich)."""
    if tty:
        table = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False)
        for index, header in enumerate(headers):
            table.add_column(header, justify="right" if index in right else "left")
        for index, row in enumerate(rows):
            style = styles[index] if index < len(styles) else ""
            table.add_row(*(Text(cell, style=style) for cell in row))
        Console().print(table)
        return
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells: Sequence[str]) -> str:
        parts = [
            cell.rjust(widths[i]) if i in right else cell.ljust(widths[i])
            for i, cell in enumerate(cells)
        ]
        return "  ".join(parts).rstrip()

    print(line(headers))
    for row in rows:
        print(line(row))


def print_notice(text: str, *, tty: bool, style: str = "bold red") -> None:
    if tty:
        Console().print(Text(text, style=style))
    else:
        print(text)


# --------------------------------------------------------------------------------------
# data sync
# --------------------------------------------------------------------------------------


def print_sync_summary(summary: SyncSummary, *, tty: bool | None = None) -> None:
    """The final table of a sync run, then one line of run totals."""
    tty = stdout_is_tty() if tty is None else tty
    arrow = " \N{RIGHTWARDS ARROW} " if tty else " to "
    if summary.aborted:
        print_notice(
            f"ABORTED: {summary.aborted}. Months already written are kept; "
            "run the same command later to resume.",
            tty=tty,
        )
    rows: list[list[str]] = []
    styles: list[str] = []
    for result in summary.series:
        stored = (
            f"{format_time(result.first_open_time)}{arrow}{format_time(result.last_open_time)}"
            if result.first_open_time and result.last_open_time
            else "-"
        )
        rows.append(
            [
                result.symbol,
                result.interval.value,
                result.status.value,
                format_count(result.candles_added),
                str(result.months_written),
                format_count(result.gap_candles_found),
                stored,
                short_reason(result.error, 200) if result.error else "",
            ]
        )
        styles.append(_STATUS_STYLES[result.status])
    print_table(
        ["SYMBOL", "INTERVAL", "STATUS", "ADDED", "MONTHS", "GAPS", "STORED RANGE (UTC)", "ERROR"],
        rows,
        tty=tty,
        right={3, 4, 5},
        styles=styles,
    )
    weight = summary.max_used_weight
    added = sum(result.candles_added for result in summary.series)
    line = (
        f"{format_count(added)} candles added, {summary.requests} requests, "
        f"{summary.rate_limit_hits} rate-limit hits, "
        f"max used weight {weight}/{IP_WEIGHT_LIMIT_1M} per minute "
        f"({100 * weight / IP_WEIGHT_LIMIT_1M:.1f}%), took {summary.duration_s:.1f}s"
    )
    print_notice(line, tty=tty, style="")


# --------------------------------------------------------------------------------------
# data status
# --------------------------------------------------------------------------------------


def print_status(reports: Sequence[SeriesReport], *, tty: bool | None = None) -> None:
    """A table with one row per stored series, then the gap and invalid-file details."""
    tty = stdout_is_tty() if tty is None else tty
    rows: list[list[str]] = []
    styles: list[str] = []
    for report in reports:
        gaps = f"{len(report.gaps)} ({format_count(report.gap_candles)})" if report.gaps else "0"
        rows.append(
            [
                report.symbol,
                report.interval.value,
                format_time(report.first_open_time),
                format_time(report.last_open_time),
                format_count(report.candles_stored),
                format_count(report.expected_candles),
                format_count(report.missing_candles),
                gaps,
                str(report.month_files),
                str(len(report.invalid_files)),
                format_age(report.last_candle_age_s),
            ]
        )
        styles.append("red" if report.invalid_files or report.missing_candles else "")
    print_table(
        [
            "SYMBOL",
            "INTERVAL",
            "FIRST (UTC)",
            "LAST (UTC)",
            "CANDLES",
            "EXPECTED",
            "MISSING",
            "GAPS (CANDLES)",
            "FILES",
            "INVALID",
            "LAST AGE",
        ],
        rows,
        tty=tty,
        right={4, 5, 6, 7, 8, 9, 10},
        styles=styles,
    )
    arrow = " \N{RIGHTWARDS ARROW} " if tty else " to "
    for report in reports:
        label = f"{report.symbol} {report.interval.value}"
        for start, end in report.gaps:
            count = report.interval.count_open_times(start, end)
            candles = "1 candle" if count == 1 else f"{format_count(count)} candles"
            print_notice(
                f"{label} gap: {format_time(start)}{arrow}{format_time(end)} ({candles})",
                tty=tty,
                style="",
            )
        for month, problems in report.invalid_files:
            print_notice(
                f"{label} invalid file {month:%Y-%m}: {'; '.join(problems)}", tty=tty, style="red"
            )


# --------------------------------------------------------------------------------------
# data import
# --------------------------------------------------------------------------------------


def print_import_report(report: ImportReport, *, tty: bool | None = None) -> None:
    """A per-series table, one line per skipped file, then the totals."""
    tty = stdout_is_tty() if tty is None else tty
    if report.series:
        print_table(
            [
                "SYMBOL",
                "INTERVAL",
                "IMPORTED",
                "MERGED",
                "REPLACED",
                "UNCHANGED",
                "SKIPPED",
                "ADDED",
                "META",
            ],
            [
                [
                    item.symbol,
                    item.interval.value,
                    str(item.imported),
                    str(item.merged),
                    str(item.replaced),
                    str(item.unchanged),
                    str(item.skipped),
                    format_count(item.candles_added),
                    item.meta.value,
                ]
                for item in report.series
            ],
            tty=tty,
            right={2, 3, 4, 5, 6, 7},
            styles=["yellow" if item.skipped else "" for item in report.series],
        )
    for item in report.files:
        if item.action is ImportAction.SKIPPED:
            reasons = short_reason("; ".join(item.problems), 300)
            print_notice(f"skipped {item.path}: {reasons}", tty=tty, style="yellow")
    for entry in report.series:
        if entry.meta is MetaAction.SKIPPED:
            label = f"{entry.symbol} {entry.interval.value}"
            reasons = short_reason("; ".join(entry.meta_problems), 300)
            print_notice(f"skipped metadata of {label}: {reasons}", tty=tty, style="yellow")
    totals = report.totals()
    line = (
        f"{totals['files']} files: {totals['imported']} imported, {totals['merged']} merged, "
        f"{totals['replaced']} replaced, {totals['unchanged']} unchanged, "
        f"{totals['skipped']} skipped; {format_count(totals['candles_added'])} candles added"
    )
    if report.dry_run:
        line += "; dry run: nothing written"
    print_notice(line, tty=tty, style="")
