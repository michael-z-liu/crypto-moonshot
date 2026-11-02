from datetime import UTC, datetime

import pytest

from crypto_moonshot.cli_output import format_age, print_status, print_sync_summary, print_table
from crypto_moonshot.data.status import SeriesReport
from crypto_moonshot.data.sync import SeriesResult, SeriesStatus, SyncSummary
from crypto_moonshot.models import Interval

H1 = Interval.H1
T0 = datetime(2024, 1, 1, tzinfo=UTC)
T1 = datetime(2024, 1, 2, 23, tzinfo=UTC)


def summary(*results: SeriesResult, aborted: str | None = None) -> SyncSummary:
    return SyncSummary(
        series=results,
        start=T0,
        end=T1,
        now=T1,
        duration_s=2.0,
        requests=10,
        rate_limit_hits=2,
        max_used_weight=1500,
        aborted=aborted,
    )


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(None, "-"), (5, "5s"), (125, "2m"), (3 * 3600 + 20 * 60, "3h 20m"), (90000, "1d 1h")],
)
def test_format_age(seconds: float | None, text: str) -> None:
    assert format_age(seconds) == text


def test_plain_table_is_aligned_with_no_trailing_spaces(capsys: pytest.CaptureFixture[str]) -> None:
    print_table(["A", "NUM"], [["xx", "1"], ["y", "1,000"]], tty=False, right={1})
    assert capsys.readouterr().out == "A     NUM\nxx      1\ny   1,000\n"


def test_sync_summary_in_a_terminal_uses_a_rich_table(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    ok = SeriesResult("BTCUSDT", H1, SeriesStatus.OK, 1, 48, 48, 1, 0, T0, T1)
    bad = SeriesResult("BADUSDT", H1, SeriesStatus.FAILED, error="HTTP 400, code -1121")

    print_sync_summary(summary(ok, bad), tty=True)

    out = capsys.readouterr().out
    assert "STORED RANGE (UTC)" in out
    assert "2024-01-01 00:00 \N{RIGHTWARDS ARROW} 2024-01-02 23:00" in out
    assert "HTTP 400, code -1121" in out
    assert "max used weight 1500/6000 per minute (25.0%)" in out
    assert "2 rate-limit hits" in out


def test_aborted_summary_is_prominent(capsys: pytest.CaptureFixture[str]) -> None:
    result = SeriesResult("BTCUSDT", H1, SeriesStatus.ABORTED, error="IP banned")
    print_sync_summary(summary(result, aborted="IP banned until later"), tty=False)
    out = capsys.readouterr().out
    assert out.startswith("ABORTED: IP banned until later.")
    assert "run the same command later to resume" in out


def test_status_in_a_terminal(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    report = SeriesReport(
        symbol="BTCUSDT",
        interval=H1,
        listing=T0,
        first_open_time=T0,
        last_open_time=T1,
        candles_stored=45,
        expected_candles=48,
        missing_candles=3,
        gaps=((T0, T0.replace(hour=2)),),
        gap_candles=2,
        month_files=2,
        invalid_files=((datetime(2024, 2, 1, tzinfo=UTC), ("bad schema",)),),
        last_candle_age_s=3600.0,
    )

    print_status([report], tty=True)

    out = capsys.readouterr().out
    assert "GAPS (CANDLES)" in out
    assert "1 (2)" in out
    assert "gap: 2024-01-01 00:00 \N{RIGHTWARDS ARROW} 2024-01-01 02:00 (2 candles)" in out
    assert "invalid file 2024-02: bad schema" in out
    assert "1h 0m" in out
