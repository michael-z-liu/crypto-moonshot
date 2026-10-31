import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from kline_fakes import JAN_1, FakeBinance, hours, unknown_symbol
from typer.testing import CliRunner

from crypto_moonshot import data_cli
from crypto_moonshot.cli import app
from crypto_moonshot.config import Settings
from crypto_moonshot.data.binance import RateLimitAbortError
from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.data.sync import KlineSource
from crypto_moonshot.models import Interval

H1 = Interval.H1
RANGE = ["--start", "2024-01-01", "--end", "2024-01-03"]  # 48 hourly candles
STAMP = re.compile(r"^\d\d:\d\d:\d\d ")

Install = Callable[[FakeBinance], FakeBinance]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "store"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MOONSHOT_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)  # a stray ./data would land in tmp_path, not the repo


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> Install:
    """Replace the source factory with a fake; also records the factory's arguments."""

    def _install(fake: FakeBinance) -> FakeBinance:
        def make_source(settings: Settings, max_rate_limit_hits: int) -> KlineSource:
            fake.settings = settings  # type: ignore[attr-defined]
            fake.max_rate_limit_hits = max_rate_limit_hits  # type: ignore[attr-defined]
            return fake

        monkeypatch.setattr(data_cli, "make_source", make_source)
        return fake

    return _install


def run(*args: str, env: dict[str, str] | None = None) -> Any:
    return CliRunner().invoke(app, list(args), env=env)


def sync_args(data_dir: Path, *extra: str) -> list[str]:
    return ["data", "sync", "BTCUSDT", "-i", "1h", *RANGE, "--data-dir", str(data_dir), *extra]


# -- data sync: output -----------------------------------------------------------------------


def test_sync_prints_lines_progress_on_stderr_and_a_summary_on_stdout(
    install: Install, data_dir: Path
) -> None:
    fake = install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))

    result = run(*sync_args(data_dir))

    assert result.exit_code == 0, result.output
    stderr = result.stderr.splitlines()
    assert all(STAMP.match(line) for line in stderr)
    assert "\x1b" not in result.stderr
    assert "sync started: 1 series, 2024-01-01 00:00 to 2024-01-03 00:00 UTC, concurrency 8" in (
        result.stderr
    )
    assert "BTCUSDT 1h  planned: 0 candles stored, 48 to fetch in 1 months" in result.stderr
    assert "BTCUSDT 1h  done: added 48 candles, 1 months written" in result.stderr
    assert "sync finished: ok, 48 candles added" in result.stderr

    header, row, *_ = result.stdout.splitlines()
    assert header.split() == [
        "SYMBOL",
        "INTERVAL",
        "STATUS",
        "ADDED",
        "MONTHS",
        "GAPS",
        "STORED",
        "RANGE",
        "(UTC)",
        "ERROR",
    ]
    assert row.split() == [
        "BTCUSDT",
        "1h",
        "ok",
        "48",
        "1",
        "0",
        "2024-01-01",
        "00:00",
        "to",
        "2024-01-02",
        "23:00",
    ]
    assert "48 candles added, 2 requests, 0 rate-limit hits" in result.stdout
    assert "max used weight 0/6000 per minute (0.0%)" in result.stdout
    assert "\x1b" not in result.stdout
    assert list(KlineStore(data_dir).stored_open_times("BTCUSDT", H1)) == hours(JAN_1, 48)
    assert fake.closed


def test_sync_second_run_is_up_to_date(install: Install, data_dir: Path) -> None:
    install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))
    run(*sync_args(data_dir))
    again = install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))

    result = run(*sync_args(data_dir))

    assert result.exit_code == 0
    assert "BTCUSDT 1h  up to date: nothing to fetch" in result.stderr
    assert again.fetches == []
    assert result.stdout.splitlines()[1].split()[:3] == ["BTCUSDT", "1h", "up_to_date"]


def test_sync_json_prints_only_the_summary_on_stdout(install: Install, data_dir: Path) -> None:
    install(FakeBinance({"BTCUSDT": hours(JAN_1, 48), "ETHUSDT": hours(JAN_1, 24)}))

    result = run(
        "data", "sync", "btcusdt", "ETHUSDT", "-i", "1h", *RANGE,
        "--data-dir", str(data_dir), "--json", "--progress", "lines",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)  # the whole of stdout is one JSON document
    assert result.stdout.count("\n") == 1
    assert set(summary) >= {
        "outcome", "aborted", "start", "end", "now", "duration_s", "requests",
        "rate_limit_hits", "max_used_weight", "candles_added", "series",
    }  # fmt: skip
    assert summary["outcome"] == "ok"
    assert summary["aborted"] is None
    assert summary["candles_added"] == 72
    assert summary["start"] == "2024-01-01T00:00:00+00:00"
    btc, eth = summary["series"]
    assert (btc["symbol"], btc["interval"], btc["status"]) == ("BTCUSDT", "1h", "ok")
    assert (btc["candles_added"], btc["months_written"], btc["error"]) == (48, 1, None)
    assert btc["first_open_time"] == "2024-01-01T00:00:00+00:00"
    assert btc["last_open_time"] == "2024-01-02T23:00:00+00:00"
    assert (eth["symbol"], eth["candles_added"]) == ("ETHUSDT", 24)
    assert "sync started: 2 series" in result.stderr  # progress still goes to stderr


def test_sync_quiet_prints_no_progress(install: Install, data_dir: Path) -> None:
    install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))

    result = run(*sync_args(data_dir, "-q"))

    assert result.exit_code == 0
    assert result.stderr == ""
    assert "BTCUSDT" in result.stdout


def test_sync_progress_none_and_bars_modes(install: Install, data_dir: Path) -> None:
    install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))
    quiet = run(*sync_args(data_dir, "--progress", "none"))
    assert quiet.exit_code == 0
    assert quiet.stderr == ""

    install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))
    bars = run(*sync_args(data_dir, "--progress", "bars"))  # not a terminal: final frame only
    assert bars.exit_code == 0
    assert "BTCUSDT 1h" in bars.stderr
    assert "TOTAL" in bars.stderr


def test_sync_verbose_logs_debug_to_stderr(install: Install, data_dir: Path) -> None:
    install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))

    result = run(*sync_args(data_dir, "-v", "--progress", "none"))

    assert result.exit_code == 0
    assert "DEBUG crypto_moonshot.data.sync: BTCUSDT 1h: 0 candles stored, 48 missing" in (
        result.stderr
    )
    assert "DEBUG" not in result.stdout


def test_sync_verbose_with_bars_routes_logs_through_the_progress_console(
    install: Install, data_dir: Path
) -> None:
    install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))

    result = run(*sync_args(data_dir, "-v", "--progress", "bars"))

    assert result.exit_code == 0
    assert "DEBUG" in result.stderr
    assert "BTCUSDT 1h: 0 candles stored, 48 missing" in result.stderr


def test_sync_passes_options_to_the_source_factory(install: Install, data_dir: Path) -> None:
    fake = install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))

    run(*sync_args(data_dir, "--max-rate-limit-hits", "5", "--concurrency", "2", "--json"))

    assert fake.max_rate_limit_hits == 5  # type: ignore[attr-defined]
    assert fake.settings.data_dir == data_dir  # type: ignore[attr-defined]


def test_sync_several_symbols_and_intervals_run_as_one_series_each(
    install: Install, data_dir: Path
) -> None:
    install(FakeBinance())  # no data anywhere: every series ends as no_data

    result = run(
        "data", "sync", "BTCUSDT", "ETHUSDT", "-i", "1m", "-i", "1M", *RANGE,
        "--data-dir", str(data_dir), "--json", "--progress", "none",
    )  # fmt: skip

    series = json.loads(result.stdout)["series"]
    assert [(s["symbol"], s["interval"], s["status"]) for s in series] == [
        ("BTCUSDT", "1m", "no_data"),
        ("BTCUSDT", "1M", "no_data"),
        ("ETHUSDT", "1m", "no_data"),
        ("ETHUSDT", "1M", "no_data"),
    ]


# -- data sync: exit codes ---------------------------------------------------------------------


def test_sync_exits_1_when_a_series_fails_and_the_others_complete(
    install: Install, data_dir: Path
) -> None:
    install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}, errors={"BADUSDT": unknown_symbol()}))

    result = run(
        "data", "sync", "BTCUSDT", "BADUSDT", "-i", "1h", *RANGE, "--data-dir", str(data_dir)
    )

    assert result.exit_code == 1
    assert "BADUSDT 1h  FAILED: HTTP 400, code -1121: Invalid symbol." in result.stderr
    assert "sync finished: failed" in result.stderr
    rows = {line.split()[0]: line for line in result.stdout.splitlines()[1:3]}
    assert rows["BTCUSDT"].split()[2] == "ok"
    assert rows["BADUSDT"].split()[2] == "failed"
    assert "code -1121" in rows["BADUSDT"]
    assert KlineStore(data_dir).stored_open_times("BTCUSDT", H1).len() == 48


def test_sync_json_reports_the_failure(install: Install, data_dir: Path) -> None:
    install(FakeBinance(errors={"BADUSDT": unknown_symbol()}))

    result = run(
        "data", "sync", "BADUSDT", "-i", "1h", *RANGE, "--data-dir", str(data_dir), "--json"
    )

    assert result.exit_code == 1
    summary = json.loads(result.stdout)
    assert summary["outcome"] == "failed"
    assert "-1121" in summary["series"][0]["error"]


def test_sync_exits_3_when_aborted_by_rate_limits(install: Install, data_dir: Path) -> None:
    abort = RateLimitAbortError("aborting after 3 rate-limit (HTTP 429) responses", status=429)
    install(FakeBinance(errors={"BTCUSDT": abort}))

    result = run(*sync_args(data_dir))

    assert result.exit_code == 3
    assert "ABORTED: aborting after 3 rate-limit (HTTP 429) responses." in result.stdout
    assert "sync finished: aborted" in result.stderr
    assert "BTCUSDT 1h  ABORTED:" in result.stderr


def test_sync_json_aborted_summary(install: Install, data_dir: Path) -> None:
    abort = RateLimitAbortError("too many 429s", status=429)
    install(FakeBinance(errors={"BTCUSDT": abort}))

    result = run(*sync_args(data_dir, "--json"))

    assert result.exit_code == 3
    summary = json.loads(result.stdout)
    assert (summary["outcome"], summary["aborted"]) == ("aborted", "too many 429s")


def test_sync_exits_130_on_ctrl_c_and_keeps_going_cleanly(install: Install, data_dir: Path) -> None:
    install(FakeBinance(errors={"BTCUSDT": KeyboardInterrupt()}))

    result = run(*sync_args(data_dir))

    assert result.exit_code == 130
    assert "Interrupted. Months already written are kept" in result.stderr


# -- data sync: usage errors -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["-i", "1x"], "invalid interval '1x'"),
        (["-i", "1H"], "invalid interval '1H'"),
        (["--start", "yesterday"], "invalid date 'yesterday'"),
        (["--end", "2024-13-45"], "invalid date '2024-13-45'"),
        (["--start", "2024-01-03", "--end", "2024-01-03"], "must be before --end"),
        (["--start", "2024-02-01", "--end", "2024-01-03"], "must be before --end"),
        (["--concurrency", "0"], "concurrency"),
        (["--max-rate-limit-hits", "0"], "max-rate-limit-hits"),
        (["--progress", "fancy"], "progress"),
        (["-v", "-q"], "cannot be used together"),
    ],
)
def test_sync_usage_errors_exit_2(
    install: Install, data_dir: Path, extra: list[str], message: str
) -> None:
    fake = install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))
    args = ["data", "sync", "BTCUSDT", "--data-dir", str(data_dir)]
    if not any(flag in extra for flag in ("--start", "--end")):
        args += RANGE

    result = run(*args, *extra)

    assert result.exit_code == 2, result.output
    assert message in result.stderr
    assert fake.fetches == []
    assert not data_dir.exists()


@pytest.mark.parametrize("symbol", ["BTC-USDT", "BTC/USDT", "BTC USDT", ""])
def test_sync_bad_symbol_exits_2(install: Install, data_dir: Path, symbol: str) -> None:
    install(FakeBinance())

    result = run("data", "sync", symbol, *RANGE, "--data-dir", str(data_dir))

    assert result.exit_code == 2
    assert "invalid symbol" in result.stderr


def test_sync_needs_a_symbol(install: Install) -> None:
    install(FakeBinance())
    assert run("data", "sync").exit_code == 2


def test_interval_is_case_sensitive_and_symbols_are_upper_cased(
    install: Install, data_dir: Path
) -> None:
    install(FakeBinance())

    result = run(
        "data", "sync", "btcusdt", "-i", "1M", "-i", "1m", *RANGE,
        "--data-dir", str(data_dir), "--json", "--progress", "none",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    series = json.loads(result.stdout)["series"]
    assert [(s["symbol"], s["interval"]) for s in series] == [("BTCUSDT", "1M"), ("BTCUSDT", "1m")]


def test_naive_dates_are_utc_and_offsets_are_converted(install: Install, data_dir: Path) -> None:
    install(FakeBinance())

    result = run(
        "data", "sync", "BTCUSDT", "-i", "1h",
        "--start", "2024-01-01T02:00+02:00", "--end", "2024-01-03T12:30",
        "--data-dir", str(data_dir), "--json", "--progress", "none",
    )  # fmt: skip

    summary = json.loads(result.stdout)
    assert summary["start"] == "2024-01-01T00:00:00+00:00"
    assert summary["end"] == "2024-01-03T12:30:00+00:00"


def test_end_defaults_to_now(install: Install, data_dir: Path) -> None:
    install(FakeBinance())
    before = datetime.now(UTC)

    result = run(
        "data", "sync", "BTCUSDT", "-i", "1h", "--start", "2024-01-01",
        "--data-dir", str(data_dir), "--json", "--progress", "none",
    )  # fmt: skip

    end = datetime.fromisoformat(json.loads(result.stdout)["end"])
    assert before <= end <= datetime.now(UTC)


def test_default_interval_is_5m(install: Install, data_dir: Path) -> None:
    install(FakeBinance())
    result = run(
        "data",
        "sync",
        "BTCUSDT",
        *RANGE,
        "--data-dir",
        str(data_dir),
        "--json",
        "--progress",
        "none",
    )
    assert json.loads(result.stdout)["series"][0]["interval"] == "5m"


# -- data dir ----------------------------------------------------------------------------------


def test_data_dir_option_and_environment_variable(
    install: Install, tmp_path: Path, data_dir: Path
) -> None:
    install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))
    by_option = run(*sync_args(data_dir, "--progress", "none"))
    assert by_option.exit_code == 0
    assert (data_dir / "klines" / "spot" / "interval=1h" / "symbol=BTCUSDT").is_dir()

    from_env = tmp_path / "from-env"
    install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))
    result = run(
        "data", "sync", "BTCUSDT", "-i", "1h", *RANGE, "--progress", "none",
        env={"MOONSHOT_DATA_DIR": str(from_env)},
    )  # fmt: skip
    assert result.exit_code == 0
    assert (from_env / "klines" / "spot" / "interval=1h" / "symbol=BTCUSDT").is_dir()
    assert not (tmp_path / "data" / "klines").exists()  # the default ./data was not used


def test_option_beats_the_environment_variable(
    install: Install, tmp_path: Path, data_dir: Path
) -> None:
    install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))
    ignored = tmp_path / "ignored"

    result = run(
        *sync_args(data_dir, "--progress", "none"), env={"MOONSHOT_DATA_DIR": str(ignored)}
    )

    assert result.exit_code == 0
    assert data_dir.exists()
    assert not ignored.exists()


# -- data status -------------------------------------------------------------------------------


def seed_store_with_gap_and_invalid_file(install: Install, data_dir: Path) -> None:
    """48 hourly candles with a 3-hour hole (a recorded gap), plus a corrupt February file."""
    without = {t for t in hours(JAN_1, 48) if 10 <= (t - JAN_1).total_seconds() / 3600 < 13}
    install(FakeBinance({"BTCUSDT": [t for t in hours(JAN_1, 48) if t not in without]}))
    assert run(*sync_args(data_dir, "--progress", "none")).exit_code == 0
    series_dir = data_dir / "klines" / "spot" / "interval=1h" / "symbol=BTCUSDT"
    (series_dir / "2024-02.parquet").write_bytes(b"this is not parquet")


def test_status_table_after_a_sync_with_a_gap_and_an_invalid_file(
    install: Install, data_dir: Path
) -> None:
    seed_store_with_gap_and_invalid_file(install, data_dir)
    # The fake lists Jan 1 00:00 as the listing; the hole is found by the sync and recorded.

    result = run("data", "status", "--data-dir", str(data_dir))

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0].split()[:4] == ["SYMBOL", "INTERVAL", "FIRST", "(UTC)"]
    row = lines[1].split()
    # candles expected missing gaps(ranges, candles) files invalid
    assert row[6:13] == ["45", "48", "0", "1", "(3)", "2", "1"]
    assert "gap: 2024-01-01 10:00 to 2024-01-01 13:00 (3 candles)" in result.stdout
    assert "invalid file 2024-02: unreadable parquet file" in result.stdout
    assert "\x1b" not in result.stdout


def test_status_json_after_a_sync_with_a_gap_and_an_invalid_file(
    install: Install, data_dir: Path
) -> None:
    seed_store_with_gap_and_invalid_file(install, data_dir)

    result = run("data", "status", "--data-dir", str(data_dir), "--json")

    assert result.exit_code == 0
    (report,) = json.loads(result.stdout)
    assert report["symbol"] == "BTCUSDT"
    assert report["interval"] == "1h"
    assert report["listing"] == "2024-01-01T00:00:00+00:00"
    assert report["first_open_time"] == "2024-01-01T00:00:00+00:00"
    assert report["last_open_time"] == "2024-01-02T23:00:00+00:00"
    assert report["candles_stored"] == 45
    assert report["expected_candles"] == 48
    assert report["missing_candles"] == 0  # the hole is a recorded gap
    assert report["gap_ranges"] == 1
    assert report["gap_candles"] == 3
    assert report["gaps"] == [
        {"start": "2024-01-01T10:00:00+00:00", "end": "2024-01-01T13:00:00+00:00", "candles": 3}
    ]
    assert report["month_files"] == 2
    (invalid,) = report["invalid_files"]
    assert invalid["month"] == "2024-02"
    assert "unreadable parquet file" in invalid["problems"][0]
    assert report["last_candle_age_s"] > 0


def test_status_counts_candles_that_are_missing_but_not_recorded_gaps(
    install: Install, data_dir: Path
) -> None:
    install(FakeBinance({"BTCUSDT": hours(JAN_1, 48)}))
    run(*sync_args(data_dir, "--progress", "none"))
    # Rewrite January without 5 candles; _meta.json records no gap for them.
    store = KlineStore(data_dir)
    from kline_fakes import candle_frame

    keep = [t for t in hours(JAN_1, 48) if t not in set(hours(JAN_1, 30)[20:25])]
    store.write_month("BTCUSDT", H1, JAN_1, candle_frame(keep))

    (report,) = json.loads(run("data", "status", "--data-dir", str(data_dir), "--json").stdout)

    assert (report["candles_stored"], report["expected_candles"]) == (43, 48)
    assert report["missing_candles"] == 5
    assert report["gap_candles"] == 0


def test_status_filters_by_symbol_and_interval(install: Install, data_dir: Path) -> None:
    install(FakeBinance({"BTCUSDT": hours(JAN_1, 48), "ETHUSDT": hours(JAN_1, 48)}))
    run("data", "sync", "BTCUSDT", "ETHUSDT", "-i", "1h", *RANGE, "--data-dir", str(data_dir), "-q")

    def status(*args: str) -> list[dict[str, Any]]:
        out = run("data", "status", *args, "--data-dir", str(data_dir), "--json")
        assert out.exit_code == 0
        return json.loads(out.stdout)  # type: ignore[no-any-return]

    assert [r["symbol"] for r in status()] == ["BTCUSDT", "ETHUSDT"]
    assert [r["symbol"] for r in status("ethusdt")] == ["ETHUSDT"]
    assert status("-i", "1h") != []
    assert status("-i", "5m") == []
    assert status("SOLUSDT") == []
    message = run("data", "status", "SOLUSDT", "--data-dir", str(data_dir))
    assert message.exit_code == 0
    assert "No stored data matches" in message.stdout


def test_status_on_an_empty_store(data_dir: Path) -> None:
    table = run("data", "status", "--data-dir", str(data_dir))
    assert table.exit_code == 0
    assert "No data stored" in table.stdout
    assert "moonshot data sync" in table.stdout

    as_json = run("data", "status", "--data-dir", str(data_dir), "--json")
    assert as_json.exit_code == 0
    assert json.loads(as_json.stdout) == []
    assert not data_dir.exists()  # status never writes


def test_status_does_not_write(install: Install, data_dir: Path) -> None:
    seed_store_with_gap_and_invalid_file(install, data_dir)
    before = {p: p.stat().st_mtime_ns for p in data_dir.rglob("*")}

    run("data", "status", "--data-dir", str(data_dir))

    assert {p: p.stat().st_mtime_ns for p in data_dir.rglob("*")} == before


def test_status_rejects_a_bad_symbol_and_interval(data_dir: Path) -> None:
    assert run("data", "status", "BTC-USDT", "--data-dir", str(data_dir)).exit_code == 2
    assert run("data", "status", "-i", "1x", "--data-dir", str(data_dir)).exit_code == 2


def test_data_help_lists_the_commands() -> None:
    result = run("data", "--help")
    assert result.exit_code == 0
    assert "sync" in result.stdout
    assert "status" in result.stdout
