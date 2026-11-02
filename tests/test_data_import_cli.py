import json
import re
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from kline_fakes import JAN_1, candle_frame, hours
from typer.testing import CliRunner

from crypto_moonshot.cli import app
from crypto_moonshot.data.store import SCHEMA_VERSION_KEY, KlineStore, SeriesMeta
from crypto_moonshot.models import Interval

H1 = Interval.H1


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MOONSHOT_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    store = KlineStore(root)
    store.write_month("BTCUSDT", H1, JAN_1, candle_frame(hours(JAN_1, 48)))
    store.write_meta("BTCUSDT", H1, SeriesMeta(listing=JAN_1))
    bad = root / "klines" / "spot" / "interval=1h" / "symbol=ETHUSDT" / "2024-01.parquet"
    bad.parent.mkdir(parents=True)
    candle_frame(hours(JAN_1, 3)).write_parquet(bad, metadata={SCHEMA_VERSION_KEY: "2"})
    return root


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "dest"


def run(*args: str) -> Any:
    return CliRunner().invoke(app, ["data", "import", *args])


def error_text(result: Any) -> str:
    """The usage error with rich's box drawing and line wrapping removed."""
    return " ".join(re.sub(r"[\u2500-\u257f]", " ", result.output).split())


def test_import_prints_a_table_skipped_files_and_totals(source: Path, data_dir: Path) -> None:
    result = run(str(source), "--data-dir", str(data_dir))

    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.stdout
    lines = result.stdout.splitlines()
    assert lines[0].split() == [
        "SYMBOL", "INTERVAL", "IMPORTED", "MERGED", "REPLACED",
        "UNCHANGED", "SKIPPED", "ADDED", "META",
    ]  # fmt: skip
    assert lines[1].split() == ["BTCUSDT", "1h", "1", "0", "0", "0", "0", "48", "updated"]
    assert lines[2].split() == ["ETHUSDT", "1h", "0", "0", "0", "0", "1", "0", "none"]
    assert (
        "skipped klines/spot/interval=1h/symbol=ETHUSDT/2024-01.parquet: schema version '2'"
        in lines[3]
    )
    assert lines[-1] == (
        "2 files: 1 imported, 0 merged, 0 replaced, 0 unchanged, 1 skipped; 48 candles added"
    )
    assert list(KlineStore(data_dir).stored_open_times("BTCUSDT", H1)) == hours(JAN_1, 48)
    assert KlineStore(data_dir).read_meta("BTCUSDT", H1).listing == JAN_1


def test_dry_run_says_nothing_was_written(source: Path, data_dir: Path) -> None:
    result = run(str(source), "--data-dir", str(data_dir), "--dry-run")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1].endswith("48 candles added; dry run: nothing written")
    assert not data_dir.exists()


def test_json_prints_only_the_report(source: Path, data_dir: Path) -> None:
    result = run(str(source), "--data-dir", str(data_dir), "--json")

    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert result.stdout.count("\n") == 1
    assert set(report) == {"source", "destination", "dry_run", "totals", "series", "files"}
    assert report["dry_run"] is False
    assert report["totals"] == {
        "files": 2, "imported": 1, "merged": 0, "replaced": 0, "unchanged": 0,
        "skipped": 1, "candles_added": 48,
    }  # fmt: skip
    imported = next(f for f in report["files"] if f["action"] == "imported")
    assert imported["candles_added"] == 48
    assert imported["month"] == "2024-01"
    skipped = next(f for f in report["files"] if f["action"] == "skipped")
    assert "schema version '2'" in skipped["problems"][0]


def test_skipped_files_still_exit_zero_and_data_dir_comes_from_the_environment(
    source: Path, tmp_path: Path
) -> None:
    result = CliRunner().invoke(
        app, ["data", "import", str(source)], env={"MOONSHOT_DATA_DIR": str(tmp_path / "env")}
    )

    assert result.exit_code == 0, result.output
    assert KlineStore(tmp_path / "env").stored_open_times("BTCUSDT", H1).len() == 48


def test_missing_source_is_a_usage_error(tmp_path: Path, data_dir: Path) -> None:
    result = run(str(tmp_path / "nope"), "--data-dir", str(data_dir))

    assert result.exit_code == 2
    assert "does not exist" in error_text(result)


def test_source_without_klines_is_a_usage_error_with_a_hint(tmp_path: Path, source: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run(str(empty), "--data-dir", str(tmp_path / "dest"))
    assert result.exit_code == 2
    assert "no klines/ folder" in error_text(result)

    result = run(str(source / "klines"), "--data-dir", str(tmp_path / "dest"))
    assert result.exit_code == 2
    assert "did you mean" in error_text(result)


def test_source_same_as_destination_is_a_usage_error(source: Path) -> None:
    result = run(str(source), "--data-dir", str(source))

    assert result.exit_code == 2
    assert "is the destination data folder" in error_text(result)


def test_source_is_untouched_by_the_cli(source: Path, data_dir: Path) -> None:
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}

    assert run(str(source), "--data-dir", str(data_dir)).exit_code == 0

    assert {p: p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    assert isinstance(pl.read_parquet(next(source.rglob("2024-01.parquet"))), pl.DataFrame)
