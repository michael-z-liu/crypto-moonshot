"""`moonshot backtest plugin` and `moonshot backtest signals`, end to end on a fake data folder."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from plugin_fakes import write_plugin
from strategy_fakes import LONE, SPIKE, at, write_capitulation_store
from typer.testing import CliRunner

from crypto_moonshot.cli import app

OUTPUTS = {"trades.parquet", "trades.csv", "equity.csv", "yearly.csv", "metrics.json"}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    write_capitulation_store(store)
    (store / "top200.txt").write_text("BTCUSDT DIPUSDT SHALLOWUSDT\n")
    (store / "delisted.txt").write_text("")
    (store / "outside_pool.txt").write_text("THINUSDT\n")
    return store


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MOONSHOT_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)


def run(*args: str) -> Any:
    return CliRunner().invoke(app, ["backtest", *args])


def common(root: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--data-dir", str(root),
        "--cache-dir", str(tmp_path / "cache"),
        "--workers", "1",
        "--out", str(tmp_path / "out"),
        *extra,
    ]  # fmt: skip


def metrics(tmp_path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((tmp_path / "out" / "metrics.json").read_text())
    return loaded


def test_plugin_without_backtest_signals_runs_rolling_windows(root: Path, tmp_path: Path) -> None:
    plugin = write_plugin(tmp_path)
    result = run("plugin", f"{plugin}:FLUSH", *common(root, tmp_path))
    assert result.exit_code == 0, result.output
    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} == OUTPUTS
    trades = pl.read_parquet(out / "trades.parquet")
    assert trades.select("symbol", "signal_time").rows() == [
        ("DIPUSDT", at(SPIKE)),
        ("DIPUSDT", at(LONE)),
    ]
    assert trades["mult"].to_list() == pytest.approx([20.0, 20.0])  # the plugin's extra column
    assert (trades["exit_time"] - trades["entry_time"]).to_list() == [timedelta(hours=1)] * 2
    doc = metrics(tmp_path)
    assert doc["strategy"] == "flush"
    assert doc["source"] == f"{plugin}:FLUSH"
    assert doc["symbols"] == "pool"
    assert doc["signals"] == 2
    assert doc["config"]["sizing"] == {"type": "equal_slots", "slots": 5}
    assert doc["config"]["exit"]["max_hold_seconds"] == 3600
    assert "YEAR" in result.stdout


def test_plugin_params_symbols_and_exit_overrides(root: Path, tmp_path: Path) -> None:
    plugin = write_plugin(tmp_path)
    result = run(
        "plugin", f"{plugin}:make", "--param", "threshold=-0.11", "--name", "mine",
        "--symbols", "dipusdt,THINUSDT", "--hold", "2h", "--take-profit", "0.5", "--stake", "40",
        *common(root, tmp_path),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    trades = pl.read_parquet(tmp_path / "out" / "trades.parquet")
    assert sorted(set(trades["symbol"])) == ["DIPUSDT", "THINUSDT"]  # THIN falls 12% > 11%
    assert set(trades["stake"].to_list()) == {40.0}
    doc = metrics(tmp_path)
    assert doc["strategy"] == "mine"
    assert doc["params"] == {"threshold": -0.11}
    assert doc["symbols"] == ["DIPUSDT", "THINUSDT"]
    assert doc["config"]["exit"]["max_hold_seconds"] == 7200
    assert doc["config"]["exit"]["take_profit"] == 0.5

    quiet = run("plugin", f"{plugin}:make", "--param", "threshold=-0.3", *common(root, tmp_path))
    assert quiet.exit_code == 0, quiet.output
    assert pl.read_parquet(tmp_path / "out" / "trades.parquet").height == 0


def test_plugin_with_backtest_signals_uses_the_batch_path(root: Path, tmp_path: Path) -> None:
    result = run(
        "plugin", "crypto_moonshot.strategy.capitulation:R2", "--name", "r2",
        "--symbols", "DIPUSDT,THINUSDT,SHALLOWUSDT", "--json", *common(root, tmp_path),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["strategy"] == "r2"
    assert doc["metrics"]["trades"] == 2
    assert doc["config"]["exit"]["take_profit"] == 0.2  # the rule's own plan
    trades = pl.read_parquet(tmp_path / "out" / "trades.parquet")
    assert sorted(trades["symbol"].to_list()) == ["DIPUSDT", "THINUSDT"]
    assert {p.name for p in (tmp_path / "out").iterdir()} == OUTPUTS


def test_plugin_json_prints_one_object(root: Path, tmp_path: Path) -> None:
    plugin = write_plugin(tmp_path)
    result = run("plugin", f"{plugin}:FLUSH", "--json", *common(root, tmp_path))
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["metrics"]["trades"] == 2
    assert doc["out"] == str(tmp_path / "out")


def test_plugin_rolling_range_options(root: Path, tmp_path: Path) -> None:
    plugin = write_plugin(tmp_path)
    start = at(SPIKE + 1).isoformat()
    result = run("plugin", f"{plugin}:FLUSH", "--start", start, *common(root, tmp_path))
    assert result.exit_code == 0, result.output
    trades = pl.read_parquet(tmp_path / "out" / "trades.parquet")
    assert trades["signal_time"].to_list() == [at(LONE)]


def test_plugin_errors(root: Path, tmp_path: Path) -> None:
    plugin = write_plugin(tmp_path)
    missing = run("plugin", f"{plugin}:Nope", *common(root, tmp_path))
    assert missing.exit_code == 1
    assert "has no attribute" in missing.output
    assert str(plugin) in missing.output
    bad_param = run("plugin", f"{plugin}:FLUSH", "--param", "oops", *common(root, tmp_path))
    assert bad_param.exit_code == 2
    assert "key=value" in bad_param.output
    no_exit = run(
        "plugin",
        "crypto_moonshot.strategy.capitulation:F2",
        "--hold",
        "none",
        *common(root, tmp_path),
    )
    assert no_exit.exit_code == 2
    assert "no exit plan" in no_exit.output


def write_signals(path: Path, *, tz: bool = True, drop: str | None = None) -> Path:
    frame = pl.DataFrame(
        {
            "symbol": ["DIPUSDT", "THINUSDT"],
            "open_time": [at(SPIKE), at(SPIKE)],
            "priority": [-0.15, -0.12],
            "score": [0.9, 0.8],
        },
        schema_overrides={"open_time": pl.Datetime("us", "UTC" if tz else None)},
    )
    if drop is not None:
        frame = frame.drop(drop)
    frame.write_parquet(path)
    return path


def test_signals_file_runs_end_to_end(root: Path, tmp_path: Path) -> None:
    file = write_signals(tmp_path / "sigs.parquet")
    result = run("signals", str(file), "--interval", "5m", "--hold", "1h", *common(root, tmp_path))
    assert result.exit_code == 0, result.output
    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} == OUTPUTS
    trades = pl.read_parquet(out / "trades.parquet")
    assert trades["symbol"].to_list() == ["DIPUSDT", "THINUSDT"]
    assert trades["score"].to_list() == [0.9, 0.8]
    doc = metrics(tmp_path)
    assert doc["signal_file"] == str(file)
    assert doc["signals"] == 2
    assert doc["config"]["exit"]["max_hold_seconds"] == 3600
    assert "mean bp per trade" in result.stdout

    as_json = run("signals", str(file), "--take-profit", "0.1", "--json", *common(root, tmp_path))
    assert as_json.exit_code == 0, as_json.output
    assert json.loads(as_json.stdout)["config"]["exit"]["take_profit"] == 0.1


def test_signals_file_errors(root: Path, tmp_path: Path) -> None:
    ok = write_signals(tmp_path / "ok.parquet")
    no_exit = run("signals", str(ok), *common(root, tmp_path))
    assert no_exit.exit_code == 2
    assert "no exit plan" in no_exit.output
    missing = run(
        "signals", str(tmp_path / "none.parquet"), "--hold", "1h", *common(root, tmp_path)
    )
    assert missing.exit_code == 2
    assert "not a file" in missing.output
    columns = write_signals(tmp_path / "cols.parquet", drop="priority")
    bad = run("signals", str(columns), "--hold", "1h", *common(root, tmp_path))
    assert bad.exit_code == 2
    assert "signals are missing columns" in bad.output
    assert "priority" in bad.output
    naive = write_signals(tmp_path / "naive.parquet", tz=False)
    result = run("signals", str(naive), "--hold", "1h", *common(root, tmp_path))
    assert result.exit_code == 2
    assert "timezone-aware" in result.output
