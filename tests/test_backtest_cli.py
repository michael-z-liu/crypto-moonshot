import csv
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from strategy_fakes import LONE, SPIKE, at, write_capitulation_store
from typer.testing import CliRunner

from crypto_moonshot.backtest_cli import TpFill, build_exit, parse_hold
from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.cli import app
from crypto_moonshot.strategy.capitulation import R2, R2_LAST_ENTRY
from crypto_moonshot.strategy.scores import write_scores

OUTPUTS = {"trades.parquet", "trades.csv", "equity.csv", "yearly.csv", "metrics.json"}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """The synthetic capitulation store with its universe lists."""
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


def load_metrics(out: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((out / "metrics.json").read_text())
    return loaded


# ---------------------------------------------------------------- strategies


def test_f2_runs_and_writes_every_output(root: Path, tmp_path: Path) -> None:
    result = run("f2", *common(root, tmp_path))
    assert result.exit_code == 0, result.output
    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} == OUTPUTS
    trades = pl.read_parquet(out / "trades.parquet")
    assert trades["symbol"].to_list() == ["DIPUSDT"]
    assert trades["signal_time"].to_list() == [at(SPIKE)]
    assert trades["exit_time"][0] - trades["entry_time"][0] == timedelta(hours=1)
    with (out / "trades.csv").open() as f:
        assert [row["symbol"] for row in csv.DictReader(f)] == ["DIPUSDT"]
    assert pl.read_csv(out / "equity.csv").height > 0
    assert pl.read_csv(out / "yearly.csv")["year"].to_list() == [2024]
    doc = load_metrics(out)
    assert doc["strategy"] == "f2"
    assert doc["universe"] == "pool"
    assert doc["signals"] == 1
    assert doc["metrics"]["trades"] == 1
    assert doc["skipped"]["slots_full"] == 0
    config = doc["config"]
    assert config["sizing"] == {"type": "equal_slots", "slots": 5}
    assert config["max_positions"] == 5
    assert config["initial_capital"] == 10_000
    assert config["exit"]["max_hold_seconds"] == 3600
    assert config["exit"]["tp_fill"] == "touch"
    assert config["costs"] == {"fee": 0.001, "slippage": 0.0005, "latency": 0.0}
    assert config["data_end"].startswith("2024-01-")  # the store's last candle close
    assert config["until"] is None
    assert "YEAR" in result.stdout
    assert "2024" in result.stdout
    assert "mean bp per trade" in result.stdout
    assert "\x1b" not in result.stdout


def test_capitulation_defaults_to_the_all_universe_fixed_stake_and_last_entry(
    root: Path, tmp_path: Path
) -> None:
    result = run("capitulation", *common(root, tmp_path))
    assert result.exit_code == 0, result.output
    out = tmp_path / "out"
    doc = load_metrics(out)
    assert doc["universe"] == "all"
    config = doc["config"]
    assert config["sizing"] == {"type": "fixed_stake", "stake": 50.0}
    assert config["max_positions"] == 200
    assert config["end"] == R2_LAST_ENTRY.isoformat()
    assert config["exit"]["take_profit"] == R2.exit.take_profit
    assert config["exit"]["max_hold_seconds"] == timedelta(days=3).total_seconds()
    trades = pl.read_parquet(out / "trades.parquet")
    assert sorted(trades["symbol"].to_list()) == ["DIPUSDT", "THINUSDT"]  # THIN is outside the pool
    assert set(trades["stake"].to_list()) == {50.0}


def test_capitulation_fine_and_tp_fill_options(root: Path, tmp_path: Path) -> None:
    result = run("capitulation", *common(root, tmp_path, "--fine", "--tp-fill", "close"))
    assert result.exit_code == 0, result.output
    config = load_metrics(tmp_path / "out")["config"]
    assert config["use_fine"] is True
    assert config["fine_interval"] == "1m"
    assert config["exit"]["tp_fill"] == "close"


@pytest.fixture
def scores_file(tmp_path: Path) -> Path:
    path = tmp_path / "scores.parquet"
    frame = pl.DataFrame(
        {
            "symbol": ["DIPUSDT", "DIPUSDT", "THINUSDT"],
            "open_time": [at(SPIKE), at(LONE), at(SPIKE)],
            "score": [0.99, 0.5, 0.97],
        },
        schema_overrides={"open_time": pl.Datetime("us", "UTC")},
    )
    write_scores(frame, path)
    return path


def test_scores_runs_with_the_base_cnn_settings(
    root: Path, tmp_path: Path, scores_file: Path
) -> None:
    result = run("scores", *common(root, tmp_path, "--scores", str(scores_file)))
    assert result.exit_code == 0, result.output
    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} == OUTPUTS
    doc = load_metrics(out)
    assert doc["threshold"] == 0.98
    assert doc["universe"] is None
    assert doc["signals"] == 1
    assert doc["config"]["sizing"] == {"type": "equal_slots", "slots": 5}
    assert pl.read_parquet(out / "trades.parquet")["symbol"].to_list() == ["DIPUSDT"]


def test_scores_threshold_option(root: Path, tmp_path: Path, scores_file: Path) -> None:
    args = common(root, tmp_path, "--scores", str(scores_file), "--threshold", "0.9")
    assert run("scores", *args).exit_code == 0
    doc = load_metrics(tmp_path / "out")
    assert doc["threshold"] == 0.9
    assert doc["signals"] == 2


def test_scores_requires_the_scores_option(root: Path, tmp_path: Path) -> None:
    result = run("scores", *common(root, tmp_path))
    assert result.exit_code == 2
    assert "--scores" in result.output
    assert not (tmp_path / "out").exists()


def test_missing_scores_file_is_a_usage_error(root: Path, tmp_path: Path) -> None:
    args = common(root, tmp_path, "--scores", str(tmp_path / "nope.parquet"))
    assert run("scores", *args).exit_code == 2


def test_rule_strategies_reject_score_options(root: Path, tmp_path: Path) -> None:
    assert run("f2", *common(root, tmp_path, "--threshold", "0.5")).exit_code == 2
    assert run("capitulation", *common(root, tmp_path, "--scores", "x.parquet")).exit_code == 2


def test_scores_reject_a_universe(root: Path, tmp_path: Path, scores_file: Path) -> None:
    args = common(root, tmp_path, "--scores", str(scores_file), "--universe", "pool")
    assert run("scores", *args).exit_code == 2


# ---------------------------------------------------------------- options


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1h", timedelta(hours=1)),
        ("90m", timedelta(minutes=90)),
        ("3d", timedelta(days=3)),
        ("1.5h", timedelta(minutes=90)),
        ("2W", timedelta(weeks=2)),
        ("30s", timedelta(seconds=30)),
        ("none", None),
    ],
)
def test_parse_hold(text: str, expected: timedelta | None) -> None:
    assert parse_hold(text) == expected


@pytest.mark.parametrize("text", ["", "1", "h", "1y", "-1h", "0h", "1h30m"])
def test_parse_hold_rejects_bad_durations(text: str) -> None:
    with pytest.raises(Exception, match="duration"):
        parse_hold(text)


def test_exit_overrides_replace_preset_levels() -> None:
    base = ExitPlan(take_profit=0.2, stop=0.1, max_hold=timedelta(days=3))
    plan = build_exit(
        base, hold="6h", take_profit=0.0, stop=None, trailing=0.05, tp_fill=TpFill.CLOSE
    )
    assert plan == ExitPlan(stop=0.1, trailing=0.05, max_hold=timedelta(hours=6), tp_fill="close")
    assert (
        build_exit(base, hold=None, take_profit=None, stop=None, trailing=None, tp_fill=None)
        == base
    )


def test_exit_options_reach_the_config(root: Path, tmp_path: Path) -> None:
    args = common(
        root, tmp_path, "--hold", "2h", "--take-profit", "0.3", "--stop", "0.1",
        "--trailing", "0.05", "--fee", "0.002", "--slippage", "0.001", "--latency", "0.0001",
        "--entry-delay", "1", "--capital", "5000", "--max-positions", "3",
        "--start", "2024-01-01", "--end", "2025-01-01",
    )  # fmt: skip
    result = run("f2", *args)
    assert result.exit_code == 0, result.output
    config = load_metrics(tmp_path / "out")["config"]
    assert config["exit"] == {
        "take_profit": 0.3,
        "stop": 0.1,
        "trailing": 0.05,
        "max_hold_seconds": 7200.0,
        "tp_fill": "touch",
    }
    assert config["costs"] == {"fee": 0.002, "slippage": 0.001, "latency": 0.0001}
    assert (config["entry_delay"], config["initial_capital"], config["max_positions"]) == (
        1,
        5000,
        3,
    )
    assert config["start"] == "2024-01-01T00:00:00+00:00"
    assert config["end"] == "2025-01-01T00:00:00+00:00"
    trades = pl.read_parquet(tmp_path / "out" / "trades.parquet")
    assert trades["entry_time"][0] == at(SPIKE + 2)  # one extra candle


def test_bad_hold_is_a_usage_error(root: Path, tmp_path: Path) -> None:
    result = run("f2", *common(root, tmp_path, "--hold", "soon"))
    assert result.exit_code == 2
    assert "invalid duration" in result.output


def test_slots_and_stake_are_mutually_exclusive(root: Path, tmp_path: Path) -> None:
    result = run("f2", *common(root, tmp_path, "--slots", "3", "--stake", "50"))
    assert result.exit_code == 2
    assert "cannot be used together" in result.output


def test_slots_set_the_default_max_positions(root: Path, tmp_path: Path) -> None:
    assert run("f2", *common(root, tmp_path, "--slots", "8")).exit_code == 0
    config = load_metrics(tmp_path / "out")["config"]
    assert config["sizing"] == {"type": "equal_slots", "slots": 8}
    assert config["max_positions"] == 8


def test_stake_switches_to_fixed_sizing(root: Path, tmp_path: Path) -> None:
    assert run("f2", *common(root, tmp_path, "--stake", "100")).exit_code == 0
    out = tmp_path / "out"
    assert load_metrics(out)["config"]["sizing"] == {"type": "fixed_stake", "stake": 100.0}
    assert pl.read_parquet(out / "trades.parquet")["stake"].to_list() == [100.0]


def test_invalid_tp_fill_and_universe_are_usage_errors(root: Path, tmp_path: Path) -> None:
    assert run("f2", *common(root, tmp_path, "--tp-fill", "maybe")).exit_code == 2
    assert run("f2", *common(root, tmp_path, "--universe", "some")).exit_code == 2
    assert run("f2", *common(root, tmp_path, "--capital", "0")).exit_code == 2


def test_universe_option_overrides_the_preset(root: Path, tmp_path: Path) -> None:
    assert run("capitulation", *common(root, tmp_path, "--universe", "pool")).exit_code == 0
    trades = pl.read_parquet(tmp_path / "out" / "trades.parquet")
    assert trades["symbol"].to_list() == ["DIPUSDT"]


def test_until_hides_later_candles(root: Path, tmp_path: Path) -> None:
    until = at(SPIKE + 3).isoformat()
    result = run("f2", *common(root, tmp_path, "--until", until))
    assert result.exit_code == 0, result.output
    config = load_metrics(tmp_path / "out")["config"]
    assert config["data_end"] == config["until"] == until
    # The spike is visible, but the hour after it is not: the position closes at the data end.
    doc = load_metrics(tmp_path / "out")
    assert doc["metrics"]["trades"] == 1
    trades = pl.read_parquet(tmp_path / "out" / "trades.parquet")
    assert trades["exit_reason"].to_list() == ["series_end"]


# ---------------------------------------------------------------- output


def test_json_prints_one_object_on_stdout(root: Path, tmp_path: Path) -> None:
    result = run("f2", *common(root, tmp_path, "--json"))
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["out"] == str(tmp_path / "out")
    assert doc["metrics"]["trades"] == 1
    assert doc == load_metrics(tmp_path / "out")
    assert "YEAR" not in result.stdout
    assert "signals:" in result.stderr  # progress stays on stderr


def test_json_has_no_nan(root: Path, tmp_path: Path) -> None:
    # No signal in range: every undefined metric is null, and the JSON is strictly valid.
    result = run("f2", *common(root, tmp_path, "--json", "--start", "2030-01-01"))
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout, parse_constant=lambda c: pytest.fail(c))
    assert doc["metrics"]["trades"] == 0
    assert doc["metrics"]["mean_bp"] is None


def test_default_out_dir_is_results_strategy_timestamp(root: Path, tmp_path: Path) -> None:
    args = ["--data-dir", str(root), "--cache-dir", str(tmp_path / "cache"), "--workers", "1"]
    assert run("f2", *args).exit_code == 0
    [out] = (tmp_path / "results").iterdir()
    assert out.name.startswith("f2-")
    assert out.name.endswith("Z")
    assert {p.name for p in out.iterdir()} == OUTPUTS


def test_data_dir_defaults_to_the_environment(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_DATA_DIR", str(root))
    args = ["--cache-dir", str(tmp_path / "cache"), "--workers", "1", "--out", str(tmp_path / "o")]
    assert run("f2", *args).exit_code == 0
    assert (tmp_path / "o" / "metrics.json").exists()


def test_signal_caches_stay_out_of_the_data_dir(root: Path, tmp_path: Path) -> None:
    before = sorted(p.relative_to(root) for p in root.rglob("*"))
    assert run("f2", *common(root, tmp_path)).exit_code == 0
    assert sorted(p.relative_to(root) for p in root.rglob("*")) == before
    assert list((tmp_path / "cache" / "signals").glob("f2-*.parquet"))


def test_empty_store_is_a_clean_error(tmp_path: Path) -> None:
    result = run("f2", *common(tmp_path / "empty", tmp_path))
    assert result.exit_code == 1
    assert "no candles stored" in result.output


def test_missing_universe_list_is_a_clean_error(root: Path, tmp_path: Path) -> None:
    (root / "top200.txt").unlink()
    result = run("f2", *common(root, tmp_path))
    assert result.exit_code == 1
    assert "top200.txt" in result.output


# ---------------------------------------------------------------- convert-scores


def test_convert_scores_writes_a_score_file(tmp_path: Path) -> None:
    meta = tmp_path / "meta.parquet"
    pl.DataFrame(
        {
            "symbol": ["AUSDT", "AUSDT", "BUSDT"],
            "open_time": [at(1), at(2), at(3)],
            "event": [10, 11, 12],
        },
        schema_overrides={"open_time": pl.Datetime("us", "UTC")},
    ).write_parquet(meta)
    npz = tmp_path / "wf.npz"
    np.savez(npz, pct=np.array([0.99, np.nan, 0.5]), raw=np.array([1.0, 2.0, 3.0]))
    out = tmp_path / "scores" / "base.parquet"
    result = run("convert-scores", str(npz), str(meta), str(out))
    assert result.exit_code == 0, result.output
    assert "wrote 2 scores" in result.stdout
    scores = pl.read_parquet(out)
    assert scores["symbol"].to_list() == ["AUSDT", "BUSDT"]
    assert scores["score"].to_list() == [0.99, 0.5]
    assert {"event", "raw"} <= set(scores.columns)


def test_convert_scores_errors_are_usage_errors(tmp_path: Path) -> None:
    missing = run(
        "convert-scores", str(tmp_path / "a.npz"), str(tmp_path / "m"), str(tmp_path / "o")
    )
    assert missing.exit_code == 2
    meta = tmp_path / "meta.parquet"
    pl.DataFrame({"symbol": ["A"], "open_time": [at(1)]}).write_parquet(meta)
    npz = tmp_path / "wf.npz"
    np.savez(npz, pct=np.array([0.1, 0.2]))
    result = run("convert-scores", str(npz), str(meta), str(tmp_path / "o.parquet"))
    assert result.exit_code == 2
    assert "expected" in result.output
    result = run("convert-scores", str(npz), str(meta), str(tmp_path / "o.parquet"), "--key", "zz")
    assert result.exit_code == 2
    assert "zz" in result.output


def test_backtest_without_a_command_shows_help() -> None:
    result = run()
    assert "f2" in result.output
    assert "convert-scores" in result.output
