"""Plugin loading, the output-frame normalizer and the rule adapter."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from plugin_fakes import at, write_plugin

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.live.plugin import (
    PluginError,
    RuleStrategy,
    SignalFrameError,
    Strategy,
    default_warmup,
    empty_signal_frame,
    load_strategy,
    normalize_signal_frame,
)
from crypto_moonshot.models import Interval
from crypto_moonshot.strategy.capitulation import F2, R2

# ---------------------------------------------------------------- loading


def test_loads_an_instance_from_a_file_path(tmp_path: Path) -> None:
    plugin = write_plugin(tmp_path)
    before = list(sys.path)
    strategy = load_strategy(f"{plugin}:FLUSH", name="mine")
    assert isinstance(strategy, Strategy)
    assert strategy.name == "mine"
    assert strategy.warmup == 60
    assert strategy.exit == ExitPlan(max_hold=timedelta(hours=1))
    assert sys.path == before
    assert str(tmp_path) not in sys.path


def test_the_module_instance_is_not_renamed(tmp_path: Path) -> None:
    plugin = write_plugin(tmp_path)
    a = load_strategy(f"{plugin}:FLUSH", name="a")
    b = load_strategy(f"{plugin}:FLUSH", name="b")
    assert (a.name, b.name) == ("a", "b")


def test_loads_a_class_and_a_factory_with_params(tmp_path: Path) -> None:
    plugin = write_plugin(tmp_path)
    cls = load_strategy(f"{plugin}:Flush", name="c", params={"threshold": -0.3})
    assert (cls.name, getattr(cls, "threshold", None)) == ("c", -0.3)
    factory = load_strategy(f"{plugin}:make", name="f", params={"threshold": -0.2})
    assert (factory.name, getattr(factory, "threshold", None)) == ("f", -0.2)
    bare = load_strategy(f"{plugin}:make", name="g")
    assert getattr(bare, "threshold", None) == -0.10


def test_loads_a_rule_by_module_path_as_a_rule_strategy() -> None:
    strategy = load_strategy("crypto_moonshot.strategy.capitulation:F2", name="f2-live")
    assert isinstance(strategy, RuleStrategy)
    assert strategy.name == "f2-live"
    assert strategy.rule is F2
    assert strategy.exit == F2.exit
    assert strategy.context_symbols == ("BTCUSDT",)
    assert strategy.warmup == default_warmup(Interval.M5) == 2304
    r2 = load_strategy("crypto_moonshot.strategy.capitulation:R2", name="r2")
    assert isinstance(r2, RuleStrategy)
    assert r2.rule is R2


@pytest.mark.parametrize(
    ("spec", "needle"),
    [
        ("no-colon-here", "expected package.module:attr"),
        ("pkg.mod:", "expected package.module:attr"),
        ("bad name:attr", "not a module name"),
        ("crypto_moonshot.nothing_here:X", "cannot import module"),
        ("crypto_moonshot.strategy.capitulation:Nope", "has no attribute"),
        ("{file}x:FLUSH", "not found"),
        ("{file}:Missing", "has no attribute"),
        ("{file}:CONSTANT", "neither callable nor"),
        ("{file}:make", "calling make failed"),  # wrong params, see below
    ],
)
def test_load_errors_name_the_source(tmp_path: Path, spec: str, needle: str) -> None:
    plugin = write_plugin(tmp_path)
    source = spec.replace("{file}", str(plugin))
    params = {"nope": 1} if spec.endswith(":make") else None
    with pytest.raises(PluginError, match=needle) as info:
        load_strategy(source, name="x", params=params)
    assert source.strip() in str(info.value)


def test_params_for_an_instance_are_an_error(tmp_path: Path) -> None:
    plugin = write_plugin(tmp_path)
    with pytest.raises(PluginError, match="params given"):
        load_strategy(f"{plugin}:FLUSH", name="x", params={"threshold": 1})
    with pytest.raises(PluginError, match="params given"):
        load_strategy("crypto_moonshot.strategy.capitulation:F2", name="x", params={"a": 1})


def test_a_file_that_fails_to_import(tmp_path: Path) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('boom')\n")
    with pytest.raises(PluginError, match="boom"):
        load_strategy(f"{broken}:X", name="x")


def test_an_object_that_is_not_a_strategy(tmp_path: Path) -> None:
    thing = tmp_path / "thing.py"
    thing.write_text(
        "class Half:\n    name = 'h'\n    warmup = 0\n    def on_candles(self, w, t, c):\n"
        "        return None\n\nHALF = Half()\nSTRATEGY_LIKE = Half\n"
    )
    with pytest.raises(PluginError) as info:
        load_strategy(f"{thing}:HALF", name="h")
    message = str(info.value)
    assert "not a strategy" in message
    assert "interval" in message
    assert "warmup" in message
    assert "context_symbols" in message
    assert "exit is missing" in message
    with pytest.raises(PluginError, match="not a strategy"):
        load_strategy(f"{thing}:Half", name="h")


def test_a_rule_with_an_interval_without_a_default_warmup() -> None:
    from dataclasses import replace

    weekly = replace(F2, interval=Interval.W1)
    with pytest.raises(ValueError, match="pass warmup"):
        RuleStrategy(weekly)
    assert RuleStrategy(weekly, warmup=10).warmup == 10


# ---------------------------------------------------------------- normalizer

T = at(100)


def frame(**columns: list[object]) -> pl.DataFrame:
    return pl.DataFrame(columns)


def test_the_normalizer_fills_defaults_and_keeps_extras() -> None:
    out = normalize_signal_frame(
        frame(symbol=["AUSDT", "BUSDT"], priority=[1, 2], score=[0.5, 0.6]), T
    )
    assert out.columns == [*empty_signal_frame().columns, "score"]
    assert out["action"].to_list() == ["enter", "enter"]
    assert out.schema["priority"] == pl.Float64
    assert out["weight"].null_count() == 2


def test_the_normalizer_accepts_no_signals_and_an_open_time_column() -> None:
    assert normalize_signal_frame(pl.DataFrame(), T).height == 0
    out = normalize_signal_frame(
        frame(symbol=["A"], open_time=[T], priority=[0.0]).with_columns(
            pl.col("open_time").dt.replace_time_zone("UTC")
        ),
        T,
    )
    assert "open_time" not in out.columns


def test_the_normalizer_parses_exit_overrides() -> None:
    out = normalize_signal_frame(
        frame(
            symbol=["A", "B"],
            priority=[0.0, 1.0],
            take_profit=[0.1, None],
            max_hold=["90m", None],
            tp_fill=["close", None],
        ),
        T,
    )
    assert out["max_hold"].to_list() == [timedelta(minutes=90), None]
    assert out["take_profit"].to_list() == [0.1, None]
    durations = normalize_signal_frame(
        frame(symbol=["A"], priority=[0.0], max_hold=[timedelta(hours=2)]), T
    )
    assert durations["max_hold"].to_list() == [timedelta(hours=2)]


@pytest.mark.parametrize(
    ("columns", "needle"),
    [
        ({"symbol": ["A"]}, "missing required columns: priority"),
        ({"priority": [1.0]}, "missing required columns: symbol"),
        ({"symbol": [1], "priority": [1.0]}, "symbol must be a string"),
        ({"symbol": ["A"], "priority": ["x"]}, "priority must be numeric"),
        ({"symbol": ["A"], "priority": [None]}, "priority must be a number"),
        ({"symbol": ["A"], "priority": [float("nan")]}, "priority must be a number"),
        ({"symbol": ["A"], "priority": [1.0], "action": ["buy"]}, "unknown signal action"),
        ({"symbol": ["A"], "priority": [1.0], "weight": [0.0]}, "weight must be"),
        ({"symbol": ["A"], "priority": [1.0], "weight": [-1.0]}, "weight must be"),
        ({"symbol": ["A"], "priority": [1.0], "take_profit": [-0.1]}, "invalid exit override"),
        ({"symbol": ["A"], "priority": [1.0], "stop": [1.5]}, "invalid exit override"),
        ({"symbol": ["A"], "priority": [1.0], "max_hold": ["soon"]}, "invalid duration"),
        ({"symbol": ["A"], "priority": [1.0], "tp_fill": ["maybe"]}, "tp_fill"),
        ({"symbol": ["A", "A"], "priority": [1.0, 2.0]}, "more than one enter signal for A"),
        (
            {"symbol": ["A", "A"], "priority": [1.0, 2.0], "action": ["exit", "exit"]},
            "more than one exit signal for A",
        ),
    ],
)
def test_the_normalizer_rejects(columns: dict[str, list[object]], needle: str) -> None:
    with pytest.raises(SignalFrameError, match=needle):
        normalize_signal_frame(pl.DataFrame(columns), T)


def test_an_enter_and_an_exit_for_one_symbol_are_allowed() -> None:
    out = normalize_signal_frame(
        frame(symbol=["A", "A"], priority=[1.0, 2.0], action=["enter", "exit"]), T
    )
    assert out.height == 2


def test_the_normalizer_rejects_signals_for_another_candle() -> None:
    other = frame(symbol=["A"], open_time=[at(99)], priority=[0.0]).with_columns(
        pl.col("open_time").dt.replace_time_zone("UTC")
    )
    with pytest.raises(SignalFrameError, match="only be for the candle"):
        normalize_signal_frame(other, T)
    naive = frame(symbol=["A"], open_time=[at(100).replace(tzinfo=None)], priority=[0.0])
    with pytest.raises(SignalFrameError, match="timezone-aware"):
        normalize_signal_frame(naive, T)
