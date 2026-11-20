"""The signal batch contract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
import pytest
from plugin_fakes import at
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.live.contract import (
    CONTRACT_VERSION,
    ExitPlanIn,
    SignalBatch,
    SignalIn,
    batch_to_frame,
    frame_to_batch,
)
from crypto_moonshot.live.plugin import SignalFrameError, normalize_signal_frame
from crypto_moonshot.models import Interval, format_duration, parse_duration

T = at(100)


def batch_dict(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "contract": 1,
        "strategy": "cnn",
        "interval": "5m",
        "candle_time": "2026-11-20T13:40:00Z",
        "signals": [{"symbol": "SOLUSDT", "priority": -0.9}],
    }
    return {**base, **changes}


def test_the_documented_example_parses() -> None:
    batch = SignalBatch.model_validate(
        {
            "contract": 1,
            "strategy": "cnn",
            "interval": "5m",
            "candle_time": "2026-11-20T13:40:00Z",
            "signals": [
                {
                    "symbol": "SOLUSDT",
                    "action": "enter",
                    "priority": -0.991,
                    "weight": None,
                    "exit": {"take_profit": 0.05, "max_hold": "2h"},
                    "info": {"score": 0.991},
                }
            ],
        }
    )
    assert batch.candle_time == datetime(2026, 11, 20, 13, 40, tzinfo=UTC)
    signal = batch.signals[0]
    assert signal.exit is not None
    assert signal.exit.to_plan() == ExitPlan(take_profit=0.05, max_hold=timedelta(hours=2))
    assert CONTRACT_VERSION == 1
    assert SignalBatch.model_validate_json(batch.model_dump_json()) == batch


def test_defaults_and_offset_times_become_utc() -> None:
    batch = SignalBatch.model_validate(batch_dict(candle_time="2026-11-20T14:40:00+01:00"))
    assert batch.candle_time == datetime(2026, 11, 20, 13, 40, tzinfo=UTC)
    assert batch.candle_time.utcoffset() == timedelta(0)
    signal = batch.signals[0]
    assert (signal.action, signal.weight, signal.exit, signal.info) == ("enter", None, None, {})
    assert SignalIn(symbol="A").priority == 0


@pytest.mark.parametrize(
    ("changes", "needle"),
    [
        ({"contract": 2}, "unknown contract version 2"),
        ({"contract": 0}, "unknown contract version"),
        ({"candle_time": "2026-11-20T13:40:00"}, "timezone"),
        ({"candle_time": "2026-11-20T13:42:00Z"}, "not a 5m candle open time"),
        ({"candle_time": "2026-11-20T13:40:30Z"}, "not a 5m candle open time"),
        ({"interval": "7m"}, "interval"),
        ({"strategy": ""}, "strategy"),
        ({"signals": [{"symbol": "A", "action": "buy"}]}, "action"),
        ({"signals": [{"symbol": ""}]}, "symbol"),
        ({"signals": [{"symbol": "A", "weight": 0}]}, "weight"),
        ({"signals": [{"symbol": "A", "weight": -1}]}, "weight"),
        ({"signals": [{"symbol": "A", "exit": {"max_hold": "soon"}}]}, "invalid duration"),
        ({"signals": [{"symbol": "A", "exit": {"max_hold": "none"}}]}, "max_hold"),
        ({"signals": [{"symbol": "A", "exit": {"stop": 2}}]}, "stop must be between"),
        ({"signals": [{"symbol": "A", "exit": {"tp_fill": "x"}}]}, "tp_fill"),
        ({"signals": [{"symbol": "A", "exit": {"hold": "1h"}}]}, "hold"),
        ({"signals": [{"symbol": "A", "info": {"x": [1]}}]}, "info"),
        ({"signals": [{"symbol": "A", "extra": 1}]}, "extra"),
        (
            {"signals": [{"symbol": "A"}, {"symbol": "A", "priority": 1}]},
            "more than one enter signal for A",
        ),
        (
            {"signals": [{"symbol": "A", "action": "exit"}, {"symbol": "A", "action": "exit"}]},
            "more than one exit signal for A",
        ),
    ],
)
def test_batches_that_break_the_contract_are_rejected(changes: dict[str, Any], needle: str) -> None:
    with pytest.raises(ValidationError, match=needle):
        SignalBatch.model_validate(batch_dict(**changes))


def test_naive_python_datetimes_and_monthly_alignment() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        SignalBatch(
            contract=1,
            strategy="s",
            interval=Interval.M5,
            candle_time=datetime(2026, 1, 1, 0, 0),  # noqa: DTZ001
        )
    SignalBatch(
        contract=1,
        strategy="s",
        interval=Interval.W1,
        candle_time=datetime(2026, 11, 16, tzinfo=UTC),
    )
    with pytest.raises(ValidationError, match="candle open time"):
        SignalBatch(
            contract=1,
            strategy="s",
            interval=Interval.W1,
            candle_time=datetime(2026, 11, 17, tzinfo=UTC),
        )


def test_an_enter_and_an_exit_for_one_symbol_are_allowed() -> None:
    batch = SignalBatch.model_validate(
        batch_dict(signals=[{"symbol": "A"}, {"symbol": "A", "action": "exit"}])
    )
    assert len(batch.signals) == 2


def test_an_empty_batch_is_valid() -> None:
    batch = SignalBatch.model_validate(batch_dict(signals=[]))
    assert batch_to_frame(batch).height == 0


def test_exit_plan_overrides_apply_over_a_base_plan() -> None:
    base = ExitPlan(take_profit=0.2, stop=0.1, max_hold=timedelta(days=3))
    override = ExitPlanIn(stop=0.05, max_hold="2h")
    assert override.to_plan(base) == ExitPlan(
        take_profit=0.2, stop=0.05, max_hold=timedelta(hours=2)
    )
    assert override.to_plan() == ExitPlan(stop=0.05, max_hold=timedelta(hours=2))
    plan = ExitPlan(take_profit=0.1, trailing=0.02, max_hold=timedelta(minutes=90), tp_fill="close")
    assert ExitPlanIn.from_plan(plan).to_plan() == plan
    assert ExitPlanIn.from_plan(plan).max_hold == "90m"


@pytest.mark.parametrize(
    ("text", "value"),
    [("90m", timedelta(minutes=90)), ("2h", timedelta(hours=2)), ("3d", timedelta(days=3))],
)
def test_durations_round_trip(text: str, value: timedelta) -> None:
    assert parse_duration(text) == value
    assert format_duration(value) == text
    assert format_duration(timedelta(weeks=1)) == "1w"
    assert format_duration(timedelta(seconds=90)) == "90s"


# ---------------------------------------------------------------- frames


def sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["SOLUSDT", "ADAUSDT", "XRPUSDT"],
            "priority": [-0.9, -0.5, 0.1],
            "action": ["enter", "enter", "exit"],
            "weight": [1.5, None, None],
            "take_profit": [0.05, None, None],
            "stop": [None, 0.02, None],
            "max_hold": ["2h", None, "45m"],
            "score": [0.9, 0.5, None],
            "rank": [1, 2, 3],
            "tag": ["a", None, "c"],
            "flag": [True, False, None],
        }
    )


def test_frame_to_batch_and_back_is_lossless() -> None:
    frame = normalize_signal_frame(sample_frame(), T)
    batch = frame_to_batch(frame, strategy="s", interval=Interval.M5, candle_time=T)
    assert batch.candle_time == T
    first = batch.signals[0]
    assert first.exit == ExitPlanIn(take_profit=0.05, max_hold="2h")
    assert first.info == {"score": 0.9, "rank": 1, "tag": "a", "flag": True}
    assert batch.signals[1].exit == ExitPlanIn(stop=0.02)
    assert batch.signals[2].action == "exit"
    assert_frame_equal(batch_to_frame(batch), frame)
    # and through JSON
    again = SignalBatch.model_validate_json(batch.model_dump_json())
    assert_frame_equal(batch_to_frame(again), frame)


def test_frame_to_batch_normalizes_and_validates() -> None:
    batch = frame_to_batch(
        pl.DataFrame({"symbol": ["A"], "priority": [1]}),
        strategy="s",
        interval=Interval.M5,
        candle_time=T,
    )
    assert batch.signals[0].priority == 1.0
    with pytest.raises(SignalFrameError, match="missing required"):
        frame_to_batch(
            pl.DataFrame({"symbol": ["A"]}), strategy="s", interval=Interval.M5, candle_time=T
        )
    with pytest.raises(ValidationError, match="candle open time"):
        frame_to_batch(
            pl.DataFrame({"symbol": ["A"], "priority": [1.0]}),
            strategy="s",
            interval=Interval.M5,
            candle_time=T + timedelta(minutes=1),
        )
    with pytest.raises(ValueError, match="scalars only"):
        frame_to_batch(
            pl.DataFrame({"symbol": ["A"], "priority": [1.0], "bad": [[1, 2]]}),
            strategy="s",
            interval=Interval.M5,
            candle_time=T,
        )


def test_info_keys_missing_from_some_signals_become_nulls_and_ints_widen() -> None:
    batch = SignalBatch.model_validate(
        batch_dict(
            signals=[
                {"symbol": "A", "info": {"n": 1}},
                {"symbol": "B", "info": {"n": 2.5, "m": "x"}},
            ]
        )
    )
    frame = batch_to_frame(batch)
    assert frame["n"].to_list() == [1.0, 2.5]
    assert frame["m"].to_list() == [None, "x"]
    mixed = SignalBatch.model_validate(
        batch_dict(signals=[{"symbol": "A", "info": {"n": 1}}, {"symbol": "B", "info": {"n": "x"}}])
    )
    with pytest.raises(ValueError, match="mixes value types"):
        batch_to_frame(mixed)
