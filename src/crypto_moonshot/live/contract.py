"""The signal contract (v1): what a strategy delivers for each closed candle.

A `SignalBatch` is the one message every strategy, in process or external, sends per closed
candle of its interval, even when it has no signals (see `docs/architecture.md`, "Signal
contract"). `frame_to_batch` and `batch_to_frame` convert between a batch and the frame
`Strategy.on_candles` returns (`plugin.normalize_signal_frame`): extra columns become each
signal's `info` and back, losslessly for scalar values.

An entry's `exit` is a partial override of the strategy's exit plan: a field left out keeps the
strategy's value. Durations are strings such as `90m`, `2h`, `3d`, `1w` (`models.parse_duration`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Self

import polars as pl
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.live.plugin import (
    CANONICAL_COLUMNS,
    EXIT_COLUMNS,
    empty_signal_frame,
    normalize_signal_frame,
)
from crypto_moonshot.models import Interval, format_duration, parse_duration

CONTRACT_VERSION = 1

type Scalar = str | int | float | bool | None


class ExitPlanIn(BaseModel):
    """`ExitPlan` as it travels in a batch: every field optional, `max_hold` a duration string."""

    model_config = ConfigDict(extra="forbid")

    take_profit: float | None = None
    stop: float | None = None
    trailing: float | None = None
    max_hold: str | None = None
    tp_fill: Literal["touch", "close"] | None = None

    @model_validator(mode="after")
    def _valid(self) -> Self:
        if self.max_hold is not None and parse_duration(self.max_hold) is None:
            msg = f"max_hold must be a duration such as '90m', got {self.max_hold!r}"
            raise ValueError(msg)
        self.to_plan()  # the levels' ranges
        return self

    @classmethod
    def from_plan(cls, plan: ExitPlan) -> ExitPlanIn:
        return cls(
            take_profit=plan.take_profit,
            stop=plan.stop,
            trailing=plan.trailing,
            max_hold=None if plan.max_hold is None else format_duration(plan.max_hold),
            tp_fill=plan.tp_fill,
        )

    def to_plan(self, base: ExitPlan | None = None) -> ExitPlan:
        """`base` (an empty plan if None) with this plan's set fields replacing its own."""
        plan = base or ExitPlan()
        return ExitPlan(
            take_profit=plan.take_profit if self.take_profit is None else self.take_profit,
            stop=plan.stop if self.stop is None else self.stop,
            trailing=plan.trailing if self.trailing is None else self.trailing,
            max_hold=plan.max_hold if self.max_hold is None else parse_duration(self.max_hold),
            tp_fill=plan.tp_fill if self.tp_fill is None else self.tp_fill,
        )


class SignalIn(BaseModel):
    """One signal: an entry (or an exit of the strategy's position) in one symbol."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1)
    action: Literal["enter", "exit"] = "enter"
    priority: float = Field(default=0.0, allow_inf_nan=False)
    weight: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    exit: ExitPlanIn | None = None
    info: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class SignalBatch(BaseModel):
    """The signals of one strategy for the candle that opened at `candle_time`."""

    model_config = ConfigDict(extra="forbid")

    contract: int
    strategy: str = Field(min_length=1)
    interval: Interval
    candle_time: AwareDatetime
    signals: list[SignalIn] = Field(default_factory=list)

    @field_validator("contract")
    @classmethod
    def _known_version(cls, value: int) -> int:
        if value != CONTRACT_VERSION:
            msg = f"unknown contract version {value}: this build speaks {CONTRACT_VERSION}"
            raise ValueError(msg)
        return value

    @field_validator("candle_time")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not self.interval.is_aligned(self.candle_time):
            msg = (
                f"candle_time {self.candle_time.isoformat()} is not a {self.interval.value} "
                "candle open time"
            )
            raise ValueError(msg)
        seen: set[tuple[str, str]] = set()
        for signal in self.signals:
            key = (signal.symbol, signal.action)
            if key in seen:
                msg = f"more than one {signal.action} signal for {signal.symbol} in one batch"
                raise ValueError(msg)
            seen.add(key)
        return self


def _info_value(value: object, column: str) -> Scalar:
    if value is None or isinstance(value, str | bool | int | float):
        return value
    msg = f"extra column {column!r} holds {type(value).__name__} values: info takes scalars only"
    raise ValueError(msg)


def frame_to_batch(
    frame: pl.DataFrame, *, strategy: str, interval: Interval, candle_time: datetime
) -> SignalBatch:
    """The batch for an `on_candles` frame (normalized here; a bad frame raises
    `SignalFrameError`). Extra columns become `info`; exit-override columns become `exit`."""
    normalized = normalize_signal_frame(frame, candle_time)
    extras = [c for c in normalized.columns if c not in CANONICAL_COLUMNS]
    signals: list[SignalIn] = []
    for row in normalized.iter_rows(named=True):
        overrides = {k: row[k] for k in EXIT_COLUMNS if row[k] is not None}
        if "max_hold" in overrides:
            overrides["max_hold"] = format_duration(overrides["max_hold"])
        signals.append(
            SignalIn(
                symbol=row["symbol"],
                action=row["action"],
                priority=row["priority"],
                weight=row["weight"],
                exit=ExitPlanIn(**overrides) if overrides else None,
                info={c: _info_value(row[c], c) for c in extras},
            )
        )
    return SignalBatch(
        contract=CONTRACT_VERSION,
        strategy=strategy,
        interval=interval,
        candle_time=candle_time,
        signals=signals,
    )


def _info_series(name: str, values: list[Scalar]) -> pl.Series:
    kinds = {type(v) for v in values if v is not None}
    if kinds <= {int, float} and float in kinds:
        return pl.Series(name, values, dtype=pl.Float64)
    if len(kinds) > 1:
        names = ", ".join(sorted(k.__name__ for k in kinds))
        msg = f"info key {name!r} mixes value types ({names}) across signals"
        raise ValueError(msg)
    return pl.Series(name, values)


def batch_to_frame(batch: SignalBatch) -> pl.DataFrame:
    """The normalized frame of a batch (inverse of `frame_to_batch`). Info keys become columns
    (null where a signal lacks the key); a key must hold one kind of value (ints and floats mix
    into floats) across the batch's signals."""
    if not batch.signals:
        return empty_signal_frame()
    rows = batch.signals
    columns: dict[str, list[Scalar]] = {}
    for position, signal in enumerate(rows):
        for key in signal.info:
            columns.setdefault(key, [None] * len(rows))
        for key, value in signal.info.items():
            columns[key][position] = value
    holds = [None if s.exit is None else s.exit.max_hold for s in rows]
    base = pl.DataFrame(
        {
            "symbol": [s.symbol for s in rows],
            "action": [s.action for s in rows],
            "priority": [s.priority for s in rows],
            "weight": [s.weight for s in rows],
            **{
                name: [None if s.exit is None else getattr(s.exit, name) for s in rows]
                for name in ("take_profit", "stop", "trailing")
            },
            "max_hold": [None if h is None else parse_duration(h) for h in holds],
            "tp_fill": [None if s.exit is None else s.exit.tp_fill for s in rows],
        },
        schema=empty_signal_frame().schema,
    )
    extras = [_info_series(name, values) for name, values in columns.items()]
    return pl.concat([base, pl.DataFrame(extras)], how="horizontal") if extras else base
