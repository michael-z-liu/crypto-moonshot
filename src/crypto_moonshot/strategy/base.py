"""The strategy interface: entry signals from closed candles, plus an exit plan.

A strategy decides *when to enter*; the exits are orders at the broker. It produces:

- **signals**: a polars frame with `symbol`, `open_time` (the signal candle *t*, whose close the
  signal uses; entry is at the open of *t+1*), `priority` (lower first when signals compete for
  a slot in one candle), plus any extra columns, carried into the trade list;
- an **`ExitPlan`** (take profit, stop, trailing stop, time exit) the broker attaches to every
  entry as a bracket.

Rule strategies (`SignalRule`) compute their signals from candles in three vectorised,
backward-looking steps, so a universe can be computed in parallel:

1. `candidates(candles)`: one symbol's whole series in, the rows it might trade out, with the
   features the rule needs (per symbol, run in a process pool);
2. `context(series)`: cross-symbol context by open time from `context_symbols` (e.g. BTC);
3. `select(candidates)`: candidates of every symbol joined with the context in, signals out.

`strategy/compute.py` runs them over a universe. Model-score strategies (`strategy/scores.py`)
read precomputed per-candle scores instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import polars as pl

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.models import Interval

SIGNAL_COLUMNS = ("symbol", "open_time", "priority")


@dataclass(frozen=True, slots=True)
class Signals:
    """A strategy's output: entry signals (see the module docstring), the exit plan for every
    entry and the interval of the signal candles."""

    frame: pl.DataFrame
    exit: ExitPlan
    interval: Interval = Interval.M5

    def __post_init__(self) -> None:
        missing = [c for c in SIGNAL_COLUMNS if c not in self.frame.columns]
        if missing:
            msg = f"signals are missing columns: {', '.join(missing)}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return self.frame.height


class SignalRule(Protocol):
    """A rule strategy over candles of one interval (see the module docstring).

    `key` identifies everything that changes the signals, for caching: equal keys must give
    equal signals on equal data.
    """

    @property
    def name(self) -> str: ...

    @property
    def interval(self) -> Interval: ...

    @property
    def exit(self) -> ExitPlan: ...

    @property
    def context_symbols(self) -> tuple[str, ...]: ...

    def key(self) -> dict[str, object]: ...

    def candidates(self, candles: pl.DataFrame) -> pl.DataFrame:
        """Rows of one symbol's whole series (`KLINE_SCHEMA`, sorted) that may become signals,
        with `open_time` and feature columns; only candles up to each row may be used."""
        ...

    def context(self, series: Mapping[str, pl.DataFrame]) -> pl.DataFrame | None:
        """Context columns by `open_time` from the `context_symbols` series, joined onto the
        candidates by open time (`None` for no context)."""
        ...

    def select(self, candidates: pl.DataFrame) -> pl.DataFrame:
        """Signals (`symbol`, `open_time`, `priority`, extra columns) from every symbol's
        candidates joined with the context."""
        ...
