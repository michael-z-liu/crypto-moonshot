"""Rolling windows give the signals of the full-history computation."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from plugin_fakes import at, write_walk_store
from polars.testing import assert_frame_equal

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.live.plugin import (
    RuleStrategy,
    default_warmup,
    rolling_signals,
)
from crypto_moonshot.models import Interval
from crypto_moonshot.strategy.capitulation import CapitulationRule
from crypto_moonshot.strategy.compute import compute_signals

RULE = CapitulationRule(
    "test",
    min_qv_base=1.0,
    exit=ExitPlan(max_hold=timedelta(hours=1)),
    max_ret_4h=-0.05,
    min_btc_vol_mult=3.0,
    spike_mult=4.0,
)
START, END = at(2304), at(2800)


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("walk")
    write_walk_store(path)
    return path


def series_of(root: Path, symbols: list[str]) -> dict[str, pl.DataFrame]:
    store = KlineStore(root)
    return {s: store.read_series(s, Interval.M5) for s in symbols}


def stored_before(frame: pl.DataFrame, series: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Add `_n`: the stored candles of the signal's symbol up to and including its candle."""
    ranks = pl.concat(
        [
            f.select(
                pl.lit(s).alias("symbol"),
                "open_time",
                (pl.int_range(pl.len()) + 1).alias("_n"),
            )
            for s, f in series.items()
        ]
    )
    return frame.join(ranks, on=["symbol", "open_time"], how="left")


def test_the_default_warmup_is_the_minimum_history() -> None:
    assert default_warmup(Interval.M5) == 8 * 288


def test_rolling_signals_equal_the_full_history_signals(root: Path) -> None:
    coins = ["AAAUSDT", "BBBUSDT", "LATEUSDT"]
    series = series_of(root, coins)
    context = series_of(root, ["BTCUSDT"])
    strategy = RuleStrategy(RULE)

    rolled = rolling_signals(strategy, series, context, START, END)
    full = compute_signals(RULE, coins, root, workers=1).signals

    window = pl.col("open_time").is_between(START, END, closed="left")
    expected = stored_before(full.frame.filter(window), series)
    # The documented edge case: a symbol's first MIN_HISTORY stored candles.
    edge = expected.filter(pl.col("_n") < default_warmup(Interval.M5))
    expected = expected.filter(pl.col("_n") >= default_warmup(Interval.M5)).drop("_n")

    assert expected.height >= 4, "the fixture should produce several signals"
    assert set(expected["symbol"]) >= {"AAAUSDT", "BBBUSDT"}
    assert edge.height >= 1, "the fixture should hit the edge case"
    assert edge["symbol"].unique().to_list() == ["LATEUSDT"]
    assert_frame_equal(rolled.frame, expected, abs_tol=1e-9, rel_tol=1e-9)
    assert rolled.exit == RULE.exit
    assert rolled.interval is Interval.M5
    # The edge case: the full history signals where a window cannot know the series' length.
    assert not set(edge.select("symbol", "open_time").rows()) & set(
        rolled.frame.select("symbol", "open_time").rows()
    )


def test_backtest_signals_is_compute_signals(root: Path) -> None:
    coins = ["AAAUSDT", "BBBUSDT", "LATEUSDT"]
    strategy = RuleStrategy(RULE, workers=1)
    batch = strategy.backtest_signals(root, coins, None)
    assert_frame_equal(batch.frame, compute_signals(RULE, coins, root, workers=1).signals.frame)


def test_rolling_signals_uses_windows_of_warmup_candles(root: Path) -> None:
    seen: list[tuple[int, str]] = []

    class Probe:
        name = "probe"
        interval = Interval.M5
        warmup = 5
        context_symbols = ("BTCUSDT",)
        exit: ExitPlan | None = None

        def on_candles(
            self,
            windows: dict[str, pl.DataFrame],
            time: object,
            context: dict[str, pl.DataFrame],
        ) -> pl.DataFrame:
            for symbol, w in windows.items():
                assert w.height <= 5
                assert w["open_time"][-1] <= time  # type: ignore[operator]
                seen.append((w.height, symbol))
            assert set(context) == {"BTCUSDT"}
            return pl.DataFrame({"symbol": ["AAAUSDT"], "priority": [1.0], "x": ["y"]})

    series = series_of(root, ["AAAUSDT", "BBBUSDT"])
    context = series_of(root, ["BTCUSDT"])
    signals = rolling_signals(Probe(), series, context, at(2400), at(2410))  # type: ignore[arg-type]
    assert signals.frame.columns == ["symbol", "open_time", "priority", "x"]
    assert signals.frame.height == 10
    assert signals.exit == ExitPlan()
    assert {h for h, _ in seen} == {5}


def test_rolling_signals_keeps_only_entries_and_non_empty_overrides(root: Path) -> None:
    class Mixed:
        name = "mixed"
        interval = Interval.M5
        warmup = 3
        context_symbols = ()
        exit: ExitPlan | None = ExitPlan(stop=0.1)

        def on_candles(
            self, windows: dict[str, pl.DataFrame], time: object, context: dict[str, pl.DataFrame]
        ) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "symbol": ["AAAUSDT", "BBBUSDT"],
                    "action": ["enter", "exit"],
                    "priority": [0.5, 0.0],
                    "weight": [2.0, None],
                }
            )

    series = series_of(root, ["AAAUSDT", "BBBUSDT"])
    signals = rolling_signals(Mixed(), series, {}, at(2400), at(2403))  # type: ignore[arg-type]
    assert signals.frame.columns == ["symbol", "open_time", "priority", "weight"]
    assert signals.frame["symbol"].unique().to_list() == ["AAAUSDT"]
    assert signals.exit == ExitPlan(stop=0.1)
    nothing = rolling_signals(Mixed(), series, {}, at(5000), at(5003))  # type: ignore[arg-type]
    assert nothing.frame.columns == ["symbol", "open_time", "priority"]
    assert nothing.frame.height == 0
