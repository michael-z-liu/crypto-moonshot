"""Replay = backtest on the same signals, and each documented divergence on its own."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from engine_fakes import Scheduled, flat_bars, write_bars
from polars.testing import assert_frame_equal
from replay_fakes import (
    COINS,
    END,
    H1,
    HOUR,
    RULE,
    START,
    N,
    at,
    coin_paths,
    rule_strategy,
    run_config,
    strategy_dict,
    trades_core,
    write_replay_store,
)

from crypto_moonshot.backtest.engine import BacktestConfig, run_backtest
from crypto_moonshot.backtest.source import ParquetCandleSource
from crypto_moonshot.broker.base import Costs, ExitPlan, Side, client_order_id
from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.live.plugin import rolling_signals
from crypto_moonshot.live.replay import ReplayFeed, run_replay
from crypto_moonshot.live.state import PositionStatus, StateStore
from crypto_moonshot.portfolio import EqualSlots, FixedStake
from crypto_moonshot.strategy.base import Signals
from crypto_moonshot.strategy.compute import compute_signals


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("replay")
    write_replay_store(path)
    return path


@pytest.fixture(scope="module")
def signals(root: Path) -> Signals:
    """The rolling (live) signals of the rule over the replay range."""
    store = KlineStore(root)
    series = {s: store.read_series(s, H1) for s in COINS}
    context = {"BTCUSDT": store.read_series("BTCUSDT", H1)}
    return rolling_signals(rule_strategy(), series, context, at(START), at(END))


def backtest(
    root: Path, signals: pl.DataFrame, sizing: FixedStake | EqualSlots, capital: float, **kw: object
) -> pl.DataFrame:
    config = BacktestConfig(
        exit=RULE.exit,
        interval=H1,
        sizing=sizing,
        initial_capital=capital,
        **kw,  # type: ignore[arg-type]
    )
    return run_backtest(signals, ParquetCandleSource(root), config).trades


def test_the_fixture_cannot_show_a_documented_divergence(root: Path, signals: Signals) -> None:
    store = KlineStore(root)
    for symbol in COINS:
        frame = store.read_series(symbol, H1)
        # No holes and every candle opens at the previous close: no exit can gap through a
        # level at a candle's open, and every entry candle exists.
        assert frame.height == N
        assert (frame["open"][1:] == frame["close"][:-1]).all()
    # Rolling signals equal the full-history ones (no listing edge case in the range).
    full = compute_signals(RULE, list(COINS), root, workers=1).signals.frame
    full = full.filter(pl.col("open_time").is_between(at(START), at(END), closed="left"))
    assert_frame_equal(full.select(signals.frame.columns), signals.frame)
    # Many signals compete for few slots: several per candle.
    per_candle = signals.frame.group_by("open_time").len()
    assert len(signals) > 60
    assert int(per_candle["len"].max()) >= 5  # type: ignore[arg-type]


def test_replay_equals_backtest_with_fixed_stakes(
    root: Path, signals: Signals, tmp_path: Path
) -> None:
    config = run_config(tmp_path / "state.sqlite", [strategy_dict(capital=350.0)])
    result = run_replay(
        config, at(START), at(END), root, tmp_path / "out", plugins={"cap": rule_strategy()}
    )
    expected = backtest(root, signals.frame, FixedStake(100.0), 350.0)
    assert result.open_positions == 0
    assert_frame_equal(trades_core(result.trades), trades_core(expected))
    reasons = set(result.trades["exit_reason"])
    assert reasons == {"take_profit", "stop", "time"}
    assert result.document["signals"]["rejected:cash"] > 10  # entries competed for cash
    # Time exits at exactly an entry time happen, and do not diverge.
    entries = set(result.trades["entry_time"].to_list())
    timed = result.trades.filter(pl.col("exit_reason") == "time")
    assert any(t in entries for t in timed["exit_time"].to_list())

    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [
        "equity.csv",
        "metrics.json",
        "trades.csv",
        "trades.parquet",
        "yearly.csv",
    ]
    assert_frame_equal(pl.read_parquet(out / "trades.parquet"), result.trades)
    document = json.loads((out / "metrics.json").read_text())
    assert document["mode"] == "replay"
    assert document["metrics"]["trades"] == expected.height
    assert document["strategies"]["cap"]["trades"] == expected.height
    assert document["start"] == at(START).isoformat()
    assert result.trades["strategy"].unique().to_list() == ["cap"]
    assert "vol_mult" in result.trades.columns  # signal info is carried into the trade list


def test_replay_equals_backtest_with_equal_slots(
    root: Path, signals: Signals, tmp_path: Path
) -> None:
    config = run_config(
        tmp_path / "state.sqlite",
        [strategy_dict(capital=1000.0, sizing={"slots": 3}, risk={"max_positions": 3})],
    )
    # The fast path: the rule's batch signals (`backtest_signals`) served per tick.
    result = run_replay(
        config, at(START), at(END), root, plugins={"cap": rule_strategy()}, signals="backtest"
    )
    run = BacktestConfig(
        exit=RULE.exit, interval=H1, sizing=EqualSlots(3), max_positions=3, initial_capital=1000.0
    )
    expected = run_backtest(signals.frame, ParquetCandleSource(root), run)
    assert expected.skipped["slots_full"] > 10
    # Stakes compound from realised equity, which replay derives from the state file and the
    # backtest sums incrementally: equal up to float rounding.
    assert_frame_equal(
        trades_core(result.trades), trades_core(expected.trades), check_exact=False, rel_tol=1e-9
    )
    assert_frame_equal(result.equity, expected.equity, check_exact=False, rel_tol=1e-9)
    assert result.metrics["trades"] == expected.metrics["trades"]


def test_the_replay_feed_ticks_windows_and_staleness(root: Path) -> None:
    feed = ReplayFeed(root, {H1: ["AAAUSDT", "BTCUSDT"]}, at(300), at(305), warmup={H1: 10})
    assert feed.tick_count == 5
    assert feed.clock() == at(300)
    windows = feed.windows(H1, ["AAAUSDT", "NOPEUSDT"], at(299), 10)
    assert list(windows) == ["AAAUSDT"]
    assert windows["AAAUSDT"]["open_time"].to_list() == [at(k) for k in range(290, 300)]
    assert feed.stale_symbols(H1, at(300), 2) == set()
    assert feed.stale_symbols(H1, at(310), 2) == {"AAAUSDT", "BTCUSDT"}  # none closed after 304


def _divergence_store(
    root: Path, bars: dict[str, list[tuple[float, float, float, float] | None]]
) -> None:
    for symbol, series in bars.items():
        write_bars(root, symbol, series, H1, at(0))


PLAN = ExitPlan(stop=0.05, max_hold=10 * HOUR)


def _both(
    root: Path,
    strategy: Scheduled,
    symbols: list[str],
    *,
    max_positions: int | None,
    end: int,
) -> tuple[pl.DataFrame, pl.DataFrame, StateStore]:
    """Replay and backtest the scheduled signals; the replay's state stays open."""
    risk = {} if max_positions is None else {"max_positions": max_positions}
    entry = strategy_dict("sched", symbols=symbols, risk=risk)
    tmp_path = root
    config = run_config(tmp_path / "state.sqlite", [entry])
    result = run_replay(config, at(1), at(end), root, plugins={"sched": strategy})
    run = BacktestConfig(
        exit=PLAN,
        interval=H1,
        sizing=FixedStake(100.0),
        max_positions=max_positions,
        initial_capital=1000.0,
        costs=Costs(),
    )
    expected = run_backtest(strategy.signal_frame(), ParquetCandleSource(root), run).trades
    return result.trades, expected, StateStore.open(tmp_path / "state.sqlite")


def _scheduled(rows: dict[int, list[tuple[str, float]]]) -> Scheduled:
    return Scheduled({at(k): r for k, r in rows.items()}, name="sched", exit=PLAN)


def test_divergence_a_gap_exit_at_the_entry_time_frees_the_slot_only_in_the_backtest(
    tmp_path: Path,
) -> None:
    # AAA is entered at 3; candle 6 opens at 90, through its 5% stop: the exit fills at the
    # open of 6. BBB signals at 5 for an entry at 6 with one slot.
    aaa = flat_bars(20)
    aaa[6] = (90.0, 90.0, 90.0, 90.0)
    aaa[7:] = flat_bars(13, 90.0)
    _divergence_store(tmp_path, {"AAAUSDT": aaa, "BBBUSDT": flat_bars(20)})
    plugin = _scheduled({2: [("AAAUSDT", 0.0)], 5: [("BBBUSDT", 0.0)]})
    live, backtested, store = _both(
        tmp_path, plugin, ["AAAUSDT", "BBBUSDT"], max_positions=1, end=19
    )
    assert backtested["symbol"].to_list() == ["AAAUSDT", "BBBUSDT"]
    assert backtested["exit_time"][0] == at(6) == backtested["entry_time"][1]
    # Live learns of the gap exit only when candle 6 closes, after the entries for 6.
    assert live["symbol"].to_list() == ["AAAUSDT"]
    assert_frame_equal(trades_core(live), trades_core(backtested.head(1)))
    assert store.signal_outcomes()["rejected:slots_full"] == 1
    store.close()


def test_divergence_a_gap_exit_keeps_the_symbol_held_for_its_own_reentry(tmp_path: Path) -> None:
    aaa = flat_bars(20)
    aaa[6] = (90.0, 90.0, 90.0, 90.0)
    aaa[7:] = flat_bars(13, 90.0)
    _divergence_store(tmp_path, {"AAAUSDT": aaa})
    plugin = _scheduled({2: [("AAAUSDT", 0.0)], 5: [("AAAUSDT", 0.0)]})
    live, backtested, store = _both(tmp_path, plugin, ["AAAUSDT"], max_positions=None, end=19)
    assert backtested["entry_time"].to_list() == [at(3), at(6)]
    assert live["entry_time"].to_list() == [at(3)]
    assert store.signal_outcomes()["rejected:symbol_held"] == 1
    store.close()


def test_a_time_exit_at_the_entry_time_does_not_diverge(tmp_path: Path) -> None:
    # AAA entered at 3 with a 3-hour hold exits at 6, the close of candle 5: known in time.
    plan = ExitPlan(max_hold=3 * HOUR)
    _divergence_store(tmp_path, {"AAAUSDT": flat_bars(20), "BBBUSDT": flat_bars(20)})
    plugin = _scheduled({2: [("AAAUSDT", 0.0)], 5: [("BBBUSDT", 0.0)]})
    plugin.exit = plan
    entry = strategy_dict("sched", symbols=["AAAUSDT", "BBBUSDT"], risk={"max_positions": 1})
    result = run_replay(
        run_config(tmp_path / "s.sqlite", [entry]),
        at(1),
        at(19),
        tmp_path,
        plugins={"sched": plugin},
    )
    run = BacktestConfig(exit=plan, interval=H1, sizing=FixedStake(100.0), max_positions=1)
    expected = run_backtest(plugin.signal_frame(), ParquetCandleSource(tmp_path), run).trades
    assert expected["symbol"].to_list() == ["AAAUSDT", "BBBUSDT"]
    assert_frame_equal(trades_core(result.trades), trades_core(expected))


def test_divergence_b_a_missing_entry_candle_holds_its_stake_until_detected(tmp_path: Path) -> None:
    # AAA's candle 6 is missing. Both signal at 5 (AAA first by priority), one slot.
    aaa = flat_bars(20)
    aaa[6] = None
    _divergence_store(tmp_path, {"AAAUSDT": aaa, "BBBUSDT": flat_bars(20)})
    plugin = _scheduled({5: [("AAAUSDT", -2.0), ("BBBUSDT", -1.0)]})
    live, backtested, store = _both(
        tmp_path, plugin, ["AAAUSDT", "BBBUSDT"], max_positions=1, end=19
    )
    # The backtest skips AAA before sizing and takes BBB.
    assert backtested["symbol"].to_list() == ["BBBUSDT"]
    # Live placed AAA (its slot is reserved), refused BBB, and cancelled AAA when candle 7
    # closed without a candle 6.
    assert live.height == 0
    assert store.closed_positions() == []
    cancelled = store.position_for_order(client_order_id("sched", "AAAUSDT", at(5), Side.BUY))
    assert cancelled is not None
    assert (cancelled.status, cancelled.exit_reason) == (
        PositionStatus.CANCELLED,
        "no_entry_candle",
    )
    assert cancelled.exit_time == at(6)
    assert store.signal_outcomes() == {"placed": 1, "rejected:slots_full": 1}
    store.close()


def test_divergence_c_rule_signals_in_a_new_listings_eighth_day_are_backtest_only(
    tmp_path: Path,
) -> None:
    # AAAUSDT lists at 100; an event at 100 + 175 falls in its days 7-8 (168..191 candles).
    paths = coin_paths()
    keep = {"BTCUSDT": paths["BTCUSDT"], "AAAUSDT": paths["AAAUSDT"]}
    event = 275
    btc, aaa = keep["BTCUSDT"], keep["AAAUSDT"]
    rng = np.random.default_rng(3)
    btc["volume"][:] = rng.lognormal(np.log(1000.0), 0.1, N)
    aaa["volume"][:] = rng.lognormal(np.log(1000.0), 0.1, N)
    btc["volume"][event] *= 6
    aaa["close"][event - 3 : event + 1] *= np.linspace(1.0, 0.9, 4)
    aaa["close"][event + 1 :] *= 0.9
    aaa["volume"][event] *= 8
    write_replay_store(tmp_path, paths=keep, first={"AAAUSDT": 100})

    full = compute_signals(RULE, ["AAAUSDT"], tmp_path, workers=1).signals.frame
    assert full["open_time"].to_list() == [at(event)]

    entry = strategy_dict(symbols=["AAAUSDT"])
    config = run_config(tmp_path / "state.sqlite", [entry])
    result = run_replay(
        config, at(event - 5), at(event + 30), tmp_path, plugins={"cap": rule_strategy()}
    )
    expected = backtest(tmp_path, full, FixedStake(100.0), 1000.0)
    assert expected.height == 1
    assert result.trades.height == 0  # the window holds 176 candles, short of the warm-up
    store = StateStore.open(config.state)
    assert store.signal_outcomes() == {}
    store.close()


def test_a_replay_resumes_after_its_last_processed_tick(
    root: Path, signals: Signals, tmp_path: Path
) -> None:
    config = run_config(tmp_path / "state.sqlite", [strategy_dict(capital=350.0)])
    precomputed = {"cap": signals}
    plugins = {"cap": rule_strategy()}
    first = run_replay(config, at(START), at(300), root, plugins=plugins, precomputed=precomputed)
    assert first.open_positions > 0 or first.trades.height > 0
    resumed = run_replay(config, at(START), at(END), root, plugins=plugins, precomputed=precomputed)
    whole = run_replay(
        run_config(tmp_path / "whole.sqlite", [strategy_dict(capital=350.0)]),
        at(START),
        at(END),
        root,
        plugins=plugins,
        precomputed=precomputed,
    )
    assert_frame_equal(resumed.trades, whole.trades)
    store = StateStore.open(config.state)
    assert store.last_tick(H1) == at(END - 1)
    assert [e.kind for e in store.events_since()].count("restart") == 1
    store.close()


def test_replay_trades_carry_the_strategy_and_open_positions_count(tmp_path: Path) -> None:
    _divergence_store(tmp_path, {"AAAUSDT": flat_bars(30)})
    plugin = _scheduled({2: [("AAAUSDT", 0.0)]})
    plugin.exit = ExitPlan(max_hold=timedelta(days=5))
    entry = strategy_dict("sched", symbols=["AAAUSDT"])
    result = run_replay(
        run_config(tmp_path / "s.sqlite", [entry]),
        at(1),
        at(20),
        tmp_path,
        plugins={"sched": plugin},
    )
    assert result.trades.height == 0
    assert result.open_positions == 1
    # The open position is valued at the last close before the end (stake less costs).
    value = 100.0 / (100.0 * 1.0005 * 1.001) * 100.0
    assert result.metrics["final_equity"] == pytest.approx(900.0 + value)
