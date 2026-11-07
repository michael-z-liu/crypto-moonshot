from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from backtest_fakes import NO_COSTS, signals, source
from broker_fakes import T0, Bar, flat

from crypto_moonshot.backtest.engine import (
    TRADE_COLUMNS,
    BacktestConfig,
    BacktestResult,
    run_backtest,
)
from crypto_moonshot.backtest.source import CandleSource
from crypto_moonshot.broker.base import Costs, ExitPlan
from crypto_moonshot.models import Interval
from crypto_moonshot.portfolio import EqualSlots, FixedStake

M5 = timedelta(minutes=5)
H1 = timedelta(hours=1)
HOLD = ExitPlan(max_hold=timedelta(minutes=10))
COSTS = Costs(fee=0.001, slippage=0.0005)


def run(
    sigs: pl.DataFrame, src: CandleSource, exit: ExitPlan = HOLD, **kwargs: object
) -> BacktestResult:
    kwargs.setdefault("costs", NO_COSTS)
    return run_backtest(sigs, src, BacktestConfig(exit=exit, **kwargs))  # type: ignore[arg-type]


def taken(result: BacktestResult) -> list[str]:
    return result.trades["symbol"].to_list()


def nonzero(skipped: dict[str, int]) -> dict[str, int]:
    return {k: v for k, v in skipped.items() if v}


# 1. Timing


def test_entry_is_at_the_open_of_the_next_candle() -> None:
    bars: list[Bar | None] = [*flat(3), (101, 101, 101, 101), *flat(10, 101)]
    result = run(signals(("AUSDT", 2, 0)), source({"AUSDT": bars}))
    [trade] = result.trades.iter_rows(named=True)
    assert trade["signal_time"] == T0 + 2 * M5
    assert trade["entry_time"] == T0 + 3 * M5
    assert trade["entry_ref"] == 101
    assert trade["exit_time"] == T0 + 3 * M5 + HOLD.max_hold  # type: ignore[operator]


def test_entry_delay_adds_candles() -> None:
    bars: list[Bar | None] = [*flat(3), (101, 101, 101, 101), *flat(10, 101)]
    result = run(signals(("AUSDT", 0, 0)), source({"AUSDT": bars}), entry_delay=2)
    assert result.trades["entry_time"].to_list() == [T0 + 3 * M5]
    assert result.trades["entry_ref"].to_list() == [101]


def test_missing_entry_candle_skips_the_signal() -> None:
    src = source({"AUSDT": [*flat(3), None, *flat(10)], "BUSDT": flat(5)})
    # A hole at the entry candle, and a signal on the last candle of an ended series.
    result = run(signals(("AUSDT", 2, 0), ("BUSDT", 4, 0), ("CUSDT", 0, 0)), src)
    assert result.trades.height == 0
    assert nonzero(result.skipped) == {"no_entry_candle": 3}
    assert result.equity.height == 0
    assert result.metrics["trades"] == 0
    assert result.metrics["total_return"] == 0


# 2. Event order


def test_same_entry_time_takes_the_lowest_priority_first() -> None:
    src = source({s: flat(10) for s in ("AUSDT", "BUSDT", "CUSDT")})
    sigs = signals(("AUSDT", 0, 2.0), ("BUSDT", 0, 1.0), ("CUSDT", 0, 3.0))
    result = run(sigs, src, max_positions=2)
    assert taken(result) == ["BUSDT", "AUSDT"]
    assert nonzero(result.skipped) == {"slots_full": 1}


def test_priority_ties_go_by_symbol() -> None:
    src = source({s: flat(10) for s in ("AUSDT", "BUSDT")})
    result = run(signals(("BUSDT", 0, 1.0), ("AUSDT", 0, 1.0)), src, max_positions=1)
    assert taken(result) == ["AUSDT"]


def test_earlier_entry_wins_over_priority() -> None:
    src = source({s: flat(10) for s in ("AUSDT", "BUSDT")})
    result = run(signals(("BUSDT", 1, 0.0), ("AUSDT", 0, 9.0)), src, max_positions=1)
    assert taken(result) == ["AUSDT"]


def test_an_exit_at_the_entry_time_frees_its_slot_first() -> None:
    src = source({s: flat(20) for s in ("AUSDT", "BUSDT", "CUSDT")})
    # A enters at 5m and exits at 15m; B would enter at 10m, C at 15m.
    sigs = signals(("AUSDT", 0, 0), ("BUSDT", 1, 0), ("CUSDT", 2, 0))
    result = run(sigs, src, max_positions=1)
    assert taken(result) == ["AUSDT", "CUSDT"]
    a, c = result.trades.iter_rows(named=True)
    assert a["exit_time"] == c["entry_time"] == T0 + 3 * M5
    assert nonzero(result.skipped) == {"slots_full": 1}


def test_one_position_per_symbol() -> None:
    src = source({"AUSDT": flat(20)})
    result = run(signals(("AUSDT", 0, 0), ("AUSDT", 1, 0), ("AUSDT", 2, 0)), src)
    assert result.trades["entry_time"].to_list() == [T0 + M5, T0 + 3 * M5]
    assert nonzero(result.skipped) == {"symbol_held": 1}


# 3. Exits through the simulated broker


def test_take_profit_and_stop_with_costs() -> None:
    plan = ExitPlan(take_profit=0.1, stop=0.05, max_hold=H1)
    src = source(
        {
            "AUSDT": [*flat(2), (100, 112, 99, 105), *flat(20)],
            "BUSDT": [*flat(2), (100, 101, 90, 95), *flat(20)],
        }
    )
    result = run(signals(("AUSDT", 1, 0), ("BUSDT", 1, 0)), src, exit=plan, costs=COSTS)
    a, b = result.trades.iter_rows(named=True)
    assert a["exit_reason"] == "take_profit"
    assert a["exit_time"] == T0 + 3 * M5
    assert a["exit_price"] == pytest.approx(110 * 0.9995)
    assert a["entry_price"] == pytest.approx(100 * 1.0005)
    assert a["net_return"] == pytest.approx(COSTS.net_return(100.05, 110 * 0.9995))
    assert a["pnl"] == pytest.approx(50 * a["net_return"])
    assert a["quantity"] == pytest.approx(50 / (100.05 * 1.001))
    assert b["exit_reason"] == "stop"
    assert b["exit_price"] == pytest.approx(95 * 0.9995)


def test_fine_candles_resolve_a_candle_that_touches_both_levels() -> None:
    plan = ExitPlan(take_profit=0.1, stop=0.05, max_hold=H1)
    coarse: list[Bar | None] = [*flat(2), (100, 112, 90, 100), *flat(20)]
    # In 1m candles the target is reached first, then the stop.
    up: list[Bar | None] = [(100, 112, 100, 111), (111, 111, 90, 92), *flat(3)]
    fine = [*flat(10), *up, *flat(100)]
    sigs = signals(("AUSDT", 1, 0))
    coarse_only = run(sigs, source({"AUSDT": coarse}, fine={"AUSDT": fine}), exit=plan)
    assert coarse_only.trades["exit_reason"].to_list() == ["stop"]
    with_fine = run(sigs, source({"AUSDT": coarse}, fine={"AUSDT": fine}), exit=plan, use_fine=True)
    assert with_fine.trades["exit_reason"].to_list() == ["take_profit"]
    assert with_fine.trades["exit_time"].to_list() == [T0 + 2 * M5 + timedelta(minutes=1)]


def test_use_fine_without_fine_candles_falls_back_to_the_signal_interval() -> None:
    plan = ExitPlan(take_profit=0.1, stop=0.05, max_hold=H1)
    src = source({"AUSDT": [*flat(2), (100, 112, 90, 100), *flat(20)]})
    result = run(signals(("AUSDT", 1, 0)), src, exit=plan, use_fine=True)
    assert result.trades["exit_reason"].to_list() == ["stop"]


def test_window_grows_until_the_exit_is_known() -> None:
    plan = ExitPlan(take_profit=0.1)  # no time exit
    bars: list[Bar | None] = [*flat(100), (100, 115, 100, 112), *flat(10)]
    result = run(signals(("AUSDT", 0, 0)), source({"AUSDT": bars}), exit=plan, window=M5)
    assert result.trades["exit_reason"].to_list() == ["take_profit"]
    assert result.trades["exit_time"].to_list() == [T0 + 101 * M5]


# 4. Sizing


def test_fixed_stake_does_not_compound() -> None:
    plan = ExitPlan(take_profit=0.5, max_hold=H1)
    up: list[Bar | None] = [*flat(2), (100, 200, 100, 150), *flat(30, 150)]
    src = source({"AUSDT": up, "BUSDT": up})
    result = run(signals(("AUSDT", 1, 0), ("BUSDT", 5, 0)), src, exit=plan, sizing=FixedStake(50))
    assert result.trades["stake"].to_list() == [50, 50]
    assert result.trades["pnl"].to_list() == [pytest.approx(25), pytest.approx(0)]
    assert result.metrics["pnl"] == pytest.approx(25)
    assert result.metrics["final_equity"] == pytest.approx(10_025)


def test_equal_slots_compound_on_realised_equity() -> None:
    plan = ExitPlan(take_profit=0.5, max_hold=timedelta(hours=2))
    # A doubles at once and exits; B rises 40% but stays open (not marked for sizing).
    a: list[Bar | None] = [*flat(1), (100, 150, 100, 150), *flat(30, 150)]
    b: list[Bar | None] = [*flat(1), (100, 140, 100, 140), *flat(30, 140)]
    src = source({"AUSDT": a, "BUSDT": b, "CUSDT": flat(40)})
    sigs = signals(("AUSDT", 0, 0), ("BUSDT", 0, 1), ("CUSDT", 3, 0))
    result = run(sigs, src, exit=plan, sizing=EqualSlots(4), initial_capital=1_000.0)
    stakes = dict(zip(taken(result), result.trades["stake"].to_list(), strict=True))
    # A made +125 on 250; equity for sizing = cash + B's cost = 1125.
    assert stakes == {"AUSDT": 250, "BUSDT": 250, "CUSDT": pytest.approx(1_125 / 4)}


def test_entry_short_of_cash_is_skipped() -> None:
    src = source({s: flat(20) for s in ("AUSDT", "BUSDT")})
    sigs = signals(("AUSDT", 0, 0), ("BUSDT", 0, 1))
    result = run(sigs, src, sizing=FixedStake(600), initial_capital=1_000.0)
    assert taken(result) == ["AUSDT"]
    assert nonzero(result.skipped) == {"cash": 1}


# 5. Risk checks


def test_max_exposure() -> None:
    src = source({s: flat(20) for s in ("AUSDT", "BUSDT", "CUSDT")})
    sigs = signals(("AUSDT", 0, 0), ("BUSDT", 0, 1), ("CUSDT", 0, 2))
    result = run(sigs, src, sizing=FixedStake(250), initial_capital=1_000.0, max_exposure=0.5)
    assert taken(result) == ["AUSDT", "BUSDT"]
    assert nonzero(result.skipped) == {"exposure": 1}


def test_daily_loss_limit_halts_entries_for_the_rest_of_the_utc_day() -> None:
    plan = ExitPlan(stop=0.3, max_hold=timedelta(hours=2))
    crash: list[Bar | None] = [*flat(2), (100, 100, 50, 60), *flat(60, 60)]
    src = source({"AUSDT": crash, "BUSDT": flat(60), "CUSDT": flat(60)}, interval=Interval.H1)
    # A loses 30% of 500 = -150 (over 10% of 1000) on day 1 at 03:00.
    sigs = signals(
        ("AUSDT", 1, 0), ("BUSDT", 5, 0), ("CUSDT", 23, 0), ("BUSDT", 30, 0),
        interval=Interval.H1,
    )  # fmt: skip
    result = run(
        sigs,
        src,
        exit=plan,
        interval=Interval.H1,
        sizing=FixedStake(500),
        initial_capital=1_000.0,
        daily_loss_limit=0.1,
    )
    assert result.trades["entry_time"].to_list() == [T0 + 2 * H1, T0 + 24 * H1, T0 + 31 * H1]
    assert result.trades["exit_reason"].to_list()[0] == "stop"
    assert nonzero(result.skipped) == {"daily_loss_limit": 1}


# 6. Ended series


def test_position_in_an_ended_series_closes_at_the_last_close() -> None:
    bars: list[Bar | None] = [*flat(2), (100, 101, 99, 97), (97, 98, 96, 96.5)]
    result = run(
        signals(("AUSDT", 1, 0)),
        source({"AUSDT": bars}),
        exit=ExitPlan(take_profit=0.5, max_hold=timedelta(days=1)),
        costs=COSTS,
    )
    [trade] = result.trades.iter_rows(named=True)
    assert trade["exit_reason"] == "series_end"
    assert trade["exit_time"] == T0 + 4 * M5
    assert trade["exit_price"] == pytest.approx(96.5 * 0.9995)
    assert result.equity["equity"][-1] == pytest.approx(10_000 + trade["pnl"])


def test_position_open_at_the_data_end_closes_at_the_last_close() -> None:
    """The series is still trading (not `ended`) but the data stops: no `max_hold`, so without
    the data-end cap the engine would widen its window forever and raise."""
    bars: list[Bar | None] = [
        *flat(2),
        (100, 101, 99, 100),
        (100, 103, 99, 102),
        (102, 104, 101, 103),
    ]
    end = T0 + 5 * M5
    plan = ExitPlan(stop=0.5)
    result = run(
        signals(("AUSDT", 1, 0)),
        source({"AUSDT": bars}, ended=False, data_end=end),
        plan,
        costs=COSTS,
    )
    [trade] = result.trades.iter_rows(named=True)
    assert trade["exit_reason"] == "series_end"
    assert trade["exit_time"] == end
    assert trade["exit_price"] == pytest.approx(103 * 0.9995)
    # A source that does not know its data end cannot tell, and says so.
    with pytest.raises(RuntimeError, match="no exit"):
        run(signals(("AUSDT", 1, 0)), source({"AUSDT": bars}, ended=False), plan)


def test_data_end_also_caps_a_time_exit_that_lies_beyond_it() -> None:
    bars: list[Bar | None] = [*flat(2), (100, 101, 99, 100), (100, 103, 99, 102)]
    result = run(
        signals(("AUSDT", 1, 0)),
        source({"AUSDT": bars}, ended=False, data_end=T0 + 4 * M5),
        exit=ExitPlan(max_hold=timedelta(days=1)),
    )
    assert result.trades["exit_reason"].to_list() == ["series_end"]
    assert result.trades["exit_price"].to_list() == [102]


# 7. Mark-to-market equity curve


def test_equity_curve_marks_open_positions_at_the_last_close() -> None:
    day = 24
    # Enter at 12:00 day 1 at 100; day 1 closes at 110; day 2 has a hole from 20:00 after a
    # close of 120; exit at the time limit on day 3 at 06:00 at 90.
    bars: list[Bar | None] = [*flat(12), *flat(11, 100), (100, 110, 100, 110)]
    bars += [*flat(19, 110)[:-1], (110, 120, 110, 120), None, None, None, None, None]
    bars += [*flat(6, 90), *flat(10, 90)]
    assert len(bars) == 2 * day + 16
    plan = ExitPlan(max_hold=timedelta(hours=42))
    result = run(
        signals(("AUSDT", 11, 0), interval=Interval.H1),
        source({"AUSDT": bars}, interval=Interval.H1),
        exit=plan,
        interval=Interval.H1,
        sizing=FixedStake(1_000),
        initial_capital=2_000.0,
    )
    [trade] = result.trades.iter_rows(named=True)
    assert trade["exit_time"] == T0 + 54 * H1
    q = 10.0
    eq = result.equity
    assert eq["time"].to_list() == [T0 + k * timedelta(days=1) for k in range(4)]
    assert eq["cash"].to_list() == [2_000, 1_000, 1_000, pytest.approx(1_900)]
    assert eq["open_value"].to_list() == [0, pytest.approx(q * 110), pytest.approx(q * 120), 0]
    assert eq["open_positions"].to_list() == [0, 1, 1, 0]
    assert eq["equity"].to_list() == [2_000, 2_100, 2_200, pytest.approx(1_900)]


def test_equity_curve_can_mark_more_often() -> None:
    bars: list[Bar | None] = [*flat(1), (100, 100, 100, 104), (104, 104, 104, 108), *flat(5, 108)]
    result = run(
        signals(("AUSDT", 0, 0)),
        source({"AUSDT": bars}),
        exit=ExitPlan(max_hold=timedelta(minutes=15)),
        sizing=FixedStake(100),
        initial_capital=1_000.0,
        mark_every=M5,
    )
    # Entry at 5m at 100 (marked at the entry open), exit at 20m at 108.
    assert result.equity["time"].to_list() == [T0 + k * M5 for k in range(1, 5)]
    assert result.equity["equity"].to_list() == [
        1_000,
        pytest.approx(1_004),
        pytest.approx(1_008),
        pytest.approx(1_008),
    ]


# Signals in, trades out


def test_trade_list_carries_signal_columns() -> None:
    src = source({s: flat(20) for s in ("AUSDT", "BUSDT")})
    sigs = signals(("BUSDT", 3, 0.5), ("AUSDT", 1, 0.25), score=[2.0, 1.0], tag=["b", "a"])
    result = run(sigs, src)
    assert result.trades.columns == [*TRADE_COLUMNS, "score", "tag"]
    assert result.trades.select("symbol", "priority", "score", "tag").rows() == [
        ("AUSDT", 0.25, 1.0, "a"),
        ("BUSDT", 0.5, 2.0, "b"),
    ]
    assert result.trades["signal_time"].dtype == pl.Datetime("us", "UTC")


def test_start_and_end_filter_signals_by_open_time() -> None:
    src = source({s: flat(20) for s in ("AUSDT", "BUSDT", "CUSDT")})
    sigs = signals(("AUSDT", 0, 0), ("BUSDT", 4, 0), ("CUSDT", 8, 0))
    result = run(sigs, src, start=T0 + 4 * M5, end=T0 + 8 * M5)
    assert taken(result) == ["BUSDT"]
    assert nonzero(result.skipped) == {"before_start": 1, "after_end": 1}


def test_signal_validation() -> None:
    src = source({"AUSDT": flat(5)})
    with pytest.raises(ValueError, match="missing columns: priority"):
        run(signals(("AUSDT", 0, 0)).drop("priority"), src)
    naive = signals(("AUSDT", 0, 0)).with_columns(pl.col("open_time").dt.replace_time_zone(None))
    with pytest.raises(ValueError, match="timezone-aware"):
        run(naive, src)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="entry_delay"):
        BacktestConfig(exit=HOLD, entry_delay=-1)
    with pytest.raises(ValueError, match="naive"):
        BacktestConfig(exit=HOLD, start=datetime(2024, 1, 1))  # noqa: DTZ001
    assert BacktestConfig(exit=HOLD, start=datetime(2024, 1, 1, tzinfo=UTC)).start == T0
