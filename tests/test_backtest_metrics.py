import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from backtest_fakes import NO_COSTS, signals, source
from broker_fakes import T0, Bar, flat

from crypto_moonshot.backtest.engine import BacktestConfig, run_backtest
from crypto_moonshot.backtest.metrics import (
    equity_metrics,
    time_in_market,
    trade_metrics,
    yearly_metrics,
)
from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.models import Interval
from crypto_moonshot.portfolio import FixedStake

DAY = timedelta(days=1)
H1 = timedelta(hours=1)
UTC_TIME = pl.Datetime("us", "UTC")


def curve(*equity: float, start: datetime = T0) -> pl.DataFrame:
    return pl.DataFrame(
        {"time": [start + k * DAY for k in range(len(equity))], "equity": list(equity)},
        schema={"time": UTC_TIME, "equity": pl.Float64},
    )


def trades(*rows: tuple[datetime, datetime, float, float]) -> pl.DataFrame:
    """(entry, exit, net_return, pnl) rows."""
    return pl.DataFrame(
        {
            "entry_time": [r[0] for r in rows],
            "exit_time": [r[1] for r in rows],
            "net_return": [r[2] for r in rows],
            "pnl": [r[3] for r in rows],
        },
        schema={
            "entry_time": UTC_TIME,
            "exit_time": UTC_TIME,
            "net_return": pl.Float64,
            "pnl": pl.Float64,
        },
    )


def test_trade_metrics() -> None:
    tr = trades(
        (T0, T0 + 2 * H1, 0.10, 5.0),
        (T0, T0 + 4 * H1, -0.05, -2.5),
        (T0, T0 + 6 * H1, 0.01, 0.5),
    )
    m = trade_metrics(tr)
    assert m == {
        "trades": 3,
        "mean_bp": pytest.approx(200),
        "win_rate": pytest.approx(2 / 3),
        "pnl": pytest.approx(3.0),
        "worst_bp": pytest.approx(-500),
        "avg_hold_hours": pytest.approx(4),
    }
    empty = trade_metrics(tr.clear())
    assert (empty["trades"], empty["pnl"]) == (0, 0)
    assert math.isnan(empty["mean_bp"])


def test_equity_metrics() -> None:
    eq = curve(100, 110, 99, 120)
    m = equity_metrics(eq, trades((T0, T0 + DAY, 0.0, 0.0)), initial_capital=100.0)
    returns = np.array([0.1, 99 / 110 - 1, 120 / 99 - 1])
    assert m["final_equity"] == 120
    assert m["total_return"] == pytest.approx(0.2)
    assert m["cagr"] == pytest.approx(1.2 ** (365.25 / 3) - 1)
    assert m["max_drawdown"] == pytest.approx(99 / 110 - 1)
    assert m["sharpe"] == pytest.approx(returns.mean() / returns.std(ddof=1) * math.sqrt(365))
    assert m["time_in_market"] == pytest.approx(1 / 3)


def test_drawdown_counts_the_initial_capital_as_a_peak() -> None:
    m = equity_metrics(curve(90, 95), trades(), initial_capital=100.0)
    assert m["max_drawdown"] == pytest.approx(-0.1)
    assert m["total_return"] == pytest.approx(-0.05)


def test_time_in_market_merges_overlapping_positions() -> None:
    tr = trades(
        (T0, T0 + 10 * H1, 0, 0),
        (T0 + 5 * H1, T0 + 12 * H1, 0, 0),
        (T0 + 6 * H1, T0 + 7 * H1, 0, 0),
        (T0 + 18 * H1, T0 + 30 * H1, 0, 0),  # clipped to the span
    )
    assert time_in_market(tr, T0, T0 + DAY) == pytest.approx(18 / 24)
    assert time_in_market(trades(), T0, T0 + DAY) == 0


def test_yearly_metrics_by_entry_year_and_calendar_year_returns() -> None:
    new_year = datetime(2024, 1, 1, tzinfo=UTC)
    eq = curve(1_000, 1_100, 1_050, start=new_year - DAY)
    tr = trades(
        (new_year - 2 * H1, new_year + H1, 0.1, 100.0),
        (new_year + 2 * H1, new_year + 3 * H1, -0.05, -50.0),
    )
    yearly = yearly_metrics(tr, eq, initial_capital=1_000.0)
    assert yearly["year"].to_list() == [2023, 2024]
    assert yearly["trades"].to_list() == [1, 1]
    assert yearly["pnl"].to_list() == [100, -50]
    assert yearly["mean_bp"].to_list() == [pytest.approx(1_000), pytest.approx(-500)]
    assert yearly["return"].to_list() == [pytest.approx(0.1), pytest.approx(1_050 / 1_100 - 1)]


def test_yearly_metrics_with_a_year_without_trades() -> None:
    eq = curve(1_000, 1_000, start=datetime(2023, 12, 31, tzinfo=UTC))
    tr = trades((datetime(2023, 12, 31, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC), 0, 0))
    yearly = yearly_metrics(tr, eq, initial_capital=1_000.0)
    assert yearly.select("year", "trades", "pnl").rows() == [(2023, 1, 0), (2024, 0, 0)]
    assert yearly["return"].to_list() == [0, 0]


def test_first_day_pnl_is_in_the_total_return() -> None:
    # One trade, entered and closed on the first day, +10% on a $1,000 stake.
    bars: list[Bar | None] = [*flat(2), (100, 115, 100, 112), *flat(30, 112)]
    config = BacktestConfig(
        exit=ExitPlan(take_profit=0.1, max_hold=timedelta(hours=6)),
        interval=Interval.H1,
        costs=NO_COSTS,
        sizing=FixedStake(1_000),
        initial_capital=10_000.0,
    )
    result = run_backtest(
        signals(("AUSDT", 1, 0), interval=Interval.H1),
        source({"AUSDT": bars}, interval=Interval.H1),
        config,
    )
    assert result.trades["exit_time"].to_list() == [T0 + 3 * H1]
    assert result.equity["equity"].to_list() == [10_000, pytest.approx(10_100)]
    assert result.metrics["total_return"] == pytest.approx(0.01)
    assert result.metrics["pnl"] == pytest.approx(100)
    assert result.yearly.select("year", "trades").rows() == [(2024, 1)]
    assert result.yearly["return"].to_list() == [pytest.approx(0.01)]
