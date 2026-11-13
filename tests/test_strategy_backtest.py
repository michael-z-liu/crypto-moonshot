"""End to end on a synthetic store: F2 signals, a Parquet candle source and the engine."""

from datetime import timedelta
from pathlib import Path

import pytest
from strategy_fakes import SPIKE, at, write_capitulation_store

from crypto_moonshot.backtest.engine import BacktestConfig, run_backtest
from crypto_moonshot.backtest.source import ParquetCandleSource
from crypto_moonshot.broker.base import Costs
from crypto_moonshot.portfolio import EqualSlots
from crypto_moonshot.strategy.capitulation import F2
from crypto_moonshot.strategy.compute import compute_signals


def test_f2_through_the_engine(tmp_path: Path) -> None:
    write_capitulation_store(tmp_path)
    signals = compute_signals(F2, ["DIPUSDT", "SHALLOWUSDT", "THINUSDT"], tmp_path, workers=1)
    source = ParquetCandleSource(tmp_path)
    config = BacktestConfig(
        exit=signals.signals.exit,
        interval=signals.signals.interval,
        sizing=EqualSlots(5),
        max_positions=5,
        costs=Costs(fee=0.0, slippage=0.0),
    )
    result = run_backtest(signals.signals.frame, source, config)
    trades = result.trades
    assert trades.height == 1
    trade = trades.row(0, named=True)
    assert trade["symbol"] == "DIPUSDT"
    assert trade["signal_time"] == at(SPIKE)
    assert trade["entry_time"] == at(SPIKE + 1)
    assert trade["exit_time"] == at(SPIKE + 1) + timedelta(hours=1)
    assert trade["exit_reason"] == "time"
    # Enters at the open of the next candle (the spike's close, 85), exits at the close of
    # the 12th candle (90).
    assert (trade["entry_ref"], trade["exit_price"]) == (85.0, 90.0)
    assert trade["stake"] == pytest.approx(2_000.0)
    assert trade["net_return"] == pytest.approx(90 / 85 - 1)
    assert trade["btc_vol_mult"] == pytest.approx(10.0)
    assert result.metrics["trades"] == 1
