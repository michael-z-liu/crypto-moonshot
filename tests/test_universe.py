from datetime import UTC, datetime
from pathlib import Path

import pytest
from store_fakes import kline_frame, write_series

from crypto_moonshot.data.universe import read_symbol_list, universe_symbols
from crypto_moonshot.models import Interval

T = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for symbol in ("BTCUSDT", "ETHUSDT", "USDCUSDT", "AAPLBUSDT", "OLDUSDT", "ODDUSDT"):
        write_series(tmp_path, symbol, Interval.M5, kline_frame([T], [1.0]))
    (tmp_path / "top200.txt").write_text("BTCUSDT ETHUSDT\nUSDCUSDT AAPLBUSDT NEWUSDT ETHUSDT\n")
    (tmp_path / "delisted.txt").write_text("OLDUSDT\n")
    (tmp_path / "outside_pool.txt").write_text("ODDUSDT GONEUSDT\n")
    return tmp_path


def test_pool_drops_pegged_stocks_and_unstored_symbols(root: Path) -> None:
    assert universe_symbols("pool", root) == ["BTCUSDT", "ETHUSDT"]


def test_all_adds_delisted_and_outside_pool(root: Path) -> None:
    assert universe_symbols("all", root) == ["BTCUSDT", "ETHUSDT", "ODDUSDT", "OLDUSDT"]
    assert universe_symbols("all", root, interval=Interval.M1) == []


def test_symbol_lists_keep_order_without_duplicates(root: Path) -> None:
    assert read_symbol_list(root / "top200.txt") == [
        "BTCUSDT",
        "ETHUSDT",
        "USDCUSDT",
        "AAPLBUSDT",
        "NEWUSDT",
    ]
