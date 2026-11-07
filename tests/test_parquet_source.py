from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from store_fakes import kline_frame, write_series

from crypto_moonshot.backtest.source import ParquetCandleSource
from crypto_moonshot.broker.candles import from_us
from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.models import Interval

M5 = timedelta(minutes=5)
M1 = timedelta(minutes=1)
# Two days either side of a month boundary.
START = datetime(2024, 1, 30, tzinfo=UTC)
FEB = datetime(2024, 2, 1, tzinfo=UTC)
END = datetime(2024, 2, 3, tzinfo=UTC)  # the data end: every full series stops here


def times(start: datetime, end: datetime, step: timedelta) -> list[datetime]:
    n = (end - start) // step
    return [start + k * step for k in range(n)]


def write(root: Path, symbol: str, interval: Interval, start: datetime, end: datetime) -> None:
    ts = times(start, end, interval.length)
    write_series(root, symbol, interval, kline_frame(ts, np.arange(len(ts)) + 100.0))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write(tmp_path, "AUSDT", Interval.M5, START, END)  # trades to the data end
    write(tmp_path, "BUSDT", Interval.M5, START, FEB + timedelta(hours=6))  # delisted
    # 1m only for part of A's history: a few hours in January, and the first hour of February.
    write(tmp_path, "AUSDT", Interval.M1, START, START + timedelta(hours=3))
    write(tmp_path, "AUSDT", Interval.M1, FEB, FEB + timedelta(hours=1))
    return tmp_path


def opens(window: object) -> list[datetime]:
    return [from_us(int(t)) for t in getattr(window, "open_time")]  # noqa: B009


def test_a_window_spans_month_files(root: Path) -> None:
    src = ParquetCandleSource(root)
    w = src.window("AUSDT", Interval.M5, FEB - 2 * M5, FEB + 2 * M5)
    assert w is not None
    assert opens(w) == [FEB - 2 * M5, FEB - M5, FEB, FEB + M5]
    assert not w.ended
    assert src.reads == 2
    k = (FEB - START) // M5
    assert w.open[0] == 100.0 + k - 2
    # The months are cached: the same window again reads nothing.
    src.window("AUSDT", Interval.M5, FEB - 2 * M5, FEB + 2 * M5)
    assert src.reads == 2


def test_a_series_that_trades_to_the_data_end_has_not_ended(root: Path) -> None:
    src = ParquetCandleSource(root)
    assert src.data_end(Interval.M5) == END
    w = src.window("AUSDT", Interval.M5, END - 2 * M5, END + timedelta(days=3))
    assert w is not None
    assert len(w) == 2
    assert not w.ended


def test_a_series_that_stops_before_the_data_end_has_ended(root: Path) -> None:
    src = ParquetCandleSource(root)
    stop = FEB + timedelta(hours=6)
    inside = src.window("BUSDT", Interval.M5, stop - timedelta(hours=1), stop - M5)
    assert inside is not None
    assert not inside.ended  # the last candle is not in the window
    w = src.window("BUSDT", Interval.M5, stop - 2 * M5, stop + timedelta(days=1))
    assert w is not None
    assert opens(w) == [stop - 2 * M5, stop - M5]
    assert w.ended
    after = src.window("BUSDT", Interval.M5, END, END + timedelta(days=10))
    assert after is not None  # February has a file: an empty window at the end of the series
    assert (len(after), after.ended) == (0, True)
    assert (
        src.window("BUSDT", Interval.M5, END + timedelta(days=30), END + timedelta(days=40)) is None
    )


def test_until_hides_newer_candles_and_sets_the_data_end(root: Path) -> None:
    until = FEB + timedelta(hours=3)
    src = ParquetCandleSource(root, until=until)
    assert src.data_end(Interval.M5) == until
    a = src.window("AUSDT", Interval.M5, until - 2 * M5, until + timedelta(days=1))
    assert a is not None
    assert opens(a) == [until - 2 * M5, until - M5]
    assert not a.ended  # A trades up to `until`
    # B stopped after `until`, so up to `until` it was still trading.
    b = src.window("BUSDT", Interval.M5, until - M5, until + timedelta(days=1))
    assert b is not None
    assert (len(b), b.ended) == (1, False)
    # Nothing at or after `until` is visible, even as a whole month.
    hidden = src.window("AUSDT", Interval.M5, until, until + timedelta(days=30))
    assert hidden is not None
    assert len(hidden) == 0
    late = ParquetCandleSource(root, until=FEB)
    assert late.window("AUSDT", Interval.M5, FEB, FEB + timedelta(days=1)) is None


def test_a_later_until_than_the_data_keeps_the_store_end(root: Path) -> None:
    src = ParquetCandleSource(root, until=END + timedelta(days=10))
    assert src.data_end(Interval.M5) == END
    w = src.window("AUSDT", Interval.M5, END - M5, END + timedelta(days=1))
    assert w is not None
    assert not w.ended


def test_missing_fine_candles(root: Path) -> None:
    src = ParquetCandleSource(root)
    # No 1m at all for B, and no 1m month file for A in March.
    assert src.window("BUSDT", Interval.M1, START, START + timedelta(hours=1)) is None
    # A January 1m file exists but covers only the first three hours: a partial window.
    w = src.window("AUSDT", Interval.M1, START + timedelta(hours=2), START + timedelta(hours=5))
    assert w is not None
    assert len(w) == 60
    assert opens(w)[-1] == START + timedelta(hours=3) - M1
    # A 1m window inside a month with a file but no candles there is empty, not None.
    empty = src.window("AUSDT", Interval.M1, FEB + timedelta(hours=2), FEB + timedelta(hours=3))
    assert empty is not None
    assert len(empty) == 0


def test_the_lru_cache_is_bounded(root: Path) -> None:
    src = ParquetCandleSource(root, cache_bytes=1)
    src.window("AUSDT", Interval.M5, FEB - M5, FEB + M5)
    src.window("AUSDT", Interval.M5, FEB - M5, FEB + M5)
    assert src.reads == 4  # one month kept at most, so both are read again


def test_read_series_respects_end(root: Path) -> None:
    store = KlineStore(root)
    frame = store.read_series("AUSDT", Interval.M5, end=FEB + M5, columns=["close"])
    assert frame.columns == ["open_time", "close"]
    assert frame["open_time"][-1] == FEB
    assert frame.height == (FEB - START) // M5 + 1
    assert store.read_series("CUSDT", Interval.M5).height == 0
