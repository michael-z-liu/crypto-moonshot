"""Fakes for plugin tests: a random-walk store with volume spikes after falls, and plugin files."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from store_fakes import kline_frame, write_series

from crypto_moonshot.models import Interval

M5 = timedelta(minutes=5)
T0 = datetime(2024, 1, 1, tzinfo=UTC)
N = 2900
EVENTS = (2330, 2410, 2455, 2520, 2590, 2700)
"""Candle indexes where BTC and the coins spike and the coins have just fallen sharply."""
LISTED_LATE = 300
"""`LATEUSDT` has no candles before this index (its first 2304 candles are the edge case)."""
HOLES = {"AAAUSDT": (2331, 2500, 2501, 2650), "BBBUSDT": (2400, 2411, 2589), "LATEUSDT": (2456,)}


def at(index: int) -> datetime:
    return T0 + index * M5


def write_walk_store(root: Path, *, seed: int = 7) -> list[str]:
    """BTCUSDT plus three coins; returns the coin symbols."""
    rng = np.random.default_rng(seed)
    coins = ["AAAUSDT", "BBBUSDT", "LATEUSDT"]
    for symbol in ["BTCUSDT", *coins]:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.003, N)))
        volume = rng.lognormal(np.log(1000.0), 0.3, N)
        for event in EVENTS:
            volume[event] *= 12.0 if symbol != "BTCUSDT" else 8.0
            if symbol != "BTCUSDT":
                close[event - 47 : event + 1] *= np.linspace(1.0, 0.85, 48)
        open_ = np.concatenate(([close[0]], close[:-1]))
        keep = np.ones(N, dtype=bool)
        keep[list(HOLES.get(symbol, ()))] = False
        if symbol == "LATEUSDT":
            keep[:LISTED_LATE] = False
        idx = np.flatnonzero(keep)
        frame = kline_frame(
            [at(int(k)) for k in idx],
            close[idx],
            volume[idx],
            open_=open_[idx],
            high=np.maximum(open_, close)[idx] * 1.002,
            low=np.minimum(open_, close)[idx] * 0.998,
        )
        write_series(root, symbol, Interval.M5, frame)
    return coins


PLUGIN_SOURCE = """
from datetime import timedelta

import polars as pl

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.models import Interval


class Flush:
    interval = Interval.M5
    warmup = 60
    context_symbols = ()
    exit = ExitPlan(max_hold=timedelta(hours=1))

    def __init__(self, name="flush", threshold=-0.10):
        self.name = name
        self.threshold = threshold

    def on_candles(self, windows, time, context):
        rows = []
        for symbol, w in windows.items():
            if w.height < 49 or w["open_time"][-1] != time:
                continue
            ret = w["close"][-1] / w["close"][-49] - 1
            mult = w["volume"][-1] / w["volume"][:-1].median()
            if ret <= self.threshold and mult >= 10:
                rows.append((symbol, ret, float(mult)))
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows, schema=["symbol", "priority", "mult"], orient="row")


def make(threshold=-0.10):
    return Flush(threshold=threshold)


FLUSH = Flush()
CONSTANT = 5
BROKEN = Flush
"""


def write_plugin(directory: Path) -> Path:
    path = directory / "my_plugin.py"
    path.write_text(PLUGIN_SOURCE)
    return path
