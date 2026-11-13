"""Shared paths, universe and protocol constants for the MA-rebound study.

The protocol (what a candidate, an entry and an exit are, and how variants are judged) is fixed in
`docs/plans/ma-rebound.md`. The constants here are that protocol in code: change the plan first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from crypto_moonshot.data.universe import universe_symbols

DATA_DIR = Path(os.environ.get("MOONSHOT_DATA_DIR", "/Users/mike/code/crypto-moonshot/data"))
KLINES = DATA_DIR / "klines" / "spot"
KLINES_5M = KLINES / "interval=5m"
KLINES_1M = KLINES / "interval=1m"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"  # a gitignored link to DATA_DIR/cache/ma_rebound (see the plan)
RESULTS = HERE / "results"
SEALED = RESULTS / "sealed"  # gitignored: 2025-26 outcomes until the M6 reveal
DAILY_PRE2020 = CACHE / "daily_pre2020"  # fetch_daily.py: one parquet per coin listed before 2020

STORE_START = datetime(2020, 1, 1, tzinfo=UTC)  # first 5m candle in the store
DATA_END = datetime(2026, 11, 2, tzinfo=UTC)  # exclusive, for every table and book
BOOK_START = datetime(2022, 1, 1, tzinfo=UTC)
HOLDOUT_START = datetime(2025, 1, 1, tzinfo=UTC)  # sealed from here until the M6 reveal
SELECT_YEARS = (2022, 2023, 2024)
FOLD_YEARS = (2022, 2023, 2024, 2025, 2026)

MIN_DAILY_QUOTE_VOLUME = 1_000_000.0  # mean over the previous 30 complete UTC days
LIQUIDITY_DAYS = 30
ATR_DAYS = 14
MAJORS_TOP = 20  # plus BTC and ETH


@dataclass(frozen=True)
class Timeframe:
    name: str  # 1h | 4h | 1d | 1w | 1M
    hold: timedelta  # T, the hold limit; the order is valid for T/2
    above_bars: int  # k: completed bars that must have stayed above the level

    @property
    def hold_days(self) -> float:
        return self.hold / timedelta(days=1)


TIMEFRAMES = {
    "1h": Timeframe("1h", timedelta(hours=8), 12),
    "4h": Timeframe("4h", timedelta(hours=32), 12),
    "1d": Timeframe("1d", timedelta(days=8), 10),
    "1w": Timeframe("1w", timedelta(days=21), 4),
    "1M": Timeframe("1M", timedelta(days=28), 2),
}
MA_LENGTHS = {
    "1h": (50, 100, 200),
    "4h": (50, 100, 200),
    "1d": (50, 100, 200),
    "1w": (20, 50, 100, 200),
    "1M": (10, 20),
}
PLACEBO_LENGTHS = (37, 73, 137, 163)  # on every timeframe, M2 only

ARM_ZONE = 0.45  # armed by a 5m close >= L(1 + 0.45u)
TRIGGER_ZONE = 0.30  # trigger: first 5m close <= L(1 + 0.30u) while armed and above
BUFFERS = (0.0, 0.05, 0.15)  # limit price L(1 + b*u); 0.05 is the primary
PRIMARY_BUFFER = 0.05

FEE = 0.001  # per side
MARKET_SLIPPAGE = 0.001  # slippage and latency on taker entries, time exits and stops
STRESS_THROUGH = 0.0005  # stress: fill only when the price trades 5 bp through the limit
STRESS_EXTRA_COST = 0.001  # stress: extra cost per side

INITIAL_CAPITAL = 10_000.0
SLOTS = 10
BASE_PERCENTILE = 0.80


@dataclass(frozen=True)
class Exit:
    """An exit of the protocol. Distances are multiples of u; `time` multiplies the hold T."""

    name: str
    time: float
    take_profit: float | None = None  # above the entry price
    stop_below_level: float | None = None  # below the level L at the fill


EXITS = {
    "E1": Exit("E1", 1.0),
    "E2": Exit("E2", 0.5),
    "E3": Exit("E3", 1.0, take_profit=1.0, stop_below_level=0.5),
    "E4": Exit("E4", 1.0, take_profit=0.5, stop_below_level=0.25),
    "E5": Exit("E5", 1.0, stop_below_level=0.5),
    "E6": Exit("E6", 2.0),  # capped at 28 days
}
MAX_HOLD = timedelta(days=28)
BASE_EXIT = "E1"


def ma_names(*, placebo: bool = False) -> list[tuple[str, int]]:
    """(timeframe, length) of every study MA, or of the placebo MAs."""
    if placebo:
        return [(tf, n) for tf in TIMEFRAMES for n in PLACEBO_LENGTHS]
    return [(tf, n) for tf, lengths in MA_LENGTHS.items() for n in lengths]


def universe() -> list[str]:
    """The "all" universe: 582 USDT pairs incl. delisted, minus pegged assets and stocks."""
    return universe_symbols("all", DATA_DIR)
