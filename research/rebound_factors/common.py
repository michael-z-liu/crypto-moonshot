"""Shared paths, universe and protocol constants for the rebound-factors study.

The protocol (entries, labels, split, factors, model, book, confirm rule) is fixed in
`docs/plans/rebound-factors.md`. The constants here are that protocol in code: change the plan
first.

Times are int64 microseconds since the Unix epoch, UTC, throughout the scripts; `D` is a UTC
midnight.

Reuse of `research/ma_rebound`: its modules are all named by their role (`common`, `events`) and
import each other as top-level modules, so importing them next to this study's own `common`
would clash in `sys.modules`. The universe comes from the same library call it uses
(`universe_symbols("all", ...)`), the pre-2020 daily candles it downloaded are read from its
cache (read only), and its daily bars are read in `check_panel.py` to cross-check `liq`.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from crypto_moonshot.data.universe import universe_symbols

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]

DATA_DIR = Path(os.environ.get("MOONSHOT_DATA_DIR", "/Users/mike/code/crypto-moonshot/data"))
KLINES_5M = DATA_DIR / "klines" / "spot" / "interval=5m"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"  # a gitignored link to DATA_DIR/cache/rebound_factors (see the plan)
SEALED_CACHE = CACHE / "sealed"  # 2024-26 outcomes; discovery scripts never open it
RESULTS = HERE / "results"
SEALED_RESULTS = RESULTS / "sealed"  # gitignored, for later milestones
MA_CACHE = HERE.parent / "ma_rebound" / "cache"  # read only
MA_DAILY_PRE2020 = MA_CACHE / "daily_pre2020"  # one parquet of daily candles per pre-2020 coin
MA_BARS_1D = MA_CACHE / "parts" / "bars_1d"

PANEL = CACHE / "panel.parquet"
LABELS = CACHE / "labels.parquet"  # discovery and the unused gap only
SEALED_LABELS = SEALED_CACHE / "labels.parquet"  # every entry with D >= 2024-01-01

MINUTE = 60_000_000
FIVE = 5 * MINUTE
HOUR = 60 * MINUTE
DAY = 24 * HOUR


def us(dt: datetime) -> int:
    return int(dt.timestamp()) * 1_000_000


def utc(t_us: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=t_us)


# --- protocol --------------------------------------------------------------------------------

DATA_END = datetime(2026, 11, 2, tzinfo=UTC)  # exclusive, for every table
FIRST_D = datetime(2020, 1, 31, tzinfo=UTC)
DISCOVERY_START = FIRST_D
DISCOVERY_END = datetime(2023, 12, 1, tzinfo=UTC)  # last discovery D (inclusive)
SEALED_START = datetime(2024, 1, 1, tzinfo=UTC)  # entries from here on are sealed
CONFIRM_START = SEALED_START
CONFIRM_END = datetime(2026, 10, 2, tzinfo=UTC)  # last confirm D (inclusive)

WINDOW = timedelta(days=30)
WINDOW_7 = timedelta(days=7)
LAST_D = DATA_END - WINDOW  # the last D whose 30-day window is complete by the data end

TARGET = 0.30  # high reaches (1 + TARGET) P
STOPS = {"10": 0.10, "20": 0.20, "ns": None}  # stop variants: low reaches (1 - stop) P
PRIMARY_STOP = "10"

MIN_LIQUIDITY = 1_000_000.0  # mean daily quote volume over the previous 30 complete UTC days
LIQUIDITY_DAYS = 30
MIN_DAYS_LISTED = 30  # complete UTC days since the first candle
STALE = DAY  # P must come from a candle that opened in the day before D (see build_panel.py)

FEE = 0.001  # per side
SLIPPAGE = 0.001  # on market fills: the entry, stop exits and time exits

DATA_END_US = us(DATA_END)
FIRST_D_US = us(FIRST_D)
LAST_D_US = us(LAST_D)
DISCOVERY_END_US = us(DISCOVERY_END)
SEALED_START_US = us(SEALED_START)
CONFIRM_END_US = us(CONFIRM_END)
WINDOW_US = WINDOW // timedelta(microseconds=1)
WINDOW_7_US = WINDOW_7 // timedelta(microseconds=1)


def split_of(d_us: int) -> str:
    """discovery | gap | confirm | unused (after the confirm range, still a complete window)."""
    if d_us <= DISCOVERY_END_US:
        return "discovery"
    if d_us < SEALED_START_US:
        return "gap"
    if d_us <= CONFIRM_END_US:
        return "confirm"
    return "unused"


def split_expr(col: str = "d_us") -> pl.Expr:
    c = pl.col(col)
    return (
        pl.when(c <= DISCOVERY_END_US)
        .then(pl.lit("discovery"))
        .when(c < SEALED_START_US)
        .then(pl.lit("gap"))
        .when(c <= CONFIRM_END_US)
        .then(pl.lit("confirm"))
        .otherwise(pl.lit("unused"))
    )


def universe() -> list[str]:
    """The "all" universe: 582 USDT pairs incl. delisted, minus pegged assets and stocks."""
    return universe_symbols("all", DATA_DIR)


# --- candles ---------------------------------------------------------------------------------

CANDLE_COLS = ("open", "high", "low", "close", "quote_volume")


def read_5m(symbol: str, end_us: int = DATA_END_US) -> pl.DataFrame:
    """A coin's 5m candles that closed by `end_us`: `t` (open time, µs), OHLC, quote volume."""
    folder = KLINES_5M / f"symbol={symbol}"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        schema = {"t": pl.Int64, **dict.fromkeys(CANDLE_COLS, pl.Float64)}
        return pl.DataFrame(schema=schema)
    df = pl.read_parquet(files, columns=["open_time", *CANDLE_COLS])
    df = df.with_columns(t=pl.col("open_time").dt.epoch("us")).select("t", *CANDLE_COLS)
    return df.filter(pl.col("t") + FIVE <= end_us).sort("t")


def first_day_pre2020(symbol: str) -> int | None:
    """UTC day number of a coin's first daily candle before 2020, if it traded then."""
    path = MA_DAILY_PRE2020 / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pl.read_parquet(path, columns=["open_time"])
    if df.height == 0:
        return None
    first = int(df.select(pl.col("open_time").dt.epoch("us").min()).item())
    return first // DAY
