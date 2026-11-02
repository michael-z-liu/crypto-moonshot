"""Shared paths, universe and loading for the volume-spike research scripts."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

DATA_DIR = Path(
    os.environ.get("MOONSHOT_DATA_DIR", "/Users/michael/Documents/code/crypto-moonshot/data")
)
KLINES_5M = DATA_DIR / "klines" / "spot" / "interval=5m"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"

BAR_MINUTES = 5
BARS_PER_HOUR = 60 // BAR_MINUTES
BARS_PER_DAY = 24 * BARS_PER_HOUR
HORIZON = 48  # forward bars kept per event (4 hours)

# Stablecoins, fiat, gold and wrapped assets: no independent price action.
PEGGED = {
    "BFUSDUSDT", "RLUSDUSDT", "RWUSDUSDT", "USDEUSDT", "UUSDT", "XUSDUSDT", "USD1USDT",
    "FDUSDUSDT", "USDCUSDT", "EURIUSDT", "EURUSDT", "PAXGUSDT", "XAUTUSDT", "WBTCUSDT",
    "WBETHUSDT",
}  # fmt: skip
# Tokenised stocks and ETFs (Binance "…B" tickers): they follow stock-market hours and news.
STOCKS = {
    "SPYBUSDT", "AAPLBUSDT", "QQQBUSDT", "GOOGLBUSDT", "NVDABUSDT", "METABUSDT", "BABABUSDT",
    "TSLABUSDT", "TQQQBUSDT", "SPCXBUSDT", "EWYBUSDT", "AMDBUSDT", "COINBUSDT", "KSTRBUSDT",
    "MSTRBUSDT", "INTCBUSDT", "MUBUSDT", "DRAMBUSDT", "MRVLBUSDT", "CRCLBUSDT", "SNDKBUSDT",
    "CRWVBUSDT", "SKHYBUSDT", "BNCBUSDT", "LITEBUSDT", "NBISBUSDT", "SOXSBUSDT", "SOXLBUSDT",
    "INTWBUSDT", "SNXXBUSDT", "KORUBUSDT",
}  # fmt: skip

TRAIN_END = datetime(2024, 1, 1, tzinfo=UTC)
VALID_END = datetime(2025, 1, 1, tzinfo=UTC)


def all_symbols() -> list[str]:
    return sorted(p.name.removeprefix("symbol=") for p in KLINES_5M.glob("symbol=*"))


def universe() -> list[str]:
    """Tradable crypto symbols: everything stored except pegged assets and tokenised stocks."""
    return [s for s in all_symbols() if s not in PEGGED and s not in STOCKS]


def load(symbol: str) -> pl.DataFrame:
    """All stored 5m candles for one symbol, sorted by open time."""
    return pl.read_parquet(KLINES_5M / f"symbol={symbol}" / "*.parquet").sort("open_time")


def split_of(col: str = "open_time") -> pl.Expr:
    """'train' before 2024, 'valid' in 2024, 'test' from 2025."""
    t = pl.col(col)
    return (
        pl.when(t < TRAIN_END)
        .then(pl.lit("train"))
        .when(t < VALID_END)
        .then(pl.lit("valid"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )
