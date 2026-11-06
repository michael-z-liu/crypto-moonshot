"""Shared paths, universe and loading for the volume-spike research scripts."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

DATA_DIR = Path(
    os.environ.get("MOONSHOT_DATA_DIR", "/Users/mike/code/crypto-moonshot/data")
)
KLINES_5M = DATA_DIR / "klines" / "spot" / "interval=5m"
HERE = Path(__file__).resolve().parent
# Which coins (M13): "pool" = the main data folder (today's top ~200, the default);
# "poolA" = + delisted pairs from the archive; "all" = + every other archived USDT pair.
UNIVERSE = os.environ.get("MOONSHOT_UNIVERSE", "pool")
ARCHIVE = HERE / "cache" / "archive"  # store written by fetch_archive.py
ARCHIVE_5M = ARCHIVE / "klines" / "spot" / "interval=5m"
CACHE = HERE / "cache" if UNIVERSE == "pool" else HERE / f"cache_{UNIVERSE}"
RESULTS = HERE / "results" if UNIVERSE == "pool" else HERE / "results" / f"universe_{UNIVERSE}"
RESULTS.mkdir(parents=True, exist_ok=True)

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


def archive_groups() -> dict[str, list[str]]:
    """Group A (delisted) and B (listed outside the pool), as fetched by fetch_archive.py.

    Read from the data folder's `delisted.txt` / `outside_pool.txt` when present (written next to
    `top200.txt` when the archive is imported there), else from results/archive_groups.json.
    """
    a, b = DATA_DIR / "delisted.txt", DATA_DIR / "outside_pool.txt"
    if a.exists() and b.exists():
        return {"A_delisted": sorted(a.read_text().split()),
                "B_listed_outside_pool": sorted(b.read_text().split())}  # fmt: skip
    path = HERE / "results" / "archive_groups.json"
    if not path.exists():
        return {"A_delisted": [], "B_listed_outside_pool": []}
    g = json.loads(path.read_text())
    return {k: sorted(g[k]) for k in ("A_delisted", "B_listed_outside_pool")}


def stored(symbol: str) -> bool:
    return (KLINES_5M / f"symbol={symbol}").exists() or (ARCHIVE_5M / f"symbol={symbol}").exists()


def universe() -> list[str]:
    """Tradable crypto symbols, minus pegged assets and tokenised stocks.

    "pool" is the coins in the data folder's `top200.txt` (today's top ~200), so importing
    archive coins into the same folder does not change it.
    """
    listed = (DATA_DIR / "top200.txt").read_text().split()
    pool = sorted(s for s in listed if s not in PEGGED and s not in STOCKS and stored(s))
    if UNIVERSE == "pool":
        return pool
    g = archive_groups()
    extra = g["A_delisted"] + (g["B_listed_outside_pool"] if UNIVERSE == "all" else [])
    return sorted(pool + [s for s in extra if stored(s)])


DATA_END = datetime(2026, 10, 25, tzinfo=UTC)
PAD_BARS = 300  # flat candles added after a series ends early or stops for a day or more


def load(symbol: str) -> pl.DataFrame:
    """All stored 5m candles for one symbol, sorted by open time.

    Reads the main store, or the archive store for coins outside it. When a series stops early
    (a delisting) or pauses for a day or more (a halt, as LUNA in May 2022), PAD_BARS flat candles
    at the last close follow, with zero volume: a position open at that moment exits at the last
    traded price instead of being dropped.
    """
    arch = ARCHIVE_5M / f"symbol={symbol}"  # a fresh fetch wins over a (maybe partial) import
    src = arch if arch.exists() else KLINES_5M / f"symbol={symbol}"
    df = pl.read_parquet(src / "*.parquet").sort("open_time")
    if UNIVERSE == "pool":
        return df
    t = df["open_time"]
    step = timedelta(minutes=BAR_MINUTES)
    gap = t.diff().shift(-1)
    ends = df.filter((gap >= timedelta(days=1)) | (gap.is_null() & (t < DATA_END)))
    if ends.height == 0:
        return df
    pads = []
    for row in ends.iter_rows(named=True):
        nxt = t.filter(t > row["open_time"]).min()
        n = PAD_BARS if nxt is None else min(PAD_BARS, int((nxt - row["open_time"]) / step) - 1)  # type: ignore[operator]
        times = pl.datetime_range(
            row["open_time"] + step, row["open_time"] + step * n, step, eager=True, time_zone="UTC"
        )
        c = row["close"]
        pads.append(pl.DataFrame({"open_time": times}).with_columns(
            open=pl.lit(c), high=pl.lit(c), low=pl.lit(c), close=pl.lit(c), volume=pl.lit(0.0),
            quote_volume=pl.lit(0.0), trade_count=pl.lit(0, dtype=pl.Int64),
            taker_buy_base_volume=pl.lit(0.0), taker_buy_quote_volume=pl.lit(0.0),
        ))  # fmt: skip
    return pl.concat([df, *[p.cast(df.schema) for p in pads]]).sort("open_time")


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
