"""Universes: which symbols a backtest trades, from symbol lists in the data root.

The data root holds plain-text lists of symbols (whitespace separated):

- `top200.txt`: today's top ~200 Binance USDT pairs.
- `delisted.txt`: pairs delisted since 2020 whose history was archived.
- `outside_pool.txt`: other listed USDT pairs outside the top 200.

Two universes, as in the volume-spike research (`research/volume_spike/common.py`):

- `pool`: `top200.txt` minus pegged assets (`PEGGED`) and tokenised stocks (`STOCKS`), only
  symbols with stored candles at the universe interval.
- `all`: `pool` plus `delisted.txt` and `outside_pool.txt`, with the same exclusions and the
  same stored-candles condition.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.models import Interval

PEGGED = frozenset(
    {
        "BFUSDUSDT", "RLUSDUSDT", "RWUSDUSDT", "USDEUSDT", "UUSDT", "XUSDUSDT", "USD1USDT",
        "FDUSDUSDT", "USDCUSDT", "EURIUSDT", "EURUSDT", "PAXGUSDT", "XAUTUSDT", "WBTCUSDT",
        "WBETHUSDT",
    }
)  # fmt: skip
"""Stablecoins, fiat, gold and wrapped assets: no independent price action."""

STOCKS = frozenset(
    {
        "SPYBUSDT", "AAPLBUSDT", "QQQBUSDT", "GOOGLBUSDT", "NVDABUSDT", "METABUSDT", "BABABUSDT",
        "TSLABUSDT", "TQQQBUSDT", "SPCXBUSDT", "EWYBUSDT", "AMDBUSDT", "COINBUSDT", "KSTRBUSDT",
        "MSTRBUSDT", "INTCBUSDT", "MUBUSDT", "DRAMBUSDT", "MRVLBUSDT", "CRCLBUSDT", "SNDKBUSDT",
        "CRWVBUSDT", "SKHYBUSDT", "BNCBUSDT", "LITEBUSDT", "NBISBUSDT", "SOXSBUSDT", "SOXLBUSDT",
        "INTWBUSDT", "SNXXBUSDT", "KORUBUSDT",
    }
)  # fmt: skip
"""Tokenised stocks and ETFs (Binance "...B" tickers): they follow stock-market hours."""

EXCLUDED = PEGGED | STOCKS

POOL_LIST = "top200.txt"
EXTRA_LISTS = ("delisted.txt", "outside_pool.txt")


class Universe(StrEnum):
    POOL = "pool"
    ALL = "all"


def read_symbol_list(path: Path) -> list[str]:
    """Symbols in a whitespace-separated list file, in file order, duplicates dropped."""
    return list(dict.fromkeys(path.read_text("utf-8").split()))


def universe_symbols(
    universe: Universe | str, root: Path, *, interval: Interval = Interval.M5
) -> list[str]:
    """The symbols of `universe`, sorted. Raises `FileNotFoundError` when a list is missing."""
    universe = Universe(universe)
    lists = [POOL_LIST] if universe is Universe.POOL else [POOL_LIST, *EXTRA_LISTS]
    store = KlineStore(root)
    symbols: set[str] = set()
    for name in lists:
        symbols.update(read_symbol_list(root / name))
    return sorted(
        s for s in symbols if s not in EXCLUDED and store.series_dir(s, interval).is_dir()
    )
