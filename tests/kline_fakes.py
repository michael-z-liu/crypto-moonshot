"""A tiny in-memory Binance for CLI tests: hourly candles per symbol, no network."""

from collections.abc import Callable, Iterable
from datetime import UTC, datetime

import polars as pl

from crypto_moonshot.data.binance import BinanceError, ClientStats, RateLimitHook
from crypto_moonshot.models import KLINE_SCHEMA, Interval


def candle_frame(times: Iterable[datetime]) -> pl.DataFrame:
    rows = list(times)
    n = len(rows)
    return pl.DataFrame(
        {
            "open_time": rows,
            "open": [1.0] * n,
            "high": [2.0] * n,
            "low": [0.5] * n,
            "close": [1.5] * n,
            "volume": [10.0] * n,
            "quote_volume": [15.0] * n,
            "trade_count": [3] * n,
            "taker_buy_base_volume": [5.0] * n,
            "taker_buy_quote_volume": [7.5] * n,
        },
        schema=KLINE_SCHEMA,
    )


class FakeBinance:
    """Implements `KlineSource`. Symbols without data have no candles; `errors` raise instead."""

    def __init__(
        self,
        data: dict[str, list[datetime]] | None = None,
        *,
        errors: dict[str, BaseException] | None = None,
    ) -> None:
        self.data = data or {}
        self.errors = errors or {}
        self.stats = ClientStats()
        self.on_rate_limit: RateLimitHook | None = None
        self.closed = False
        self.fetches: list[tuple[str, Interval, datetime, datetime]] = []

    async def aclose(self) -> None:
        self.closed = True

    async def first_open_time(self, symbol: str, interval: Interval) -> datetime | None:
        self.stats.requests += 1
        if symbol in self.errors:
            raise self.errors[symbol]
        times = self.data.get(symbol, [])
        return times[0] if times else None

    async def fetch_klines(
        self,
        symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
        *,
        on_page: Callable[[int], None] | None = None,
    ) -> pl.DataFrame:
        self.fetches.append((symbol, interval, start, end))
        self.stats.requests += 1
        times = [t for t in self.data.get(symbol, []) if start <= t < end]
        if on_page is not None:
            on_page(len(times))
        return candle_frame(times)


def hours(start: datetime, count: int) -> list[datetime]:
    return list(Interval.H1.open_times(start, start + count * Interval.H1.length))


def unknown_symbol() -> BinanceError:
    return BinanceError(
        "HTTP 400, code -1121: Invalid symbol.", status=400, code=-1121, msg="Invalid symbol."
    )


JAN_1 = datetime(2024, 1, 1, tzinfo=UTC)
