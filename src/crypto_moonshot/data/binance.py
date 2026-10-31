"""Async Binance spot klines client with rate-limit safety.

Only public market data is fetched, so no API key is needed (or ever sent). One client shares a
weight budget, a rate-limit hit counter and a ban flag across all of its requests, and is safe
to use from many asyncio tasks at once.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any, Self

import httpx
import polars as pl

from crypto_moonshot.config import Settings
from crypto_moonshot.models import EPOCH, KLINE_SCHEMA, Interval, as_utc

KLINES_PATH = "/api/v3/klines"
KLINES_WEIGHT = 2
"""Request weight of `GET /api/v3/klines` with a limit of up to 1000 rows."""
PAGE_LIMIT = 1000
IP_WEIGHT_LIMIT_1M = 6000
"""Binance's request weight limit per IP per minute."""
DEFAULT_WEIGHT_BUDGET = IP_WEIGHT_LIMIT_1M // 2
USED_WEIGHT_HEADER = "X-MBX-USED-WEIGHT-1M"
DEFAULT_RETRY_AFTER_S = 60.0
_MINUTE_MARGIN_S = 0.5
"""Extra wait past the minute boundary, so the weight window has surely reset."""
_MAX_BACKOFF_S = 60.0


class BinanceError(Exception):
    """A request to Binance failed. `code` and `msg` are Binance's, when it sent them."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: int | None = None,
        msg: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.msg = msg


class RateLimitAbortError(BinanceError):
    """Too many HTTP 429 responses; the client refuses all further requests."""


class IpBannedError(BinanceError):
    """HTTP 418: the IP is banned. The client refuses all further requests."""


@dataclass
class ClientStats:
    """Counters for run summaries."""

    requests: int = 0
    """HTTP requests sent, including retries."""
    rate_limit_hits: int = 0
    """HTTP 429 responses received."""
    max_used_weight: int = 0
    """Highest `X-MBX-USED-WEIGHT-1M` seen."""


Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _from_ms(ms: int) -> datetime:
    return EPOCH + timedelta(milliseconds=ms)


def _to_ms(dt: datetime) -> int:
    return (as_utc(dt) - EPOCH) // timedelta(milliseconds=1)


def klines_to_frame(rows: Sequence[Sequence[Any]]) -> pl.DataFrame:
    """Convert Binance kline arrays to a frame in the stored schema (`KLINE_SCHEMA`).

    Each row is `[openTime, open, high, low, close, volume, closeTime, quoteVolume, trades,
    takerBuyBase, takerBuyQuote, ignore]`, with times in ms and prices as strings.
    """
    columns: dict[str, list[Any]] = {name: [] for name in KLINE_SCHEMA}
    for row in rows:
        if len(row) < 11:
            msg = f"malformed kline row with {len(row)} fields: {row!r}"
            raise BinanceError(msg)
        columns["open_time"].append(_from_ms(int(row[0])))
        columns["open"].append(float(row[1]))
        columns["high"].append(float(row[2]))
        columns["low"].append(float(row[3]))
        columns["close"].append(float(row[4]))
        columns["volume"].append(float(row[5]))
        columns["quote_volume"].append(float(row[7]))
        columns["trade_count"].append(int(row[8]))
        columns["taker_buy_base_volume"].append(float(row[9]))
        columns["taker_buy_quote_volume"].append(float(row[10]))
    return pl.DataFrame(columns, schema=KLINE_SCHEMA)


def empty_klines() -> pl.DataFrame:
    return pl.DataFrame(schema=KLINE_SCHEMA)


class BinanceClient:
    """Fetches closed klines, pacing itself to stay well under Binance's IP limits.

    Args:
        http: The `httpx.AsyncClient` to use (tests pass one with a `MockTransport`). If it has
            no `base_url`, pass `base_url` here.
        base_url: Prefix for request URLs, when `http` has none.
        now: Clock, returning an aware UTC datetime.
        sleep: Awaitable sleep, injectable so tests do not wait.
        weight_budget: Used weight per minute to stay under (default: 50% of the 6,000 limit).
        max_rate_limit_hits: HTTP 429 responses tolerated before aborting for good.
        max_attempts: Attempts per request on 5xx and network errors.
        backoff_base: Seconds before the first retry; doubles each time.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        base_url: str | None = None,
        now: Clock = _utc_now,
        sleep: Sleep = asyncio.sleep,
        weight_budget: int = DEFAULT_WEIGHT_BUDGET,
        request_weight: int = KLINES_WEIGHT,
        max_rate_limit_hits: int = 3,
        max_attempts: int = 5,
        backoff_base: float = 1.0,
        _owns_http: bool = False,
    ) -> None:
        if weight_budget < request_weight:
            msg = "weight_budget must be at least the weight of one request"
            raise ValueError(msg)
        if max_attempts < 1 or max_rate_limit_hits < 1:
            msg = "max_attempts and max_rate_limit_hits must be at least 1"
            raise ValueError(msg)
        self._http = http
        self._url = (base_url.rstrip("/") if base_url else "") + KLINES_PATH
        self._now = now
        self._sleep = sleep
        self._weight_budget = weight_budget
        self._request_weight = request_weight
        self._max_rate_limit_hits = max_rate_limit_hits
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._owns_http = _owns_http
        self.stats = ClientStats()
        self._lock = asyncio.Lock()
        self._inflight_weight = 0
        self._used_weight = 0
        self._used_minute: datetime | None = None
        self._fatal: Callable[[], BinanceError] | None = None
        self._pause_until: datetime | None = None

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> Self:
        """Build a client (owning its `httpx.AsyncClient`) for the configured Binance host."""
        http = httpx.AsyncClient(base_url=settings.binance_base_url, timeout=30.0)
        return cls(http, _owns_http=True, **kwargs)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # -- public API -------------------------------------------------------------------------

    async def fetch_klines(
        self,
        symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
        *,
        on_page: Callable[[int], None] | None = None,
    ) -> pl.DataFrame:
        """All closed candles with open time in `[start, end)`, sorted by open time.

        Pages through the API 1,000 candles at a time. A candle that has not closed yet
        (`open_time + interval > now`) is dropped. `on_page` is called with the number of
        candles each page returned, before filtering.
        """
        start, end = as_utc(start), as_utc(end)
        end_ms = _to_ms(end)
        frames: list[pl.DataFrame] = []
        cursor = start
        while cursor < end:
            rows = await self._get_rows(
                {
                    "symbol": symbol,
                    "interval": interval.value,
                    "startTime": _to_ms(cursor),
                    "endTime": end_ms - 1,  # endTime is inclusive of the open time
                    "limit": PAGE_LIMIT,
                }
            )
            if on_page is not None:
                on_page(len(rows))
            if not rows:
                break
            page = klines_to_frame(rows)
            now = self._now()
            frames.append(page.filter(self._closed_before_end(interval, end, now)))
            last_open = page["open_time"][-1]
            if len(rows) < PAGE_LIMIT:
                break
            next_cursor = interval.next(last_open)
            if next_cursor <= cursor:  # pragma: no cover - defensive against a misbehaving server
                break
            cursor = next_cursor
        if not frames:
            return empty_klines()
        return pl.concat(frames).unique("open_time", keep="first", maintain_order=True)

    async def first_open_time(self, symbol: str, interval: Interval) -> datetime | None:
        """Open time of the symbol's first candle (its listing date), or None if there is none."""
        rows = await self._get_rows(
            {"symbol": symbol, "interval": interval.value, "startTime": 0, "limit": 1}
        )
        if not rows:
            return None
        return _from_ms(int(rows[0][0]))

    # -- internals --------------------------------------------------------------------------

    @staticmethod
    def _closed_before_end(interval: Interval, end: datetime, now: datetime) -> pl.Expr:
        """Rows that are in range and closed. Vectorised check where possible."""
        open_time = pl.col("open_time")
        if interval.is_monthly:
            return (open_time < end) & (open_time.dt.offset_by("1mo") <= now)
        return (open_time < end) & (open_time + interval.length <= now)

    async def _get_rows(self, params: Mapping[str, str | int]) -> list[list[Any]]:
        payload = await self._request(params)
        if not isinstance(payload, list):
            msg = f"unexpected klines response: {str(payload)[:200]}"
            raise BinanceError(msg)
        return payload

    def _check_fatal(self) -> None:
        if self._fatal is not None:
            raise self._fatal()

    def _window_start(self, now: datetime) -> datetime:
        return now.replace(second=0, microsecond=0)

    def _used_in_window(self, now: datetime) -> int:
        if self._used_minute != self._window_start(now):
            return 0
        return self._used_weight

    async def _reserve(self, weight: int) -> None:
        """Wait, if needed, until `weight` fits in the budget, then count it as in flight."""
        async with self._lock:
            self._check_fatal()
            if self._pause_until is not None:  # a 429 told every request to wait
                remaining = (self._pause_until - self._now()).total_seconds()
                self._pause_until = None
                await self._sleep(remaining)
                self._check_fatal()
            while True:
                now = self._now()
                used = self._used_in_window(now)
                if used + self._inflight_weight + weight <= self._weight_budget:
                    break
                wait = (
                    self._window_start(now) + timedelta(minutes=1) - now
                ).total_seconds() + _MINUTE_MARGIN_S
                await self._sleep(wait)
                self._check_fatal()
                self._used_weight, self._used_minute = 0, None  # the window has reset
            self._inflight_weight += weight

    def _release(self, weight: int, response: httpx.Response | None) -> None:
        self._inflight_weight -= weight
        if response is None:
            return
        header = response.headers.get(USED_WEIGHT_HEADER)
        if header is None:
            return
        try:
            used = int(header)
        except ValueError:
            return
        window = self._window_start(self._now())
        if self._used_minute == window:
            used = max(used, self._used_weight)
        self._used_weight, self._used_minute = used, window
        self.stats.max_used_weight = max(self.stats.max_used_weight, used)

    async def _request(self, params: Mapping[str, str | int]) -> Any:
        weight = self._request_weight
        attempt = 0
        while True:
            self._check_fatal()
            await self._reserve(weight)
            response: httpx.Response | None = None
            try:
                self.stats.requests += 1
                response = await self._http.get(self._url, params=params)
            except httpx.TransportError as exc:
                failure = f"network error: {exc!r}"
                status = None
            else:
                failure = ""
                status = response.status_code
            finally:
                self._release(weight, response)

            if response is not None and status is not None:
                if status == 200:
                    return response.json()
                if status == 418:
                    self._ban(response)
                if status == 429:
                    self._handle_rate_limit(response)
                    continue
                if status < 500:
                    raise self._api_error(response)
                failure = f"HTTP {status}"

            attempt += 1
            if attempt >= self._max_attempts:
                msg = f"giving up after {attempt} attempts: {failure}"
                raise BinanceError(msg, status=status)
            delay = min(self._backoff_base * 2 ** (attempt - 1), _MAX_BACKOFF_S)
            await self._sleep(delay)

    def _ban(self, response: httpx.Response) -> None:
        retry_after = response.headers.get("Retry-After")
        until = f" (Retry-After: {retry_after}s)" if retry_after else ""
        msg = f"HTTP 418: IP banned by Binance{until}; stop all requests"
        self._fatal = lambda: IpBannedError(msg, status=418)
        raise IpBannedError(msg, status=418)

    def _handle_rate_limit(self, response: httpx.Response) -> None:
        self.stats.rate_limit_hits += 1
        if self.stats.rate_limit_hits >= self._max_rate_limit_hits:
            msg = f"aborting after {self.stats.rate_limit_hits} rate-limit (HTTP 429) responses"
            self._fatal = lambda: RateLimitAbortError(msg, status=429)
            raise RateLimitAbortError(msg, status=429)
        try:
            retry_after = float(response.headers["Retry-After"])
        except (KeyError, ValueError):
            retry_after = DEFAULT_RETRY_AFTER_S
        # Pause every request of this client, not just the one that was limited.
        until = self._now() + timedelta(seconds=retry_after)
        if self._pause_until is None or until > self._pause_until:
            self._pause_until = until

    @staticmethod
    def _api_error(response: httpx.Response) -> BinanceError:
        code: int | None = None
        text = response.text[:200]
        api_msg: str | None = None
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict):
            raw_code = body.get("code")
            code = raw_code if isinstance(raw_code, int) else None
            text = api_msg = str(body.get("msg", text))
        message = f"HTTP {response.status_code}" + (f", code {code}" if code is not None else "")
        return BinanceError(
            f"{message}: {text}", status=response.status_code, code=code, msg=api_msg
        )
