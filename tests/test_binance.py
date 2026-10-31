import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx
import polars as pl
import pytest

from crypto_moonshot.config import Settings
from crypto_moonshot.data.binance import (
    BinanceClient,
    BinanceError,
    IpBannedError,
    RateLimitAbortError,
    RateLimitHit,
    klines_to_frame,
)
from crypto_moonshot.models import KLINE_SCHEMA, Interval

pytestmark = pytest.mark.anyio

T0 = datetime(2024, 1, 1, tzinfo=UTC)
BASE_URL = "https://binance.test"


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def row(
    open_time: datetime, interval: Interval = Interval.H1, price: float = 100.0
) -> list[object]:
    close_time = interval.next(open_time) - timedelta(milliseconds=1)
    return [
        to_ms(open_time),
        f"{price}",
        f"{price + 2}",
        f"{price - 1}",
        f"{price + 1}",
        "10.5",
        to_ms(close_time),
        "1050.25",
        42,
        "5.25",
        "525.125",
        "0",
    ]


class FakeClock:
    """A controllable clock whose `sleep` advances it, recording every wait."""

    def __init__(self, now: datetime) -> None:
        self.current = now
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.current

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.current += timedelta(seconds=seconds)


class FakeBinance:
    """Serves klines for one interval from a list of open times, like the real endpoint."""

    def __init__(
        self,
        open_times: list[datetime],
        interval: Interval = Interval.H1,
        clock: FakeClock | None = None,
    ) -> None:
        self.open_times = open_times
        self.interval = interval
        self.clock = clock
        self.requests: list[httpx.Request] = []
        self.weight_per_minute: dict[datetime, int] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        params = request.url.params
        start = int(params["startTime"])
        end = int(params["endTime"]) if "endTime" in params else None
        limit = int(params["limit"])
        selected = [
            t for t in self.open_times if to_ms(t) >= start and (end is None or to_ms(t) <= end)
        ][:limit]
        headers = {}
        if self.clock is not None:
            minute = self.clock.now().replace(second=0, microsecond=0)
            used = self.weight_per_minute.get(minute, 0) + 2
            self.weight_per_minute[minute] = used
            headers["X-MBX-USED-WEIGHT-1M"] = str(used)
        return httpx.Response(200, json=[row(t, self.interval) for t in selected], headers=headers)

    def param(self, index: int, name: str) -> int:
        return int(self.requests[index].url.params[name])


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    clock: FakeClock,
    **kwargs: object,
) -> BinanceClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    return BinanceClient(http, now=clock.now, sleep=clock.sleep, **kwargs)  # type: ignore[arg-type]


def hourly(count: int, start: datetime = T0) -> list[datetime]:
    return [start + timedelta(hours=i) for i in range(count)]


FAR_FUTURE = datetime(2030, 1, 1, tzinfo=UTC)


def test_klines_to_frame_schema_and_values() -> None:
    frame = klines_to_frame([row(T0, price=100.0)])
    assert frame.schema == KLINE_SCHEMA
    assert frame.columns == [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    assert frame.row(0, named=True) == {
        "open_time": T0,
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 10.5,
        "quote_volume": 1050.25,
        "trade_count": 42,
        "taker_buy_base_volume": 5.25,
        "taker_buy_quote_volume": 525.125,
    }
    assert frame["open_time"][0].utcoffset() == timedelta(0)


def test_klines_to_frame_rejects_short_rows() -> None:
    with pytest.raises(BinanceError, match="malformed"):
        klines_to_frame([[1, "2"]])


async def test_fetch_klines_pages_until_a_short_page() -> None:
    clock = FakeClock(FAR_FUTURE)
    server = FakeBinance(hourly(2500))
    client = make_client(server, clock)
    pages: list[int] = []
    end = T0 + timedelta(hours=3000)

    frame = await client.fetch_klines("BTCUSDT", Interval.H1, T0, end, on_page=pages.append)

    assert frame.height == 2500
    assert frame.schema == KLINE_SCHEMA
    assert frame["open_time"].is_sorted()
    assert frame["open_time"].n_unique() == 2500
    assert frame["open_time"][0] == T0
    assert frame["open_time"][-1] == T0 + timedelta(hours=2499)
    assert pages == [1000, 1000, 500]
    assert len(server.requests) == 3
    # The next page starts at the candle after the last one received.
    assert server.param(0, "startTime") == to_ms(T0)
    assert server.param(1, "startTime") == to_ms(T0 + timedelta(hours=1000))
    assert server.param(2, "startTime") == to_ms(T0 + timedelta(hours=2000))
    for i in range(3):
        assert server.param(i, "limit") == 1000
        assert server.param(i, "endTime") == to_ms(end) - 1
        assert server.requests[i].url.params["symbol"] == "BTCUSDT"
        assert server.requests[i].url.params["interval"] == "1h"
        assert server.requests[i].url.path == "/api/v3/klines"
        assert str(server.requests[i].url).startswith(BASE_URL)
    assert client.stats.requests == 3


async def test_fetch_klines_exact_page_multiple_ends_on_an_empty_page() -> None:
    clock = FakeClock(FAR_FUTURE)
    server = FakeBinance(hourly(2000))
    client = make_client(server, clock)
    frame = await client.fetch_klines("BTCUSDT", Interval.H1, T0, T0 + timedelta(hours=5000))
    assert frame.height == 2000
    assert len(server.requests) == 3
    assert server.param(2, "startTime") == to_ms(T0 + timedelta(hours=2000))


async def test_fetch_klines_excludes_candles_at_or_after_end() -> None:
    clock = FakeClock(FAR_FUTURE)
    server = FakeBinance(hourly(100))
    client = make_client(server, clock)
    end = T0 + timedelta(hours=10)
    frame = await client.fetch_klines("BTCUSDT", Interval.H1, T0, end)
    assert frame.height == 10
    assert frame["open_time"][-1] == end - timedelta(hours=1)


async def test_fetch_klines_drops_the_still_forming_candle() -> None:
    now = T0 + timedelta(hours=5, minutes=30)  # the 05:00 candle is still open
    clock = FakeClock(now)
    server = FakeBinance(hourly(6))  # 00:00 .. 05:00
    client = make_client(server, clock)
    frame = await client.fetch_klines("BTCUSDT", Interval.H1, T0, T0 + timedelta(days=1))
    assert frame["open_time"].to_list() == hourly(5)

    clock.current = T0 + timedelta(hours=6)  # exactly closed now
    frame = await client.fetch_klines("BTCUSDT", Interval.H1, T0, T0 + timedelta(days=1))
    assert frame.height == 6


async def test_fetch_klines_drops_the_forming_monthly_candle() -> None:
    months = [datetime(2024, m, 1, tzinfo=UTC) for m in (1, 2, 3)]
    clock = FakeClock(datetime(2024, 3, 15, tzinfo=UTC))
    server = FakeBinance(months, Interval.MO1)
    client = make_client(server, clock)
    frame = await client.fetch_klines("BTCUSDT", Interval.MO1, months[0], FAR_FUTURE)
    assert frame["open_time"].to_list() == months[:2]


async def test_fetch_klines_empty_result() -> None:
    clock = FakeClock(FAR_FUTURE)
    server = FakeBinance([])
    client = make_client(server, clock)
    pages: list[int] = []
    frame = await client.fetch_klines(
        "BTCUSDT", Interval.H1, T0, T0 + timedelta(days=1), on_page=pages.append
    )
    assert frame.is_empty()
    assert frame.schema == KLINE_SCHEMA
    assert pages == [0]
    assert len(server.requests) == 1


async def test_fetch_klines_with_nothing_to_fetch_makes_no_request() -> None:
    clock = FakeClock(FAR_FUTURE)
    server = FakeBinance(hourly(5))
    client = make_client(server, clock)
    frame = await client.fetch_klines("BTCUSDT", Interval.H1, T0, T0)
    assert frame.is_empty()
    assert server.requests == []


async def test_fetch_klines_rejects_naive_datetimes() -> None:
    client = make_client(FakeBinance([]), FakeClock(FAR_FUTURE))
    with pytest.raises(ValueError, match="timezone-aware"):
        await client.fetch_klines("BTCUSDT", Interval.H1, datetime(2024, 1, 1), T0)  # noqa: DTZ001


async def test_first_open_time() -> None:
    listing = datetime(2020, 8, 11, 6, tzinfo=UTC)
    server = FakeBinance(hourly(10, listing))
    client = make_client(server, FakeClock(FAR_FUTURE))
    assert await client.first_open_time("SOLUSDT", Interval.H1) == listing
    params = server.requests[0].url.params
    assert params["startTime"] == "0"
    assert params["limit"] == "1"
    assert "endTime" not in params


async def test_first_open_time_is_none_without_candles() -> None:
    client = make_client(FakeBinance([]), FakeClock(FAR_FUTURE))
    assert await client.first_open_time("SOLUSDT", Interval.H1) is None


async def test_client_without_base_url_on_http_uses_base_url_argument() -> None:
    server = FakeBinance(hourly(1))
    http = httpx.AsyncClient(transport=httpx.MockTransport(server))
    client = BinanceClient(http, base_url="https://example.test/", now=lambda: FAR_FUTURE)
    await client.first_open_time("BTCUSDT", Interval.H1)
    assert str(server.requests[0].url).startswith("https://example.test/api/v3/klines?")


async def test_from_settings_uses_configured_host() -> None:
    client = BinanceClient.from_settings(Settings(binance_base_url="http://localhost:1"))
    assert str(client._http.base_url) == "http://localhost:1"
    await client.aclose()


# -- weight budget ------------------------------------------------------------------------


async def test_pauses_until_the_next_minute_when_the_budget_would_be_exceeded() -> None:
    clock = FakeClock(datetime(2024, 5, 1, 10, 0, 30, tzinfo=UTC))
    server = FakeBinance(hourly(3), clock=clock)
    client = make_client(server, clock, weight_budget=10)

    for _ in range(5):  # used weight reaches 10 == budget: allowed
        await client.first_open_time("BTCUSDT", Interval.H1)
    assert clock.sleeps == []
    assert client.stats.max_used_weight == 10

    await client.first_open_time("BTCUSDT", Interval.H1)  # would be 12 > 10
    assert clock.sleeps == [pytest.approx(30.5)]  # 30s to 10:01:00 plus a small margin
    assert len(server.requests) == 6
    assert client.stats.max_used_weight == 10  # the new minute started from zero
    assert clock.now() >= datetime(2024, 5, 1, 10, 1, tzinfo=UTC)


async def test_weight_from_a_previous_minute_is_ignored() -> None:
    clock = FakeClock(datetime(2024, 5, 1, 10, 0, 10, tzinfo=UTC))
    server = FakeBinance(hourly(3), clock=clock)
    client = make_client(server, clock, weight_budget=10)
    for _ in range(5):
        await client.first_open_time("BTCUSDT", Interval.H1)
    clock.current += timedelta(minutes=1)  # a new window; no sleep needed
    await client.first_open_time("BTCUSDT", Interval.H1)
    assert clock.sleeps == []


async def test_in_flight_requests_count_against_the_budget() -> None:
    clock = FakeClock(datetime(2024, 5, 1, 10, 0, 59, tzinfo=UTC))
    server = FakeBinance(hourly(3), clock=clock)
    client = make_client(server, clock, weight_budget=6)

    results = await asyncio.gather(
        *(client.first_open_time("BTCUSDT", Interval.H1) for _ in range(8))
    )

    assert len(results) == 8
    assert clock.sleeps, "expected at least one pause for the next minute"
    assert client.stats.max_used_weight <= 6
    assert max(server.weight_per_minute.values()) <= 6


async def test_budget_must_fit_one_request() -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(FakeBinance([])))
    with pytest.raises(ValueError, match="weight_budget"):
        BinanceClient(http, weight_budget=1)


# -- 429 / 418 / errors ------------------------------------------------------------------


def sequence_handler(
    responses: list[httpx.Response | Exception],
) -> tuple[Callable[[httpx.Request], httpx.Response], list[httpx.Request]]:
    """Return the given responses in order (the last one repeats); raise exceptions."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = responses[min(len(seen) - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    return handler, seen


def ok() -> httpx.Response:
    return httpx.Response(200, json=[row(T0)])


async def test_429_waits_retry_after_then_retries() -> None:
    handler, seen = sequence_handler(
        [httpx.Response(429, headers={"Retry-After": "7"}, json={}), ok()]
    )
    clock = FakeClock(FAR_FUTURE)
    client = make_client(handler, clock)
    assert await client.first_open_time("BTCUSDT", Interval.H1) == T0
    assert clock.sleeps == [7.0]
    assert len(seen) == 2
    assert client.stats.rate_limit_hits == 1
    assert client.stats.requests == 2


async def test_429_without_retry_after_waits_a_minute() -> None:
    handler, _ = sequence_handler([httpx.Response(429, json={}), ok()])
    clock = FakeClock(FAR_FUTURE)
    client = make_client(handler, clock)
    await client.first_open_time("BTCUSDT", Interval.H1)
    assert clock.sleeps == [60.0]


async def test_429_pauses_other_concurrent_requests_too() -> None:
    handler, _ = sequence_handler(
        [httpx.Response(429, headers={"Retry-After": "5"}, json={}), ok()]
    )
    clock = FakeClock(FAR_FUTURE)
    client = make_client(handler, clock)
    await asyncio.gather(*(client.first_open_time("BTCUSDT", Interval.H1) for _ in range(3)))
    assert client.stats.rate_limit_hits >= 1
    assert 5.0 in clock.sleeps


async def test_aborts_after_max_rate_limit_hits_and_stays_aborted() -> None:
    handler, seen = sequence_handler([httpx.Response(429, headers={"Retry-After": "7"}, json={})])
    clock = FakeClock(FAR_FUTURE)
    client = make_client(handler, clock, max_rate_limit_hits=3)

    with pytest.raises(RateLimitAbortError):
        await client.first_open_time("BTCUSDT", Interval.H1)
    assert len(seen) == 3
    assert clock.sleeps == [7.0, 7.0]  # waits between hits, not after the last
    assert client.stats.rate_limit_hits == 3

    with pytest.raises(RateLimitAbortError):
        await client.first_open_time("ETHUSDT", Interval.H1)
    with pytest.raises(RateLimitAbortError):
        await client.fetch_klines("ETHUSDT", Interval.H1, T0, T0 + timedelta(days=1))
    assert len(seen) == 3  # no further requests
    assert clock.sleeps == [7.0, 7.0]


async def test_on_rate_limit_hook_reports_each_429() -> None:
    handler, _ = sequence_handler([httpx.Response(429, headers={"Retry-After": "7"}, json={})])
    clock = FakeClock(FAR_FUTURE)
    hits: list[RateLimitHit] = []
    client = make_client(handler, clock, max_rate_limit_hits=3, on_rate_limit=hits.append)

    with pytest.raises(RateLimitAbortError):
        await client.first_open_time("BTCUSDT", Interval.H1)
    assert hits == [
        RateLimitHit(hits=1, max_hits=3, retry_after=7.0, aborting=False),
        RateLimitHit(hits=2, max_hits=3, retry_after=7.0, aborting=False),
        RateLimitHit(hits=3, max_hits=3, retry_after=0.0, aborting=True),
    ]


async def test_on_rate_limit_hook_can_be_set_later_and_its_errors_are_ignored() -> None:
    handler, _ = sequence_handler(
        [httpx.Response(429, headers={"Retry-After": "5"}, json={}), ok()]
    )
    clock = FakeClock(FAR_FUTURE)
    client = make_client(handler, clock)
    calls: list[RateLimitHit] = []

    def hook(hit: RateLimitHit) -> None:
        calls.append(hit)
        raise RuntimeError("broken hook")

    client.on_rate_limit = hook
    assert await client.first_open_time("BTCUSDT", Interval.H1) == T0
    assert calls == [RateLimitHit(hits=1, max_hits=3, retry_after=5.0, aborting=False)]
    assert clock.sleeps == [5.0]


async def test_418_aborts_immediately_and_stays_banned() -> None:
    handler, seen = sequence_handler([httpx.Response(418, headers={"Retry-After": "600"}, json={})])
    clock = FakeClock(FAR_FUTURE)
    client = make_client(handler, clock)

    with pytest.raises(IpBannedError, match="418"):
        await client.first_open_time("BTCUSDT", Interval.H1)
    assert len(seen) == 1
    assert clock.sleeps == []

    with pytest.raises(IpBannedError):
        await client.first_open_time("BTCUSDT", Interval.H1)
    with pytest.raises(IpBannedError):
        await client.fetch_klines("BTCUSDT", Interval.H1, T0, T0 + timedelta(days=1))
    assert len(seen) == 1


async def test_5xx_is_retried_with_exponential_backoff() -> None:
    handler, seen = sequence_handler([httpx.Response(500), httpx.Response(503, text="busy"), ok()])
    clock = FakeClock(FAR_FUTURE)
    client = make_client(handler, clock)
    assert await client.first_open_time("BTCUSDT", Interval.H1) == T0
    assert len(seen) == 3
    assert clock.sleeps == [1.0, 2.0]
    assert client.stats.rate_limit_hits == 0


async def test_5xx_retries_are_exhausted() -> None:
    handler, seen = sequence_handler([httpx.Response(502)])
    clock = FakeClock(FAR_FUTURE)
    client = make_client(handler, clock)
    with pytest.raises(BinanceError, match="5 attempts") as excinfo:
        await client.first_open_time("BTCUSDT", Interval.H1)
    assert excinfo.value.status == 502
    assert len(seen) == 5
    assert clock.sleeps == [1.0, 2.0, 4.0, 8.0]
    # A server error does not poison the client.
    handler2, _ = sequence_handler([ok()])
    assert await make_client(handler2, clock).first_open_time("BTCUSDT", Interval.H1) == T0


async def test_network_errors_and_timeouts_are_retried() -> None:
    handler, seen = sequence_handler(
        [httpx.ConnectTimeout("slow"), httpx.ConnectError("refused"), ok()]
    )
    clock = FakeClock(FAR_FUTURE)
    client = make_client(handler, clock)
    assert await client.first_open_time("BTCUSDT", Interval.H1) == T0
    assert len(seen) == 3
    assert clock.sleeps == [1.0, 2.0]


async def test_persistent_network_errors_raise_binance_error() -> None:
    handler, seen = sequence_handler([httpx.ReadTimeout("slow")])
    clock = FakeClock(FAR_FUTURE)
    client = make_client(handler, clock, max_attempts=3)
    with pytest.raises(BinanceError, match="network error"):
        await client.first_open_time("BTCUSDT", Interval.H1)
    assert len(seen) == 3
    assert clock.sleeps == [1.0, 2.0]


async def test_unknown_symbol_carries_binance_code_and_message() -> None:
    handler, seen = sequence_handler(
        [httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})]
    )
    clock = FakeClock(FAR_FUTURE)
    client = make_client(handler, clock)
    with pytest.raises(BinanceError) as excinfo:
        await client.fetch_klines("NOPE", Interval.H1, T0, T0 + timedelta(days=1))
    err = excinfo.value
    assert not isinstance(err, IpBannedError | RateLimitAbortError)
    assert err.code == -1121
    assert err.msg == "Invalid symbol."
    assert err.status == 400
    assert "-1121" in str(err)
    assert "Invalid symbol." in str(err)
    assert len(seen) == 1  # no retry
    assert clock.sleeps == []
    # The client keeps working for other symbols.
    assert client.stats.rate_limit_hits == 0


async def test_non_json_4xx_still_raises_binance_error() -> None:
    handler, seen = sequence_handler([httpx.Response(403, text="forbidden by WAF")])
    client = make_client(handler, FakeClock(FAR_FUTURE))
    with pytest.raises(BinanceError, match="forbidden by WAF") as excinfo:
        await client.first_open_time("BTCUSDT", Interval.H1)
    assert excinfo.value.code is None
    assert len(seen) == 1


async def test_stats_track_requests_and_max_used_weight() -> None:
    handler, _ = sequence_handler(
        [
            httpx.Response(200, json=[row(T0)], headers={"X-MBX-USED-WEIGHT-1M": "40"}),
            httpx.Response(200, json=[row(T0)], headers={"X-MBX-USED-WEIGHT-1M": "42"}),
        ]
    )
    client = make_client(handler, FakeClock(FAR_FUTURE))
    await client.first_open_time("BTCUSDT", Interval.H1)
    await client.first_open_time("BTCUSDT", Interval.H1)
    assert client.stats.requests == 2
    assert client.stats.max_used_weight == 42


def test_empty_frame_dtypes_match_schema() -> None:
    assert pl.DataFrame(schema=KLINE_SCHEMA).schema == KLINE_SCHEMA
