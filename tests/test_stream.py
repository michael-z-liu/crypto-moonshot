import asyncio
from datetime import UTC, datetime

import pytest
from stream_fakes import FakeStreamServer, kline_message, until

from crypto_moonshot.data.stream import (
    ClosedCandle,
    Connected,
    Disconnected,
    KlineStream,
    StreamConfig,
    StreamEvent,
    SubscriptionError,
    parse_kline_event,
)
from crypto_moonshot.models import Interval

pytestmark = pytest.mark.anyio

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def config(url: str, **kwargs: object) -> StreamConfig:
    base: dict[str, object] = {
        "base_url": url,
        "subscribe_gap": 0.0,
        "backoff_base": 0.01,
        "backoff_max": 0.05,
    }
    return StreamConfig(**{**base, **kwargs})  # type: ignore[arg-type]


async def collect(stream: KlineStream, sink: list[StreamEvent]) -> None:
    async for event in stream.events():
        sink.append(event)


def test_parse_closed_kline() -> None:
    parsed = parse_kline_event(kline_message("BTCUSDT", Interval.M5, T0, price=10.0))
    assert parsed is not None
    symbol, interval, candle = parsed
    assert (symbol, interval) == ("BTCUSDT", Interval.M5)
    assert candle.open_time == T0
    assert (candle.open, candle.high, candle.low, candle.close) == (10.0, 12.0, 9.0, 11.0)
    assert (candle.volume, candle.quote_volume, candle.trade_count) == (10.5, 1050.25, 42)
    assert (candle.taker_buy_base_volume, candle.taker_buy_quote_volume) == (5.25, 525.125)


def test_parse_ignores_forming_acks_and_junk() -> None:
    assert parse_kline_event(kline_message("BTCUSDT", Interval.M5, T0, closed=False)) is None
    assert parse_kline_event({"result": None, "id": 1}) is None
    assert parse_kline_event({"data": {"e": "trade"}}) is None
    assert parse_kline_event({"data": {"e": "kline", "k": {"x": True}}}) is None
    assert parse_kline_event([1, 2]) is None


async def test_streams_closed_candles_only() -> None:
    async with FakeStreamServer() as server:
        stream = KlineStream([("btcusdt", Interval.M5)], config(server.url))
        events: list[StreamEvent] = []
        task = asyncio.create_task(collect(stream, events))
        await until(lambda: len(server.connections) == 1 and server.subscribed(0) != [])
        await server.send(
            kline_message("BTCUSDT", Interval.M5, T0, closed=False),
            kline_message("BTCUSDT", Interval.M5, T0),
        )
        await until(lambda: any(isinstance(e, ClosedCandle) for e in events))
        await stream.aclose()
        await task
    closed = [e for e in events if isinstance(e, ClosedCandle)]
    assert len(closed) == 1
    assert closed[0].symbol == "BTCUSDT"
    assert closed[0].candle.open_time == T0
    assert stream.last_message_at is not None
    assert isinstance(events[0], Connected)
    assert not events[0].reconnect


async def test_chunks_subscriptions_and_splits_connections() -> None:
    pairs = [(f"C{i}USDT", Interval.M5) for i in range(5)]
    async with FakeStreamServer() as server:
        stream = KlineStream(pairs, config(server.url, chunk_size=2, max_streams=3))
        assert stream.connection_count == 2
        events: list[StreamEvent] = []
        task = asyncio.create_task(collect(stream, events))
        await until(lambda: len(server.received) == 2 and sum(map(len, server.received)) == 3)
        await stream.aclose()
        await task
    by_size = sorted(server.received, key=lambda msgs: -len(msgs))
    # 3 streams in chunks of 2 + 1, 2 streams in one message
    assert [len(m["params"]) for m in by_size[0]] == [2, 1]
    assert [len(m["params"]) for m in by_size[1]] == [2]
    names = sorted(server.subscribed(0) + server.subscribed(1))
    assert names == sorted(f"c{i}usdt@kline_5m" for i in range(5))
    assert all(m["method"] == "SUBSCRIBE" for msgs in server.received for m in msgs)
    assert sorted(m["id"] for m in by_size[0]) == [1, 2]


async def test_subscription_error_is_reported() -> None:
    async with FakeStreamServer() as server:
        server.reject_subscriptions = True
        stream = KlineStream([("BTCUSDT", Interval.M5)], config(server.url))
        events: list[StreamEvent] = []
        task = asyncio.create_task(collect(stream, events))
        await until(lambda: any(isinstance(e, SubscriptionError) for e in events))
        await stream.aclose()
        await task
    error = next(e for e in events if isinstance(e, SubscriptionError))
    assert "Invalid request" in error.message


async def test_reconnects_and_resubscribes_after_a_drop() -> None:
    async with FakeStreamServer() as server:
        stream = KlineStream([("BTCUSDT", Interval.M5)], config(server.url))
        events: list[StreamEvent] = []
        task = asyncio.create_task(collect(stream, events))
        await until(lambda: len(server.connections) == 1 and server.subscribed(0) != [])
        await server.drop()
        await until(lambda: len(server.connections) == 2 and server.subscribed(1) != [])
        await server.send(kline_message("BTCUSDT", Interval.M5, T0))
        await until(lambda: any(isinstance(e, ClosedCandle) for e in events))
        await stream.aclose()
        await task
    kinds = [type(e) for e in events]
    assert kinds[:3] == [Connected, Disconnected, Connected]
    reconnect = [e for e in events if isinstance(e, Connected)][1]
    assert reconnect.reconnect
    assert server.subscribed(0) == server.subscribed(1)


async def test_silent_connection_is_treated_as_dead() -> None:
    async with FakeStreamServer() as server:
        stream = KlineStream(
            [("BTCUSDT", Interval.M5)], config(server.url, silence_timeout=0.1, ping_interval=None)
        )
        events: list[StreamEvent] = []
        task = asyncio.create_task(collect(stream, events))
        await until(lambda: len(server.connections) >= 2)
        await stream.aclose()
        await task
    disconnected = [e for e in events if isinstance(e, Disconnected)]
    assert disconnected
    assert "no message" in disconnected[0].reason


async def test_unreachable_server_is_retried_and_reported_once_connected() -> None:
    stream = KlineStream([("BTCUSDT", Interval.M5)], config("ws://127.0.0.1:1"))
    events: list[StreamEvent] = []
    task = asyncio.create_task(collect(stream, events))
    await asyncio.sleep(0.15)
    await stream.aclose()
    await task
    assert events == []  # never connected: nothing to report, and no crash
