"""A `BinanceBroker` wired to the fake exchange (`fake_exchange.FakeExchange`): REST through an
`httpx.MockTransport`, the user data stream through the fake's WebSocket server on 127.0.0.1.
Nothing reaches a real host: the REST base URL is under `.invalid` (reserved, never resolves)
and the transport is a mock."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

import httpx
from fake_exchange import API_KEY, API_SECRET, FakeExchange
from pydantic import SecretStr

from crypto_moonshot.broker.binance import BinanceBroker
from crypto_moonshot.exchange.client import SignedClient
from crypto_moonshot.exchange.ws_api import StreamMessage, UserDataStream

REST_URL = "https://testnet.invalid"
T0 = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
D = Decimal


class Clock:
    """A settable aware-UTC clock (callable)."""

    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float = 0, **kwargs: float) -> None:
        self.now += timedelta(seconds=seconds, **kwargs)


async def fast_sleep(seconds: float) -> None:
    """Waits a little real time whatever is asked (reconnect backoff, rate-limit pauses)."""
    await asyncio.sleep(min(seconds, 0.01))


async def until(check: Callable[[], bool], timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not check():
        if loop.time() > deadline:
            msg = "condition not reached"
            raise TimeoutError(msg)
        await asyncio.sleep(0.002)


@dataclass
class BrokerRig:
    clock: Clock
    fake: FakeExchange
    client: SignedClient
    broker: BinanceBroker
    ws_url: str | None

    @property
    def stream(self) -> UserDataStream:
        stream = self.broker._stream
        assert stream is not None
        return stream

    async def delivered(self) -> None:
        """Wait until the stream has received every event the fake delivered."""
        if self.ws_url is None or not self.broker.stream_connected:
            return
        await self.fake.flushed()
        await until(lambda: self.stream.received >= self.fake.sent)
        await asyncio.sleep(0.002)

    def hosts(self) -> set[str]:
        """Every host the broker talked to (REST requests and the WebSocket URL)."""
        out = {r.url.host for r in self.fake.requests}
        if self.ws_url is not None:
            out.add(urlparse(self.ws_url).hostname or "")
        return out

    async def close(self) -> None:
        await self.broker.aclose()
        await self.client.aclose()
        await self.fake.stop_ws()


async def make_rig(
    *,
    stream: bool = True,
    clock: Clock | None = None,
    fake: FakeExchange | None = None,
    start: bool = True,
    **broker_kwargs: Any,
) -> BrokerRig:
    clock = clock or Clock()
    fake = fake or FakeExchange(clock)
    fake.sent = fake.delivered = 0  # counted per stream client
    url = await fake.start_ws() if stream else None
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake))
    client = SignedClient(
        SecretStr(API_KEY),
        SecretStr(API_SECRET),
        base_url=REST_URL,
        http=http,
        now=clock,
        sleep=fast_sleep,
        backoff_base=0.0,
    )

    def factory(sink: Callable[[StreamMessage], None]) -> UserDataStream:
        assert url is not None
        return UserDataStream(
            SecretStr(API_KEY),
            SecretStr(API_SECRET),
            sink,
            url=url,
            now=clock,
            sleep=fast_sleep,
            backoff_base=0.005,
            backoff_max=0.02,
        )

    options: dict[str, Any] = {
        "stream": factory if stream else None,
        "clock": clock,
        "poll_interval_down": timedelta(0),
    }
    options.update(broker_kwargs)
    broker = BinanceBroker(client, **options)
    rig = BrokerRig(clock, fake, client, broker, url)
    if start:
        await broker.start()
        if stream:
            await until(lambda: broker.stream_connected)
    return rig
