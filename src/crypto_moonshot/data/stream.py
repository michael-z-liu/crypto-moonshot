"""Asyncio client for Binance's kline WebSocket streams (public market data, no key).

`KlineStream` connects to the combined-stream endpoint (`<base>/stream`) of the market-data host
and subscribes with `SUBSCRIBE` messages to `<symbol>@kline_<interval>` streams. It yields typed
events: `ClosedCandle` for every kline update with `x == true` (updates of a candle still forming
are dropped), plus `Connected`, `Disconnected` and `SubscriptionError` so a consumer can react
(the feed backfills over REST after a reconnect).

Rules it keeps:

- At most `max_streams` streams (default 1024, Binance's limit) per connection; more streams open
  more connections. Streams are split evenly in sorted order.
- Subscriptions go out in messages of `chunk_size` streams (default 200), at least `subscribe_gap`
  seconds apart (default 0.25 s, i.e. 4 messages a second, under Binance's 5 per second).
- Any disconnect or error (refused connection, closed socket, no message for `silence_timeout`
  seconds when set) reconnects after `min(backoff_max, backoff_base * 2**n)` seconds scaled by a
  random factor between 0.5 and 1 (jitter); `n` resets once a message has arrived on the new
  connection. Each connection re-subscribes to all of its streams. Ping/pong is the library's.

Each connection runs in its own task; events of all of them come out of one queue, in arrival
order. A closed candle may be delivered more than once (Binance repeats updates, and a reconnect
overlaps): consumers deduplicate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from crypto_moonshot.models import EPOCH, Interval

logger = logging.getLogger(__name__)

STREAM_PATH = "/stream"
MAX_STREAMS_PER_CONNECTION = 1024
DEFAULT_CHUNK_SIZE = 200
DEFAULT_SUBSCRIBE_GAP_S = 0.25

Clock = Callable[[], datetime]
Sleep = Callable[[float], Awaitable[None]]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Candle(NamedTuple):
    """A closed candle: the columns of `KLINE_SCHEMA`, in order."""

    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trade_count: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float


@dataclass(frozen=True)
class ClosedCandle:
    symbol: str
    interval: Interval
    candle: Candle


@dataclass(frozen=True)
class Connected:
    """A connection subscribed to its streams. `reconnect` is False for its first time."""

    connection: int
    reconnect: bool
    streams: tuple[tuple[str, Interval], ...]


@dataclass(frozen=True)
class Disconnected:
    connection: int
    reason: str
    streams: tuple[tuple[str, Interval], ...]


@dataclass(frozen=True)
class SubscriptionError:
    connection: int
    message: str


StreamEvent = ClosedCandle | Connected | Disconnected | SubscriptionError


@dataclass(frozen=True)
class StreamConfig:
    base_url: str = "wss://data-stream.binance.vision"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    max_streams: int = MAX_STREAMS_PER_CONNECTION
    subscribe_gap: float = DEFAULT_SUBSCRIBE_GAP_S
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    silence_timeout: float | None = None
    """Treat a connection with no message for this many seconds as dead (None: never)."""
    ping_interval: float | None = 20.0

    def __post_init__(self) -> None:
        if self.chunk_size < 1 or self.max_streams < 1:
            msg = "chunk_size and max_streams must be at least 1"
            raise ValueError(msg)


def stream_name(symbol: str, interval: Interval) -> str:
    return f"{symbol.lower()}@kline_{interval.value}"


def _from_ms(ms: int) -> datetime:
    return EPOCH + timedelta(milliseconds=ms)


def parse_kline_event(message: Any) -> tuple[str, Interval, Candle] | None:
    """The closed candle in a combined-stream message, or None for anything else (a forming
    update, an acknowledgement, another event type)."""
    if not isinstance(message, dict):
        return None
    data = message.get("data")
    if not isinstance(data, dict) or data.get("e") != "kline":
        return None
    k = data.get("k")
    if not isinstance(k, dict) or k.get("x") is not True:
        return None
    try:
        candle = Candle(
            _from_ms(int(k["t"])),
            float(k["o"]),
            float(k["h"]),
            float(k["l"]),
            float(k["c"]),
            float(k["v"]),
            float(k["q"]),
            int(k["n"]),
            float(k["V"]),
            float(k["Q"]),
        )
        return str(k["s"]).upper(), Interval(k["i"]), candle
    except (KeyError, ValueError, TypeError):
        logger.warning("ignoring malformed kline event: %.200s", message)
        return None


class KlineStream:
    """Kline streams for `streams`, a set of `(symbol, interval)`. See the module docstring.

    Call `start()` (or iterate `events()`, which starts it) and `aclose()` when done.
    `last_message_at` is the time of the latest message of any kind.
    """

    def __init__(
        self,
        streams: Iterable[tuple[str, Interval]],
        config: StreamConfig | None = None,
        *,
        now: Clock = _utc_now,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or StreamConfig()
        self._now = now
        self._sleep = sleep
        self._rng = rng or random.Random()
        unique = sorted({(s.upper(), i) for s, i in streams}, key=lambda x: (x[0], x[1].value))
        size = self.config.max_streams
        self._groups = [tuple(unique[i : i + size]) for i in range(0, len(unique), size)]
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []
        self.last_message_at: datetime | None = None

    @property
    def connection_count(self) -> int:
        return len(self._groups)

    def start(self) -> None:
        if self._tasks:
            return
        self._tasks = [
            asyncio.create_task(self._run(index, group), name=f"kline-stream-{index}")
            for index, group in enumerate(self._groups)
        ]

    async def aclose(self) -> None:
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._queue.put_nowait(None)

    async def events(self) -> AsyncIterator[StreamEvent]:
        """Events until `aclose()`."""
        self.start()
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    # -- internals --------------------------------------------------------------------------

    def _backoff(self, attempt: int) -> float:
        cfg = self.config
        delay: float = min(cfg.backoff_max, cfg.backoff_base * 2**attempt)
        return float(delay * (0.5 + self._rng.random() / 2))

    async def _run(self, index: int, group: Sequence[tuple[str, Interval]]) -> None:
        cfg = self.config
        url = cfg.base_url.rstrip("/") + STREAM_PATH
        names = [stream_name(s, i) for s, i in group]
        streams = tuple(group)
        attempt = 0
        first = True
        while True:
            connected = False
            try:
                async with connect(url, ping_interval=cfg.ping_interval, max_size=None) as ws:
                    await self._subscribe(ws, names)
                    self._queue.put_nowait(Connected(index, not first, streams))
                    connected, first = True, False
                    while True:
                        async with asyncio.timeout(cfg.silence_timeout):
                            raw = await ws.recv()
                        attempt = 0
                        self._handle(index, raw)
            except TimeoutError:
                reason = f"no message for {cfg.silence_timeout}s"
            except (OSError, WebSocketException) as exc:
                reason = f"{type(exc).__name__}: {exc}"
            if connected:
                self._queue.put_nowait(Disconnected(index, reason, streams))
            await self._sleep(self._backoff(attempt))
            attempt += 1

    async def _subscribe(self, ws: ClientConnection, names: Sequence[str]) -> None:
        size = self.config.chunk_size
        for number, start in enumerate(range(0, len(names), size)):
            if number:
                await self._sleep(self.config.subscribe_gap)
            message = {"method": "SUBSCRIBE", "params": list(names[start : start + size])}
            await ws.send(json.dumps({**message, "id": number + 1}))

    def _handle(self, index: int, raw: str | bytes) -> None:
        self.last_message_at = self._now()
        try:
            message = json.loads(raw)
        except ValueError:
            logger.warning("ignoring non-JSON stream message")
            return
        if isinstance(message, dict) and message.get("error") is not None:
            self._queue.put_nowait(SubscriptionError(index, f"{message['error']}"))
            return
        if isinstance(message, dict) and "code" in message and "msg" in message:
            self._queue.put_nowait(SubscriptionError(index, f"{message['code']}: {message['msg']}"))
            return
        parsed = parse_kline_event(message)
        if parsed is None:
            return
        symbol, interval, candle = parsed
        self._queue.put_nowait(ClosedCandle(symbol, interval, candle))
