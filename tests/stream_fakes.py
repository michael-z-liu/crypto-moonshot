"""A local fake of Binance's combined kline stream: a real WebSocket server on 127.0.0.1."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime, timedelta
from types import TracebackType
from typing import Any, Self

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from crypto_moonshot.models import EPOCH, Interval


def kline_message(
    symbol: str,
    interval: Interval,
    open_time: datetime,
    *,
    closed: bool = True,
    price: float = 100.0,
) -> dict[str, Any]:
    ms = (open_time - EPOCH) // timedelta(milliseconds=1)
    close_ms = (interval.next(open_time) - EPOCH) // timedelta(milliseconds=1) - 1
    return {
        "stream": f"{symbol.lower()}@kline_{interval.value}",
        "data": {
            "e": "kline",
            "E": close_ms,
            "s": symbol,
            "k": {
                "t": ms,
                "T": close_ms,
                "s": symbol,
                "i": interval.value,
                "o": f"{price}",
                "h": f"{price + 2}",
                "l": f"{price - 1}",
                "c": f"{price + 1}",
                "v": "10.5",
                "n": 42,
                "x": closed,
                "q": "1050.25",
                "V": "5.25",
                "Q": "525.125",
                "B": "0",
            },
        },
    }


class FakeStreamServer:
    def __init__(self) -> None:
        self.connections: list[ServerConnection] = []
        self.received: list[list[dict[str, Any]]] = []
        self.reject_subscriptions = False
        self._server: Server | None = None

    @property
    def url(self) -> str:
        assert self._server is not None
        port = next(iter(self._server.sockets)).getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    async def __aenter__(self) -> Self:
        self._server = await serve(self._handler, "127.0.0.1", 0)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handler(self, ws: ServerConnection) -> None:
        index = len(self.connections)
        self.connections.append(ws)
        self.received.append([])
        assert ws.request is not None
        assert ws.request.path == "/stream"
        try:
            async for raw in ws:
                message = json.loads(raw)
                self.received[index].append(message)
                if message.get("method") == "SUBSCRIBE":
                    if self.reject_subscriptions:
                        reply: dict[str, Any] = {"error": {"code": 2, "msg": "Invalid request"}}
                    else:
                        reply = {"result": None}
                    await ws.send(json.dumps({**reply, "id": message["id"]}))
        except ConnectionClosed:
            return

    def subscribed(self, connection: int) -> list[str]:
        """Stream names a connection has subscribed to so far."""
        names: list[str] = []
        for message in self.received[connection]:
            if message.get("method") == "SUBSCRIBE":
                names.extend(message["params"])
        return names

    async def send(self, *messages: dict[str, Any], connection: int = -1) -> None:
        ws = self.connections[connection]
        for message in messages:
            await ws.send(json.dumps(message))

    async def drop(self, connection: int = -1) -> None:
        await self.connections[connection].close(code=1011)


async def until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    """Wait (polling) until `condition()` holds; fail the test after `timeout` seconds."""
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)
