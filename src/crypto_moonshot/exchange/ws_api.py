"""The Binance WebSocket API user data stream (built for the spot testnet).

Binance retired the REST `listenKey` streams; account events now come over a WebSocket API
connection (`wss://ws-api.testnet.binance.vision/ws-api/v3` on the testnet) after a signed
`userDataStream.subscribe.signature` request. See `docs/binance-spot-api.md`, "User data
stream (WebSocket API)". Rules:

- **Requests** are JSON frames `{"id", "method", "params"}`; the answer carries the same `id`
  and either `result` (status 200) or `error` (`{"code", "msg"}`, raised as `WsApiError`).
- **Subscribing** signs `apiKey`, `timestamp` and `recvWindow`: the parameters sorted by name,
  joined as `name=value&...`, HMAC-SHA256 with the secret, hex (the REST `sign`).
- **Events** arrive as `{"subscriptionId", "event": {...}}`. `executionReport`, `listStatus` and
  `outboundAccountPosition` are parsed (`ExecutionReport`, `ListStatus`, `AccountPosition`);
  anything else is passed on as `OtherEvent`. `eventStreamTerminated` ends the subscription: the
  client reports a gap (`StreamDown`) and subscribes again (`StreamUp(reconnect=True)`).
  `serverShutdown` (no `subscriptionId`) closes the connection: the client reconnects.
- **Connections** are re-opened with exponential backoff (with jitter) after any failure, and
  rotated before Binance's 24-hour limit (`max_age`). The server pings every 20 s; the
  `websockets` library answers. Every reconnect is a `StreamUp(reconnect=True)`: events sent
  while the stream was down are lost, so the consumer must catch up over REST.
- Keys and signatures never reach a log line, an exception or a repr.

Everything the stream hears is passed to one `sink` callable, on the event-loop thread, in the
order received.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from pydantic import SecretStr
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from crypto_moonshot.exchange.client import DEFAULT_RECV_WINDOW_MS, sign
from crypto_moonshot.exchange.models import dec, from_ms, to_ms

logger = logging.getLogger(__name__)

TESTNET_WS_API_URL = "wss://ws-api.testnet.binance.vision/ws-api/v3"
MAX_CONNECTION_AGE = timedelta(hours=23, minutes=30)
"""Binance closes a WebSocket API connection after 24 hours; rotate before."""
SUBSCRIBE_METHOD = "userDataStream.subscribe.signature"

Clock = Callable[[], datetime]
Sleep = Callable[[float], Awaitable[None]]
Connect = Callable[..., Any]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class WsApiError(Exception):
    """A WebSocket API request failed: Binance's `code` and `msg` (never a parameter)."""

    def __init__(self, message: str, *, status: int | None = None, code: int | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


# -- events -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """An `executionReport`: one change of one order (`docs/binance-spot-api.md`)."""

    symbol: str
    client_order_id: str
    """`c`; on a cancel this is the cancel request's ID, see `orig_client_order_id`."""
    orig_client_order_id: str
    """`C`: the cancelled order's client ID (empty unless cancelled)."""
    side: str
    order_type: str
    execution_type: str
    status: str
    reject_reason: str
    order_id: int
    order_list_id: int | None
    quantity: Decimal
    price: Decimal
    stop_price: Decimal
    last_quantity: Decimal
    last_price: Decimal
    cumulative_quantity: Decimal
    cumulative_quote: Decimal
    commission: Decimal
    commission_asset: str
    trade_id: int | None
    time: datetime
    """Transaction time (`T`)."""

    @property
    def order_client_id(self) -> str:
        """The client ID of the order this report is about (`C` on cancels, else `c`)."""
        if self.execution_type == "CANCELED" and self.orig_client_order_id:
            return self.orig_client_order_id
        return self.client_order_id

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> ExecutionReport:
        trade = raw.get("t")
        list_id = raw.get("g")
        return cls(
            symbol=str(raw["s"]),
            client_order_id=str(raw.get("c", "")),
            orig_client_order_id=str(raw.get("C") or ""),
            side=str(raw.get("S", "")),
            order_type=str(raw.get("o", "")),
            execution_type=str(raw.get("x", "")),
            status=str(raw.get("X", "")),
            reject_reason=str(raw.get("r", "NONE")),
            order_id=int(raw["i"]),
            order_list_id=None if list_id is None or int(list_id) < 0 else int(list_id),
            quantity=dec(raw.get("q")),
            price=dec(raw.get("p")),
            stop_price=dec(raw.get("P")),
            last_quantity=dec(raw.get("l")),
            last_price=dec(raw.get("L")),
            cumulative_quantity=dec(raw.get("z")),
            cumulative_quote=dec(raw.get("Z")),
            commission=dec(raw.get("n")),
            commission_asset=str(raw.get("N") or ""),
            trade_id=None if trade is None or int(trade) < 0 else int(trade),
            time=from_ms(int(raw.get("T", raw.get("E", 0)))),
        )


@dataclass(frozen=True, slots=True)
class ListStatus:
    """A `listStatus`: an OCO list changed (`list_order_status` `EXECUTING` / `ALL_DONE`)."""

    symbol: str
    order_list_id: int
    list_status_type: str
    list_order_status: str
    list_client_order_id: str
    time: datetime
    orders: tuple[tuple[str, int, str], ...]
    """(symbol, order ID, client order ID) of each leg."""

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> ListStatus:
        return cls(
            symbol=str(raw["s"]),
            order_list_id=int(raw["g"]),
            list_status_type=str(raw.get("l", "")),
            list_order_status=str(raw.get("L", "")),
            list_client_order_id=str(raw.get("C", "")),
            time=from_ms(int(raw.get("T", raw.get("E", 0)))),
            orders=tuple((str(o["s"]), int(o["i"]), str(o["c"])) for o in raw.get("O", ())),
        )


@dataclass(frozen=True, slots=True)
class AccountPosition:
    """An `outboundAccountPosition`: the balances that changed, `asset -> (free, locked)`."""

    time: datetime
    balances: Mapping[str, tuple[Decimal, Decimal]]

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> AccountPosition:
        return cls(
            time=from_ms(int(raw.get("u", raw.get("E", 0)))),
            balances={str(b["a"]): (dec(b.get("f")), dec(b.get("l"))) for b in raw.get("B", ())},
        )


@dataclass(frozen=True, slots=True)
class OtherEvent:
    """Any other user data event (`balanceUpdate`, `externalLockUpdate`, ...), unparsed."""

    kind: str
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StreamUp:
    """The stream is subscribed. `reconnect`: not the first time, so events may have been
    missed while it was down."""

    reconnect: bool


@dataclass(frozen=True, slots=True)
class StreamDown:
    """The stream stopped delivering events (`reason`); it reconnects by itself."""

    reason: str


UserDataEvent = ExecutionReport | ListStatus | AccountPosition | OtherEvent
StreamMessage = UserDataEvent | StreamUp | StreamDown
Sink = Callable[[StreamMessage], None]


def parse_event(raw: Mapping[str, Any]) -> UserDataEvent:
    """The typed form of one user data event (the `event` object of a frame)."""
    kind = str(raw.get("e", ""))
    if kind == "executionReport":
        return ExecutionReport.from_json(raw)
    if kind == "listStatus":
        return ListStatus.from_json(raw)
    if kind == "outboundAccountPosition":
        return AccountPosition.from_json(raw)
    return OtherEvent(kind, dict(raw))


def signed_params(
    api_key: SecretStr, api_secret: SecretStr, timestamp_ms: int, recv_window: int
) -> dict[str, Any]:
    """The parameters of a signed WebSocket API request: `apiKey`, `recvWindow`, `timestamp`
    and the `signature` over all of them sorted by name."""
    params: dict[str, Any] = {
        "apiKey": api_key.get_secret_value(),
        "recvWindow": recv_window,
        "timestamp": timestamp_ms,
    }
    payload = "&".join(f"{k}={params[k]}" for k in sorted(params))
    params["signature"] = sign(api_secret.get_secret_value(), payload)
    return params


# -- the client -------------------------------------------------------------------------------


class _Restart(Exception):
    """The connection must be re-opened (`serverShutdown`, rotation)."""


class UserDataStream:
    """The user data stream on one WebSocket API connection; see the module docstring.

    Args:
        api_key, api_secret: the account's keys (only sent in the signed subscribe request).
        sink: called with every `StreamMessage`, in order, on the event-loop thread.
        url: the WebSocket API endpoint (default the spot testnet).
        recv_window: `recvWindow` in ms of the subscribe request.
        now: aware-UTC clock for the request timestamp (pass the REST client's clock plus its
            server-time offset if the local clock is off).
        sleep: waits between reconnects (tests pass a fast one).
        backoff_base, backoff_max: reconnect delays in seconds (doubling, with jitter).
        max_age: rotate the connection after this long.
        request_timeout: seconds to wait for an answer to a request.
        connect: the `websockets` connect function (tests may wrap it).
    """

    def __init__(
        self,
        api_key: SecretStr,
        api_secret: SecretStr,
        sink: Sink,
        *,
        url: str = TESTNET_WS_API_URL,
        recv_window: int = DEFAULT_RECV_WINDOW_MS,
        now: Clock = _utc_now,
        sleep: Sleep = asyncio.sleep,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        max_age: timedelta = MAX_CONNECTION_AGE,
        request_timeout: float = 10.0,
        ping_interval: float | None = 20.0,
        connect: Connect = connect,
        rng: random.Random | None = None,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._sink = sink
        self.url = url
        self._recv_window = recv_window
        self._now = now
        self._sleep = sleep
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._max_age = max_age
        self._request_timeout = request_timeout
        self._ping_interval = ping_interval
        self._connect = connect
        self._rng = rng or random.Random()
        self._ids = itertools.count(1)
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._ws: ClientConnection | None = None
        self._task: asyncio.Task[None] | None = None
        self._resubscribing: asyncio.Task[None] | None = None
        self._subscribed = False
        self._ever_up = False
        self.subscription_id: int | None = None
        self.received = 0
        """Event frames received (tests wait on it)."""
        self.connects = 0
        """Connections opened."""
        self.last_error: str | None = None

    def __repr__(self) -> str:
        return f"UserDataStream(url={self.url!r}, connected={self.connected})"

    __str__ = __repr__

    @property
    def connected(self) -> bool:
        """Whether the stream is subscribed now (events are being delivered)."""
        return self._subscribed

    def start(self) -> None:
        """Start connecting in the background (idempotent)."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="user-data-stream")

    async def aclose(self) -> None:
        task, self._task = self._task, None
        for running in (task, self._resubscribing):
            if running is not None:
                running.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await running
        self._fail_pending("closed")
        self._mark_down("closed")

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        """Send a request on the open connection and return its `result`. Raises `WsApiError`
        for an error answer, `ConnectionError` without a connection, `TimeoutError`."""
        ws = self._ws
        if ws is None:
            msg = "the WebSocket API connection is not open"
            raise ConnectionError(msg)
        request_id = f"req-{next(self._ids)}"
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        frame: dict[str, Any] = {"id": request_id, "method": method}
        if params:
            frame["params"] = dict(params)
        try:
            await ws.send(json.dumps(frame))
            async with asyncio.timeout(self._request_timeout):
                return await future
        finally:
            self._pending.pop(request_id, None)

    # -- internals --------------------------------------------------------------------------

    def _backoff(self, attempt: int) -> float:
        delay = min(self._backoff_max, self._backoff_base * 2**attempt)
        return float(delay * (0.5 + self._rng.random() / 2))

    async def _run(self) -> None:
        attempt = 0
        while True:
            rotate = False
            try:
                async with self._connect(
                    self.url, ping_interval=self._ping_interval, max_size=None
                ) as ws:
                    self.connects += 1
                    self._ws = ws
                    reader = asyncio.create_task(self._read(ws))
                    try:
                        await self._subscribe()
                        attempt = 0
                        done, _ = await asyncio.wait(
                            {reader}, timeout=self._max_age.total_seconds()
                        )
                        if not done:
                            rotate = True
                            reason = "rotating the connection before its 24-hour limit"
                        else:
                            reader.result()
                            reason = "connection closed"
                    finally:
                        reader.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await reader
            except _Restart as exc:
                reason = str(exc)
            except (OSError, WebSocketException, WsApiError, TimeoutError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
            finally:
                self._ws = None
                self._fail_pending("connection closed")
            self.last_error = reason
            self._mark_down(reason)
            logger.info("user data stream down: %s", reason)
            if not rotate:
                await self._sleep(self._backoff(attempt))
                attempt += 1

    async def _subscribe(self) -> None:
        params = signed_params(
            self._api_key, self._api_secret, to_ms(self._now()), self._recv_window
        )
        result = await self.request(SUBSCRIBE_METHOD, params)
        subscription = result.get("subscriptionId") if isinstance(result, dict) else None
        self.subscription_id = None if subscription is None else int(subscription)
        self._subscribed = True
        reconnect = self._ever_up
        self._ever_up = True
        logger.info("user data stream subscribed (subscription %s)", self.subscription_id)
        self._sink(StreamUp(reconnect))

    async def _resubscribe(self) -> None:
        try:
            await self._subscribe()
        except (OSError, WebSocketException, WsApiError, TimeoutError) as exc:
            logger.warning("user data stream: resubscribing failed (%s)", type(exc).__name__)
            ws = self._ws
            if ws is not None:
                await ws.close()  # the run loop reconnects

    def _mark_down(self, reason: str) -> None:
        if self._subscribed:
            self._subscribed = False
            self._sink(StreamDown(reason))

    def _fail_pending(self, reason: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ConnectionError(reason))
        self._pending.clear()

    async def _read(self, ws: ClientConnection) -> None:
        async for raw in ws:
            try:
                message = json.loads(raw)
            except ValueError:
                logger.warning("user data stream: ignoring a non-JSON frame")
                continue
            if not isinstance(message, dict):
                continue
            request_id = message.get("id")
            if request_id is not None:
                self._answer(str(request_id), message)
                continue
            event = message.get("event")
            if not isinstance(event, dict):
                continue
            kind = event.get("e")
            if kind == "serverShutdown":
                raise _Restart("server shutdown")
            self.received += 1
            if kind == "eventStreamTerminated":
                self._mark_down("event stream terminated")
                self._resubscribing = asyncio.create_task(self._resubscribe())
                continue
            try:
                parsed = parse_event(event)
            except (KeyError, ValueError, TypeError, ArithmeticError):
                logger.warning("user data stream: ignoring a malformed %s event", kind)
                continue
            self._sink(parsed)

    def _answer(self, request_id: str, message: Mapping[str, Any]) -> None:
        future = self._pending.get(request_id)
        if future is None or future.done():
            return
        status = message.get("status")
        if status == 200 and "error" not in message:
            future.set_result(message.get("result"))
            return
        error = message.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        text = error.get("msg") if isinstance(error, dict) else None
        future.set_exception(
            WsApiError(
                f"WebSocket API error {status}, code {code}: {text}",
                status=status if isinstance(status, int) else None,
                code=code if isinstance(code, int) else None,
            )
        )
