"""A fake Binance spot exchange for broker tests: REST (an `httpx.MockTransport` handler) and the
WebSocket API user data stream (a `websockets` server on 127.0.0.1), sharing one matching engine.

The test drives a price per symbol (`set_price`, or `walk` along a candle: open, then the low
and high in the order a candle of that direction most plausibly took, then the close). Orders:

- `MARKET` fills at the current price plus `slippage` (against the order: up for buys, down for
  sells), in `split` fills one tick apart (each worse than the last). `hold_market` keeps the
  market orders of a symbol open until `release_market` fills them, in parts.
- Commission: `commission` of the base asset on buys, of USDT on sells (`commission_asset`
  overrides both, e.g. `BNB`, at `bnb_price`).
- `LIMIT_MAKER` rests (rejected `-2010` if it would take) and fills at its price when the price
  touches it, at most `liquidity[symbol]` per price step (partial fills).
- `STOP_LOSS` triggers when the price reaches its stop (or its trailing level: the peak since
  placement minus `trailingDelta` basis points) and then fills as a market order (with
  slippage beyond the level); `-2021` if a stop is already crossed at placement.
- OCO (`LIMIT_MAKER` above, `STOP_LOSS` below): the first execution of one leg expires the
  other; the list is `ALL_DONE` when no leg can fill any more. Cancelling a leg cancels the list.
- Balances free/locked, `exchangeInfo` filters (`PRICE_FILTER`, `LOT_SIZE`, `NOTIONAL`,
  `PERCENT_PRICE_BY_SIDE` against the current price, `TRAILING_DELTA`), signatures (REST and
  WebSocket API) and timestamps are checked.
- User data events (`executionReport`, `listStatus`, `outboundAccountPosition`) go to every
  subscribed WebSocket connection; with none subscribed they are lost (as on Binance).

Faults: REST errors before or after the request took effect (`fail`, `fail_after`, `timeout`
with `placed`), a ban (`ban`), dropped connections (`drop_connections`, with `refuse` to keep
the client out for a while), held or omitted events (`hold_events`, `release_events`,
`omit_events`), `terminate_stream` (`eventStreamTerminated`), `server_shutdown`, and `reset`
(the monthly testnet reset: orders gone, balances back to the start).

Nothing here talks to a real host; the keys are obviously fake.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from typing import Any
from urllib.parse import parse_qsl

import httpx
from exchange_fakes import API_KEY, API_SECRET, DEFAULT_SYMBOLS, symbol_json
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

ZERO = Decimal(0)
EIGHT = Decimal("0.00000001")
OPEN = ("NEW", "PARTIALLY_FILLED")

WEIGHTS = {
    ("GET", "/api/v3/time"): 1,
    ("GET", "/api/v3/exchangeInfo"): 20,
    ("GET", "/api/v3/ticker/price"): 2,
    ("GET", "/api/v3/account"): 20,
    ("POST", "/api/v3/order"): 1,
    ("POST", "/api/v3/orderList/oco"): 1,
    ("DELETE", "/api/v3/order"): 1,
    ("DELETE", "/api/v3/orderList"): 1,
    ("GET", "/api/v3/order"): 4,
    ("GET", "/api/v3/openOrders"): 6,
    ("GET", "/api/v3/openOrderList"): 6,
    ("GET", "/api/v3/orderList"): 4,
    ("GET", "/api/v3/myTrades"): 5,
}
_PUBLIC = {
    ("GET", "/api/v3/time"),
    ("GET", "/api/v3/exchangeInfo"),
    ("GET", "/api/v3/ticker/price"),
}

SYMBOLS = [
    *DEFAULT_SYMBOLS,
    symbol_json("AAAUSDT", "AAA", tick="0.01000000", step="0.00100000", min_qty="0.00100000"),
    symbol_json("BBBUSDT", "BBB", tick="0.01000000", step="0.00100000", min_qty="0.00100000"),
    symbol_json("CCCUSDT", "CCC", tick="0.01000000", step="0.00100000", min_qty="0.00100000"),
    symbol_json("DDDUSDT", "DDD", tick="0.01000000", step="0.00100000", min_qty="0.00100000"),
]


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _s(value: Decimal) -> str:
    return f"{value:f}"


class ApiError(Exception):
    def __init__(self, status: int, code: int, msg: str) -> None:
        super().__init__(msg)
        self.status, self.code, self.msg = status, code, msg


@dataclass
class Fault:
    """A scripted REST failure for the next `times` requests whose path contains `path`.
    `after`: the request takes effect first (an order placed whose answer is lost)."""

    path: str
    method: str | None = None
    times: int = 1
    status: int = 500
    body: dict[str, Any] | None = None
    timeout: bool = False
    after: bool = False


@dataclass
class FakeOrder:
    symbol: str
    order_id: int
    client_id: str
    side: str
    type: str
    quantity: Decimal
    time: datetime
    price: Decimal = ZERO
    stop_price: Decimal | None = None
    trailing_delta: int | None = None
    quote_qty: Decimal | None = None
    list_id: int | None = None
    status: str = "NEW"
    executed: Decimal = ZERO
    cumulative_quote: Decimal = ZERO
    fills: list[dict[str, str]] = field(default_factory=list)
    peak: Decimal = ZERO
    triggered: bool = False
    update_time: datetime | None = None
    locked: Decimal = ZERO
    """What the order still holds locked (base for sells, USDT for held market buys)."""

    @property
    def open(self) -> bool:
        return self.status in OPEN

    @property
    def remaining(self) -> Decimal:
        return self.quantity - self.executed

    def json(self, *, place: bool, full: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "symbol": self.symbol,
            "orderId": self.order_id,
            "orderListId": -1 if self.list_id is None else self.list_id,
            "clientOrderId": self.client_id,
            "price": _s(self.price),
            "origQty": _s(self.quantity),
            "executedQty": _s(self.executed),
            "cummulativeQuoteQty": _s(self.cumulative_quote),
            "status": self.status,
            "timeInForce": "GTC",
            "type": self.type,
            "side": self.side,
        }
        if self.quote_qty is not None:
            out["origQuoteOrderQty"] = _s(self.quote_qty)
        if self.stop_price is not None:
            out["stopPrice"] = _s(self.stop_price)
        if self.trailing_delta is not None:
            out["trailingDelta"] = self.trailing_delta
        if place:
            out["transactTime"] = to_ms(self.update_time or self.time)
        else:
            out["time"] = to_ms(self.time)
            out["updateTime"] = to_ms(self.update_time or self.time)
        if full:
            out["fills"] = list(self.fills)
        return out


@dataclass
class FakeList:
    list_id: int
    client_id: str
    symbol: str
    legs: list[FakeOrder]
    time: datetime
    status: str = "EXECUTING"
    status_type: str = "EXEC_STARTED"

    def json(self, *, reports: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "orderListId": self.list_id,
            "contingencyType": "OCO",
            "listStatusType": self.status_type,
            "listOrderStatus": self.status,
            "listClientOrderId": self.client_id,
            "transactionTime": to_ms(self.time),
            "symbol": self.symbol,
            "orders": [
                {"symbol": o.symbol, "orderId": o.order_id, "clientOrderId": o.client_id}
                for o in self.legs
            ],
        }
        if reports:
            out["orderReports"] = [o.json(place=True) for o in self.legs]
        return out


@dataclass(eq=False)
class _Conn:
    ws: ServerConnection
    queue: asyncio.Queue[tuple[str, bool] | None] = field(default_factory=asyncio.Queue)
    """(frame, whether it is a user data event) to send; None closes the connection."""
    subscriptions: set[int] = field(default_factory=set)


class FakeExchange:
    """See the module docstring. `clock` is the exchange's aware-UTC clock (share it with the
    client so timestamps agree)."""

    def __init__(
        self,
        clock: Callable[[], datetime],
        *,
        api_key: str = API_KEY,
        api_secret: str = API_SECRET,
        symbols: list[dict[str, Any]] | None = None,
        usdt: Decimal = Decimal(10_000),
        slippage: Decimal = ZERO,
        split: int = 1,
        commission: Decimal = Decimal("0.001"),
    ) -> None:
        self.clock = clock
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = [dict(s) for s in (symbols if symbols is not None else SYMBOLS)]
        self.initial_usdt = usdt
        self.slippage = slippage
        self.split = split
        self.commission = commission
        self.commission_asset: str | None = None
        self.bnb_price = Decimal(300)
        self.prices: dict[str, Decimal] = {
            "BTCUSDT": Decimal("30000.00"),
            "ETHUSDT": Decimal("2000.00"),
            "SHIBUSDT": Decimal("0.00001000"),
            "XRPUSDT": Decimal("0.500000"),
            "AAAUSDT": Decimal("100.00"),
            "BBBUSDT": Decimal("100.00"),
            "CCCUSDT": Decimal("100.00"),
            "DDDUSDT": Decimal("100.00"),
        }
        self.balances: dict[str, list[Decimal]] = {"USDT": [usdt, ZERO]}
        self.orders: dict[str, FakeOrder] = {}
        self.lists: dict[str, FakeList] = {}
        self.trades: list[dict[str, Any]] = []
        self.requests: list[httpx.Request] = []
        self.faults: list[Fault] = []
        self.liquidity: dict[str, Decimal] = {}
        self.hold_market: set[str] = set()
        self.events: list[dict[str, Any]] = []
        """Every user data event generated (delivered or not)."""
        self.delivered = 0
        """User data event frames queued for a subscriber."""
        self.sent = 0
        """User data event frames actually sent."""
        self.held: list[dict[str, Any]] = []
        self._holding = False
        self._omit = 0
        self.refuse = False
        """Close every new WebSocket connection at once (the client stays out)."""
        self.ws_requests: list[dict[str, Any]] = []
        self._conns: set[_Conn] = set()
        self._server: Server | None = None
        self._next_order_id = 1000
        self._next_list_id = 1
        self._next_trade_id = 1
        self._next_subscription = 0

    # -- test controls ------------------------------------------------------------------------

    @property
    def now(self) -> datetime:
        return self.clock()

    def fail(
        self,
        path: str,
        *,
        method: str | None = None,
        status: int = 400,
        code: int | None = None,
        msg: str = "",
        times: int = 1,
    ) -> None:
        """The next `times` requests to `path` fail without taking effect."""
        body = None if code is None else {"code": code, "msg": msg}
        self.faults.append(Fault(path, method, times, status, body))

    def fail_after(
        self,
        path: str,
        *,
        method: str | None = None,
        status: int = 503,
        code: int = -1007,
        msg: str = "Timeout waiting for response from backend server. Send status unknown; "
        "execution status unknown.",
        times: int = 1,
    ) -> None:
        """The next `times` requests to `path` take effect, then answer with an error."""
        body = {"code": code, "msg": msg}
        self.faults.append(Fault(path, method, times, status, body, after=True))

    def timeout(
        self, path: str, *, method: str | None = None, placed: bool = False, times: int = 1
    ) -> None:
        """The next `times` requests to `path` time out; `placed`: after taking effect."""
        self.faults.append(Fault(path, method, times, timeout=True, after=placed))

    def ban(self) -> None:
        self.fail("", status=418, code=-1003, msg="Way too many requests; banned.", times=10**9)

    def set_filter(self, symbol: str, kind: str, **values: Any) -> None:
        """Change one filter of `symbol`'s rules (clients see it on their next refresh)."""
        for s in self.symbols:
            if s["symbol"] == symbol:
                s["filters"] = [
                    {**f, **values} if f["filterType"] == kind else f for f in s["filters"]
                ]

    def deposit(self, asset: str, amount: Decimal) -> None:
        self._bal(asset)[0] += amount

    def set_price(self, symbol: str, price: Decimal) -> None:
        """Move the market to `price` and execute whatever resting orders it reaches."""
        self.prices[symbol] = price
        for order in [o for o in self.orders.values() if o.symbol == symbol and o.open]:
            if order.open:  # an earlier fill may have expired it
                self._check_resting(order, price)

    def walk(self, symbol: str, o: float, h: float, lo: float, c: float) -> None:
        """Move along a candle: open, then low and high (low first for an up candle), close."""
        path = [o, lo, h, c] if c >= o else [o, h, lo, c]
        for price in path:
            self.set_price(symbol, Decimal(str(price)))

    def release_market(self, symbol: str, parts: int = 1) -> None:
        """Held market orders of `symbol` fill now, in `parts` fills (one per call)."""
        self.hold_market.discard(symbol)
        for order in [o for o in self.orders.values() if o.symbol == symbol and o.open]:
            if order.type == "MARKET":
                self._market_fill(order, parts=parts)

    def fill_part(self, client_id: str, fraction: Decimal) -> None:
        """Fill `fraction` of a held market order now (it stays partially filled)."""
        order = self.orders[client_id]
        price = self._market_price(order)
        if order.quote_qty is not None:
            spend = (order.quote_qty * fraction) - order.cumulative_quote
            qty = self._floor(order.symbol, spend / price)
        else:
            qty = self._floor(order.symbol, order.quantity * fraction) - order.executed
        if qty > 0:
            self._execute(order, qty, price)
            self._account_event(order.symbol)

    def hold_events(self) -> None:
        """Keep user data events back until `release_events` (a slow stream)."""
        self._holding = True

    def release_events(self) -> None:
        self._holding = False
        held, self.held = self.held, []
        for event in held:
            self._deliver(event)

    def omit_events(self, n: int) -> None:
        """Lose the next `n` user data events (never delivered)."""
        self._omit += n

    def reset(self) -> None:
        """The monthly testnet reset: every order and list is gone, balances start over."""
        self.orders.clear()
        self.lists.clear()
        self.trades.clear()
        self.balances = {"USDT": [self.initial_usdt, ZERO]}

    def requests_to(self, path: str, method: str | None = None) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == path and method in (None, r.method)]

    def balance(self, asset: str) -> tuple[Decimal, Decimal]:
        free, locked = self._bal(asset)
        return free, locked

    # -- the WebSocket API server ------------------------------------------------------------

    async def start_ws(self) -> str:
        """Start the WebSocket API server on 127.0.0.1 and return its URL."""
        self._server = await serve(self._handle_ws, "127.0.0.1", 0)
        port = next(iter(self._server.sockets)).getsockname()[1]
        return f"ws://127.0.0.1:{port}/ws-api/v3"

    async def stop_ws(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @property
    def subscribed(self) -> int:
        """WebSocket connections with a live user data subscription."""
        return sum(1 for c in self._conns if c.subscriptions)

    async def drop_connections(self) -> None:
        """Close every WebSocket connection (events emitted until the client is back are
        lost)."""
        for conn in list(self._conns):
            conn.subscriptions.clear()
            await conn.ws.close()
        self._conns.clear()

    def terminate_stream(self) -> None:
        """Send `eventStreamTerminated` and end every subscription (connections stay open)."""
        for conn in self._conns:
            for sub in list(conn.subscriptions):
                event = {"e": "eventStreamTerminated", "E": to_ms(self.now)}
                conn.queue.put_nowait((json.dumps({"subscriptionId": sub, "event": event}), False))
            conn.subscriptions.clear()

    def server_shutdown(self) -> None:
        """Send `serverShutdown` (no subscription ID) on every connection, then close them."""
        for conn in self._conns:
            event = {"e": "serverShutdown", "E": to_ms(self.now)}
            conn.queue.put_nowait((json.dumps({"event": event}), False))
            conn.subscriptions.clear()
            conn.queue.put_nowait(None)

    async def flushed(self, timeout: float = 2.0) -> None:
        """Wait until every queued frame was sent (the client may still be reading)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while any(not c.queue.empty() for c in self._conns):
            if loop.time() > deadline:
                raise TimeoutError("frames not flushed")
            await asyncio.sleep(0.001)

    async def _handle_ws(self, ws: ServerConnection) -> None:
        if self.refuse:
            await ws.close()
            return
        conn = _Conn(ws)
        self._conns.add(conn)
        writer = asyncio.create_task(self._write(conn))
        try:
            async for raw in ws:
                message = json.loads(raw)
                self.ws_requests.append(message)
                conn.queue.put_nowait((json.dumps(self._ws_answer(conn, message)), False))
        except ConnectionClosed:
            pass
        finally:
            self._conns.discard(conn)
            writer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await writer

    async def _write(self, conn: _Conn) -> None:
        while True:
            item = await conn.queue.get()
            if item is None:
                await conn.ws.close()
                return
            frame, is_event = item
            try:
                await conn.ws.send(frame)
            except ConnectionClosed:
                return
            if is_event:
                self.sent += 1

    def _ws_answer(self, conn: _Conn, message: dict[str, Any]) -> dict[str, Any]:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}
        if method == "userDataStream.subscribe.signature":
            error = self._check_ws_signature(params)
            if error is not None:
                return {"id": request_id, "status": 400, "error": error}
            self._next_subscription += 1
            conn.subscriptions.add(self._next_subscription)
            return {
                "id": request_id,
                "status": 200,
                "result": {"subscriptionId": self._next_subscription},
            }
        if method == "userDataStream.unsubscribe":
            conn.subscriptions.clear()
            return {"id": request_id, "status": 200, "result": {}}
        if method == "ping":
            return {"id": request_id, "status": 200, "result": {}}
        return {
            "id": request_id,
            "status": 400,
            "error": {"code": -1102, "msg": f"Unknown method {method}"},
        }

    def _check_ws_signature(self, params: dict[str, Any]) -> dict[str, Any] | None:
        signature = params.get("signature")
        rest = {k: v for k, v in params.items() if k != "signature"}
        if params.get("apiKey") != self.api_key:
            return {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."}
        payload = "&".join(f"{k}={rest[k]}" for k in sorted(rest))
        expected = hmac.new(self.api_secret.encode(), payload.encode(), hashlib.sha256)
        if not isinstance(signature, str) or not hmac.compare_digest(
            expected.hexdigest(), signature
        ):
            return {"code": -1022, "msg": "Signature for this request is not valid."}
        timestamp = int(rest.get("timestamp", 0))
        window = int(rest.get("recvWindow", 5000))
        server = to_ms(self.now)
        if not (timestamp < server + 1000 and server - timestamp <= window):
            return {
                "code": -1021,
                "msg": "Timestamp for this request is outside of the recvWindow.",
            }
        return None

    # -- user data events --------------------------------------------------------------------

    def _emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        if self._omit > 0:
            self._omit -= 1
            return
        if self._holding:
            self.held.append(event)
            return
        self._deliver(event)

    def _deliver(self, event: dict[str, Any]) -> None:
        for conn in self._conns:
            for sub in conn.subscriptions:
                self.delivered += 1
                conn.queue.put_nowait((json.dumps({"subscriptionId": sub, "event": event}), True))

    def _report(
        self,
        order: FakeOrder,
        execution: str,
        *,
        last_qty: Decimal = ZERO,
        last_price: Decimal = ZERO,
        commission: Decimal = ZERO,
        commission_asset: str | None = None,
        trade_id: int = -1,
        cancel_id: str | None = None,
    ) -> None:
        now = to_ms(self.now)
        self._emit(
            {
                "e": "executionReport",
                "E": now,
                "s": order.symbol,
                "c": cancel_id if cancel_id is not None else order.client_id,
                "S": order.side,
                "o": order.type,
                "f": "GTC",
                "q": _s(order.quantity),
                "p": _s(order.price),
                "P": _s(order.stop_price or ZERO),
                "C": order.client_id if cancel_id is not None else "",
                "x": execution,
                "X": order.status,
                "r": "NONE",
                "i": order.order_id,
                "l": _s(last_qty),
                "z": _s(order.executed),
                "L": _s(last_price),
                "n": _s(commission),
                "N": commission_asset,
                "T": now,
                "t": trade_id,
                "g": -1 if order.list_id is None else order.list_id,
                "Z": _s(order.cumulative_quote),
                "Y": _s(last_qty * last_price),
                "Q": _s(order.quote_qty or ZERO),
            }
        )

    def _list_event(self, listed: FakeList) -> None:
        now = to_ms(self.now)
        self._emit(
            {
                "e": "listStatus",
                "E": now,
                "s": listed.symbol,
                "g": listed.list_id,
                "c": "OCO",
                "l": listed.status_type,
                "L": listed.status,
                "r": "NONE",
                "C": listed.client_id,
                "T": now,
                "O": [{"s": o.symbol, "i": o.order_id, "c": o.client_id} for o in listed.legs],
            }
        )

    def _account_event(self, symbol: str) -> None:
        assets = ["USDT", self._base(symbol)]
        if self.commission_asset is not None:
            assets.append(self.commission_asset)
        self._emit(
            {
                "e": "outboundAccountPosition",
                "E": to_ms(self.now),
                "u": to_ms(self.now),
                "B": [{"a": a, "f": _s(self._bal(a)[0]), "l": _s(self._bal(a)[1])} for a in assets],
            }
        )

    # -- the REST handler --------------------------------------------------------------------

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        after: Fault | None = None
        for fault in self.faults:
            if (
                fault.times > 0
                and fault.path in request.url.path
                and fault.method in (None, request.method)
            ):
                fault.times -= 1
                if not fault.after:
                    if fault.timeout:
                        raise httpx.ReadTimeout("fake timeout")
                    return httpx.Response(fault.status, json=fault.body)
                after = fault
                break
        key = (request.method, request.url.path)
        if key not in WEIGHTS:
            response = httpx.Response(404, json={"code": -1102, "msg": "No such endpoint."})
        else:
            try:
                if key in _PUBLIC:
                    params = dict(parse_qsl(request.url.query.decode()))
                else:
                    params = self._authenticate(request)
                body = self._ROUTES[key](self, params)
                response = httpx.Response(200, json=body)
            except ApiError as err:
                response = httpx.Response(err.status, json={"code": err.code, "msg": err.msg})
        if after is not None:
            if after.timeout:
                raise httpx.ReadTimeout("fake timeout")
            return httpx.Response(after.status, json=after.body)
        return response

    def _authenticate(self, request: httpx.Request) -> dict[str, str]:
        supplied = request.headers.get("X-MBX-APIKEY")
        if supplied is None:
            raise ApiError(401, -2014, "API-key format invalid.")
        raw = request.url.query.decode()
        head, sep, signature = raw.rpartition("&signature=")
        if not sep:
            raise ApiError(400, -1102, "Mandatory parameter 'signature' was not sent.")
        params = dict(parse_qsl(head))
        expected = hmac.new(self.api_secret.encode(), head.encode(), hashlib.sha256).hexdigest()
        if supplied != self.api_key:
            raise ApiError(401, -2015, "Invalid API-key, IP, or permissions for action.")
        if not hmac.compare_digest(expected, signature):
            raise ApiError(400, -1022, "Signature for this request is not valid.")
        timestamp = int(params["timestamp"])
        window = int(params.get("recvWindow", "5000"))
        server = to_ms(self.now)
        if not (timestamp < server + 1000 and server - timestamp <= window):
            raise ApiError(400, -1021, "Timestamp for this request is outside of the recvWindow.")
        return params

    # -- rules -------------------------------------------------------------------------------

    def _symbol(self, name: str) -> dict[str, Any]:
        for s in self.symbols:
            if s["symbol"] == name:
                return s
        raise ApiError(400, -1121, "Invalid symbol.")

    def _filter(self, symbol: str, kind: str) -> dict[str, Any]:
        for f in self._symbol(symbol)["filters"]:
            if f["filterType"] == kind:
                return dict(f)
        return {}

    def _tick(self, symbol: str) -> Decimal:
        return Decimal(self._filter(symbol, "PRICE_FILTER")["tickSize"])

    def _step(self, symbol: str) -> Decimal:
        return Decimal(self._filter(symbol, "LOT_SIZE")["stepSize"])

    def _floor(self, symbol: str, qty: Decimal) -> Decimal:
        step = self._step(symbol)
        return (qty // step) * step if step else qty

    def _base(self, symbol: str) -> str:
        return str(self._symbol(symbol)["baseAsset"])

    def _check(
        self,
        symbol: str,
        side: str,
        quantity: Decimal | None,
        price: Decimal | None,
        notional: Decimal,
        *,
        market: bool = False,
    ) -> None:
        if self._symbol(symbol)["status"] != "TRADING":
            raise ApiError(400, -1013, "Market is closed.")
        if price is not None:
            pf = self._filter(symbol, "PRICE_FILTER")
            tick = Decimal(pf["tickSize"])
            if price < Decimal(pf["minPrice"]) or (tick and price % tick != 0):
                raise ApiError(400, -1013, "Filter failure: PRICE_FILTER")
            pp = self._filter(symbol, "PERCENT_PRICE_BY_SIDE")
            if pp:
                last = self.prices[symbol]
                up, down = (
                    ("askMultiplierUp", "askMultiplierDown")
                    if side == "SELL"
                    else ("bidMultiplierUp", "bidMultiplierDown")
                )
                if price > last * Decimal(str(pp[up])) or price < last * Decimal(str(pp[down])):
                    raise ApiError(400, -1013, "Filter failure: PERCENT_PRICE_BY_SIDE")
        if quantity is not None:
            lot = self._filter(symbol, "LOT_SIZE")
            step = Decimal(lot["stepSize"])
            if (
                quantity < Decimal(lot["minQty"])
                or quantity > Decimal(lot["maxQty"])
                or (step and quantity % step != 0)
            ):
                raise ApiError(400, -1013, "Filter failure: LOT_SIZE")
        notional_filter = self._filter(symbol, "NOTIONAL")
        if (not market or notional_filter.get("applyMinToMarket", True)) and notional < Decimal(
            notional_filter["minNotional"]
        ):
            raise ApiError(400, -1013, "Filter failure: NOTIONAL")

    def _check_trailing(self, symbol: str, delta: int) -> None:
        tf = self._filter(symbol, "TRAILING_DELTA")
        if tf and not (
            int(tf["minTrailingBelowDelta"]) <= delta <= int(tf["maxTrailingBelowDelta"])
        ):
            raise ApiError(400, -1013, "Filter failure: TRAILING_DELTA")

    # -- balances and fills -----------------------------------------------------------------

    def _bal(self, asset: str) -> list[Decimal]:
        return self.balances.setdefault(asset, [ZERO, ZERO])

    def _lock(self, asset: str, amount: Decimal) -> None:
        bal = self._bal(asset)
        if bal[0] < amount:
            raise ApiError(400, -2010, "Account has insufficient balance for requested action.")
        bal[0] -= amount
        bal[1] += amount

    def _unlock(self, asset: str, amount: Decimal) -> None:
        bal = self._bal(asset)
        bal[1] -= amount
        bal[0] += amount

    def _market_price(self, order: FakeOrder) -> Decimal:
        price = self.prices[order.symbol]
        tick = self._tick(order.symbol)
        if order.side == "BUY":
            raw = price * (1 + self.slippage)
            return (-(-raw // tick)) * tick if tick else raw
        raw = price * (1 - self.slippage)
        return (raw // tick) * tick if tick else raw

    def _execute(self, order: FakeOrder, qty: Decimal, price: Decimal) -> None:
        """Execute `qty` of `order` at `price`: balances, fills, trade, `executionReport`."""
        base = self._base(order.symbol)
        value = qty * price
        if self.commission_asset is not None:
            fee = (value * self.commission / self.bnb_price).quantize(EIGHT, ROUND_DOWN)
            fee_asset = self.commission_asset
        elif order.side == "BUY":
            fee, fee_asset = (qty * self.commission).quantize(EIGHT, ROUND_DOWN), base
        else:
            fee, fee_asset = (value * self.commission).quantize(EIGHT, ROUND_DOWN), "USDT"
        if order.side == "BUY":
            usdt = self._bal("USDT")
            if order.locked > 0:
                take = min(order.locked, value)
                usdt[1] -= take
                order.locked -= take
                usdt[0] -= value - take
            else:
                usdt[0] -= value
            self._bal(base)[0] += qty
        else:
            bal = self._bal(base)
            if order.locked > 0:
                bal[1] -= qty
                order.locked -= qty
            else:
                bal[0] -= qty
            self._bal("USDT")[0] += value
        self._bal(fee_asset)[0] -= fee
        trade_id = self._next_trade_id
        self._next_trade_id += 1
        order.executed += qty
        order.cumulative_quote += value
        order.update_time = self.now
        done = order.quantity > 0 and order.executed >= order.quantity
        order.status = "FILLED" if done else "PARTIALLY_FILLED"
        order.fills.append(
            {
                "price": _s(price),
                "qty": _s(qty),
                "commission": _s(fee),
                "commissionAsset": fee_asset,
                "tradeId": str(trade_id),
            }
        )
        self.trades.append(
            {
                "symbol": order.symbol,
                "id": trade_id,
                "orderId": order.order_id,
                "orderListId": -1 if order.list_id is None else order.list_id,
                "price": _s(price),
                "qty": _s(qty),
                "quoteQty": _s(value),
                "commission": _s(fee),
                "commissionAsset": fee_asset,
                "time": to_ms(self.now),
                "isBuyer": order.side == "BUY",
                "isMaker": order.type == "LIMIT_MAKER",
                "isBestMatch": True,
            }
        )
        self._report(
            order,
            "TRADE",
            last_qty=qty,
            last_price=price,
            commission=fee,
            commission_asset=fee_asset,
            trade_id=trade_id,
        )

    def _market_fill(self, order: FakeOrder, *, parts: int | None = None) -> None:
        """Fill a market order completely, in `split` (or `parts`) fills one tick apart."""
        n = max(1, parts if parts is not None else self.split)
        tick = self._tick(order.symbol)
        sign = 1 if order.side == "BUY" else -1
        base_price = self._market_price(order)
        chunks: list[tuple[Decimal, Decimal]] = []
        spent, sold = order.cumulative_quote, order.executed
        for i in range(n):
            price = base_price + sign * i * tick
            if order.quote_qty is not None:
                budget = (order.quote_qty - spent) / (n - i)
                qty = self._floor(order.symbol, budget / price)
            else:
                left = order.quantity - sold
                qty = left if i == n - 1 else self._floor(order.symbol, left / (n - i))
            if qty > 0:
                chunks.append((qty, price))
                spent += qty * price
                sold += qty
        if order.quote_qty is not None:
            order.quantity = sold
        for qty, price in chunks:
            self._execute(order, qty, price)
        if order.locked > 0:  # what a held order reserved and did not use
            asset = "USDT" if order.side == "BUY" else self._base(order.symbol)
            self._unlock(asset, order.locked)
            order.locked = ZERO
        if order.executed == 0:
            order.status = "EXPIRED"
            self._report(order, "EXPIRED")
        self._finish(order)
        self._account_event(order.symbol)

    def _check_resting(self, order: FakeOrder, price: Decimal) -> None:
        if order.type == "MARKET":
            return
        if order.type == "LIMIT_MAKER":
            touched = price >= order.price if order.side == "SELL" else price <= order.price
            if not touched:
                return
            qty = order.remaining
            limit = self.liquidity.get(order.symbol)
            if limit is not None:
                qty = min(qty, limit)
            self._execute(order, qty, order.price)
            self._expire_other_leg(order)
            self._finish(order)
            self._account_event(order.symbol)
            return
        if order.type == "STOP_LOSS":
            if order.trailing_delta is not None:
                order.peak = max(order.peak, price)
                level = order.peak * (1 - Decimal(order.trailing_delta) / 10_000)
                if order.stop_price is not None:
                    level = max(level, order.stop_price)
            else:
                assert order.stop_price is not None
                level = order.stop_price
            if price <= level:
                order.triggered = True
                self._expire_other_leg(order)
                self._market_fill(order)

    def _expire_other_leg(self, order: FakeOrder) -> None:
        if order.list_id is None:
            return
        listed = self._list_of(order)
        for leg in listed.legs:
            if leg is not order and leg.open:
                leg.status = "EXPIRED"
                leg.update_time = self.now
                leg.locked = ZERO  # the pair shares one lock, kept by the executing leg
                self._report(leg, "EXPIRED")

    def _finish(self, order: FakeOrder) -> None:
        """Close the order's list once no leg can fill any more."""
        if order.list_id is None:
            return
        listed = self._list_of(order)
        if listed.status == "EXECUTING" and not any(leg.open for leg in listed.legs):
            listed.status, listed.status_type = "ALL_DONE", "ALL_DONE"
            self._list_event(listed)

    def _list_of(self, order: FakeOrder) -> FakeList:
        return next(ls for ls in self.lists.values() if ls.list_id == order.list_id)

    # -- endpoints ---------------------------------------------------------------------------

    def _time(self, params: dict[str, str]) -> Any:
        return {"serverTime": to_ms(self.now)}

    def _exchange_info(self, params: dict[str, str]) -> Any:
        return {
            "timezone": "UTC",
            "serverTime": to_ms(self.now),
            "rateLimits": [],
            "symbols": self.symbols,
        }

    def _ticker(self, params: dict[str, str]) -> Any:
        symbol = params["symbol"]
        self._symbol(symbol)
        return {"symbol": symbol, "price": _s(self.prices[symbol])}

    def _account(self, params: dict[str, str]) -> Any:
        return {
            "canTrade": True,
            "updateTime": to_ms(self.now),
            "balances": [
                {"asset": a, "free": _s(b[0]), "locked": _s(b[1])}
                for a, b in sorted(self.balances.items())
            ],
        }

    def _check_new_id(self, client_id: str) -> None:
        existing = self.orders.get(client_id)
        if existing is not None and existing.open:
            raise ApiError(400, -2010, "Duplicate order sent.")
        listed = self.lists.get(client_id)
        if listed is not None and listed.status == "EXECUTING":
            raise ApiError(400, -2010, "Duplicate order sent.")

    def _create(self, symbol: str, client_id: str, side: str, kind: str, qty: Decimal) -> FakeOrder:
        order = FakeOrder(
            symbol=symbol,
            order_id=self._next_order_id,
            client_id=client_id,
            side=side,
            type=kind,
            quantity=qty,
            time=self.now,
        )
        order.peak = self.prices[symbol]
        self._next_order_id += 1
        return order

    def _new_order(self, params: dict[str, str]) -> Any:
        symbol, side, kind = params["symbol"], params["side"], params["type"]
        self._symbol(symbol)
        client_id = params.get("newClientOrderId", f"auto-{self._next_order_id}")
        self._check_new_id(client_id)
        quantity = Decimal(params["quantity"]) if "quantity" in params else None
        quote_qty = Decimal(params["quoteOrderQty"]) if "quoteOrderQty" in params else None
        price = Decimal(params["price"]) if "price" in params else None
        stop = Decimal(params["stopPrice"]) if "stopPrice" in params else None
        trailing = int(params["trailingDelta"]) if "trailingDelta" in params else None
        last = self.prices[symbol]
        base = self._base(symbol)
        if kind == "MARKET":
            if quote_qty is not None:
                self._check(symbol, side, None, None, quote_qty, market=True)
                if self._bal("USDT")[0] < quote_qty:
                    raise ApiError(
                        400, -2010, "Account has insufficient balance for requested action."
                    )
            else:
                assert quantity is not None
                self._check(symbol, side, quantity, None, quantity * last, market=True)
                held = self._bal(base if side == "SELL" else "USDT")[0]
                if held < (quantity if side == "SELL" else quantity * last):
                    raise ApiError(
                        400, -2010, "Account has insufficient balance for requested action."
                    )
        elif kind == "LIMIT_MAKER":
            assert quantity is not None
            assert price is not None
            self._check(symbol, side, quantity, price, quantity * price)
            if (side == "BUY" and price >= last) or (side == "SELL" and price <= last):
                raise ApiError(400, -2010, "Order would immediately match and take.")
        elif kind == "STOP_LOSS":
            assert quantity is not None
            if stop is None and trailing is None:
                raise ApiError(400, -1102, "Mandatory parameter 'stopPrice' was not sent.")
            self._check(symbol, side, quantity, stop, quantity * (stop or last))
            if trailing is not None:
                self._check_trailing(symbol, trailing)
            if stop is not None and (
                (side == "SELL" and stop >= last) or (side == "BUY" and stop <= last)
            ):
                raise ApiError(400, -2021, "Order would trigger immediately.")
        else:
            raise ApiError(400, -1116, "Invalid orderType.")
        order = self._create(symbol, client_id, side, kind, quantity or ZERO)
        order.quote_qty, order.price = quote_qty, price or ZERO
        order.stop_price, order.trailing_delta = stop, trailing
        self.orders[client_id] = order
        full = params.get("newOrderRespType") == "FULL"
        if kind == "MARKET":
            self._report(order, "NEW")
            if symbol in self.hold_market:
                if side == "BUY":
                    amount = quote_qty if quote_qty is not None else (quantity or ZERO) * last
                    self._lock("USDT", amount)
                    order.locked = amount
                else:
                    self._lock(base, order.quantity)
                    order.locked = order.quantity
            else:
                self._market_fill(order)
        else:
            self._lock(base if side == "SELL" else "USDT", order.quantity)
            order.locked = order.quantity
            self._report(order, "NEW")
            self._account_event(symbol)
        return order.json(place=True, full=full)

    def _new_oco(self, params: dict[str, str]) -> Any:
        symbol, side = params["symbol"], params["side"]
        quantity = Decimal(params["quantity"])
        list_client_id = params.get("listClientOrderId", f"list-{self._next_list_id}")
        above_type, below_type = params["aboveType"], params["belowType"]
        above_price = Decimal(params["abovePrice"]) if "abovePrice" in params else None
        below_stop = Decimal(params["belowStopPrice"]) if "belowStopPrice" in params else None
        below_trailing = (
            int(params["belowTrailingDelta"]) if "belowTrailingDelta" in params else None
        )
        if side != "SELL" or above_type != "LIMIT_MAKER" or below_type != "STOP_LOSS":
            raise ApiError(400, -1013, "Only the SELL LIMIT_MAKER + STOP_LOSS bracket is faked.")
        if above_price is None or (below_stop is None and below_trailing is None):
            raise ApiError(400, -1102, "Mandatory parameter was not sent.")
        last = self.prices[symbol]
        self._check(symbol, side, quantity, above_price, quantity * above_price)
        if below_stop is not None:
            self._check(symbol, side, None, below_stop, quantity * below_stop)
        if below_trailing is not None:
            self._check_trailing(symbol, below_trailing)
        if not (above_price > last and (below_stop is None or last > below_stop)):
            raise ApiError(
                400, -1013, "The relationship of the prices for the orders is not correct."
            )
        ids = (
            params.get("aboveClientOrderId", f"{list_client_id}-a"),
            params.get("belowClientOrderId", f"{list_client_id}-b"),
        )
        for client_id in (list_client_id, *ids):
            self._check_new_id(client_id)
        base = self._base(symbol)
        self._lock(base, quantity)  # one lock for the pair
        list_id = self._next_list_id
        self._next_list_id += 1
        above = self._create(symbol, ids[0], side, above_type, quantity)
        above.price = above_price
        below = self._create(symbol, ids[1], side, below_type, quantity)
        below.stop_price, below.trailing_delta = below_stop, below_trailing
        above.list_id = below.list_id = list_id
        above.locked = below.locked = quantity
        self.orders[ids[0]], self.orders[ids[1]] = above, below
        listed = FakeList(list_id, list_client_id, symbol, [above, below], self.now)
        self.lists[list_client_id] = listed
        self._report(below, "NEW")
        self._report(above, "NEW")
        self._list_event(listed)
        listed.status_type = "EXEC_STARTED"
        self._account_event(symbol)
        return listed.json(reports=True)

    def _find_order(self, params: dict[str, str], *, cancel: bool) -> FakeOrder:
        order = self.orders.get(params.get("origClientOrderId", ""))
        if order is None or order.symbol != params.get("symbol"):
            if cancel:
                raise ApiError(400, -2011, "Unknown order sent.")
            raise ApiError(400, -2013, "Order does not exist.")
        return order

    def _cancel_one(self, order: FakeOrder, cancel_id: str) -> None:
        if order.locked > 0:
            asset = self._base(order.symbol) if order.side == "SELL" else "USDT"
            self._unlock(asset, order.locked)
        order.locked = ZERO
        order.status = "CANCELED"
        order.update_time = self.now
        self._report(order, "CANCELED", cancel_id=cancel_id)

    def _cancel_list(self, listed: FakeList) -> None:
        if listed.status != "EXECUTING":
            raise ApiError(400, -2011, "Unknown order sent.")
        base = self._base(listed.symbol)
        locked = max((leg.locked for leg in listed.legs if leg.open), default=ZERO)
        if locked > 0:
            self._unlock(base, locked)
        for leg in listed.legs:
            if leg.open:
                leg.locked = ZERO
                leg.status = "CANCELED"
                leg.update_time = self.now
                self._report(leg, "CANCELED", cancel_id=f"cancel-{leg.order_id}")
        listed.status, listed.status_type = "ALL_DONE", "ALL_DONE"
        self._list_event(listed)
        self._account_event(listed.symbol)

    def _cancel_order(self, params: dict[str, str]) -> Any:
        order = self._find_order(params, cancel=True)
        if not order.open:
            raise ApiError(400, -2011, "Unknown order sent.")
        if order.list_id is not None:  # cancelling one leg cancels the list
            self._cancel_list(self._list_of(order))
        else:
            self._cancel_one(order, f"cancel-{order.order_id}")
            self._account_event(order.symbol)
        out = order.json(place=True)
        out["origClientOrderId"] = order.client_id
        out["clientOrderId"] = f"cancel-{order.order_id}"
        return out

    def _cancel_order_list(self, params: dict[str, str]) -> Any:
        listed = self._find_list(params.get("listClientOrderId", ""))
        self._cancel_list(listed)
        return listed.json(reports=True)

    def _find_list(self, client_id: str) -> FakeList:
        listed = self.lists.get(client_id)
        if listed is None:
            raise ApiError(400, -2011, "Order list does not exist.")
        return listed

    def _get_order(self, params: dict[str, str]) -> Any:
        return self._find_order(params, cancel=False).json(place=False)

    def _open_orders(self, params: dict[str, str]) -> Any:
        symbol = params.get("symbol")
        return [
            o.json(place=False)
            for o in self.orders.values()
            if o.open and symbol in (None, o.symbol)
        ]

    def _open_order_lists(self, params: dict[str, str]) -> Any:
        return [ls.json(reports=False) for ls in self.lists.values() if ls.status == "EXECUTING"]

    def _get_order_list(self, params: dict[str, str]) -> Any:
        return self._find_list(params.get("origClientOrderId", "")).json(reports=False)

    def _my_trades(self, params: dict[str, str]) -> Any:
        self._symbol(params["symbol"])
        wanted = int(params["orderId"]) if "orderId" in params else None
        return [
            t
            for t in self.trades
            if t["symbol"] == params["symbol"] and wanted in (None, t["orderId"])
        ]

    _ROUTES = {  # noqa: RUF012
        ("GET", "/api/v3/time"): _time,
        ("GET", "/api/v3/exchangeInfo"): _exchange_info,
        ("GET", "/api/v3/ticker/price"): _ticker,
        ("GET", "/api/v3/account"): _account,
        ("POST", "/api/v3/order"): _new_order,
        ("POST", "/api/v3/orderList/oco"): _new_oco,
        ("DELETE", "/api/v3/order"): _cancel_order,
        ("DELETE", "/api/v3/orderList"): _cancel_order_list,
        ("GET", "/api/v3/order"): _get_order,
        ("GET", "/api/v3/openOrders"): _open_orders,
        ("GET", "/api/v3/openOrderList"): _open_order_lists,
        ("GET", "/api/v3/orderList"): _get_order_list,
        ("GET", "/api/v3/myTrades"): _my_trades,
    }


__all__ = ["API_KEY", "API_SECRET", "SYMBOLS", "FakeExchange", "FakeList", "FakeOrder"]
