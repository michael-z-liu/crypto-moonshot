"""A fake Binance spot testnet REST server (an `httpx.MockTransport` handler) for client tests.

It checks signatures with a fake secret (-1022) and timestamps against its own clock (-1021),
serves `time`, `exchangeInfo` and `account`, accepts orders, OCO lists, cancels and queries with
simple deterministic matching (a market order fills at the symbol's price; resting orders fill
when `set_price` crosses them), and returns the weight and order-count headers. Faults (429,
418, 5xx, -1007, timeouts, any error code) are injected with `fail` and `timeout`.

Nothing here talks to a network. The keys are obviously fake.
"""

from __future__ import annotations

import hashlib
import hmac
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qsl

import httpx
from pydantic import SecretStr

from crypto_moonshot.exchange.client import SignedClient

API_KEY = "fake-test-key-not-a-real-key"
API_SECRET = "fake-test-secret-not-a-real-secret"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
COMMISSION_RATE = Decimal("0.001")

WEIGHTS = {
    ("GET", "/api/v3/time"): 1,
    ("GET", "/api/v3/exchangeInfo"): 20,
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


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


class FakeClock:
    """A controllable clock whose `sleep` advances it, recording every wait."""

    def __init__(self, now: datetime = T0) -> None:
        self.current = now
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.current

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.current += timedelta(seconds=seconds)


def symbol_json(
    symbol: str,
    base: str,
    *,
    tick: str,
    step: str,
    min_qty: str,
    min_notional: str = "5.00000000",
    max_qty: str = "9000.00000000",
    status: str = "TRADING",
) -> dict[str, Any]:
    """One `symbols[]` entry of `exchangeInfo`, with the filters a real USDT pair carries."""
    return {
        "symbol": symbol,
        "status": status,
        "baseAsset": base,
        "baseAssetPrecision": 8,
        "quoteAsset": "USDT",
        "quotePrecision": 8,
        "orderTypes": ["LIMIT", "LIMIT_MAKER", "MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT"],
        "quoteOrderQtyMarketAllowed": True,
        "ocoAllowed": True,
        "filters": [
            {
                "filterType": "PRICE_FILTER",
                "minPrice": tick,
                "maxPrice": "1000000.00000000",
                "tickSize": tick,
            },
            {
                "filterType": "LOT_SIZE",
                "minQty": min_qty,
                "maxQty": max_qty,
                "stepSize": step,
            },
            {
                "filterType": "MARKET_LOT_SIZE",
                "minQty": "0.00000000",
                "maxQty": "100.00000000",
                "stepSize": "0.00000000",
            },
            {
                "filterType": "NOTIONAL",
                "minNotional": min_notional,
                "applyMinToMarket": True,
                "maxNotional": "9000000.00000000",
                "applyMaxToMarket": False,
                "avgPriceMins": 5,
            },
            {
                "filterType": "PERCENT_PRICE_BY_SIDE",
                "bidMultiplierUp": "5",
                "bidMultiplierDown": "0.2",
                "askMultiplierUp": "5",
                "askMultiplierDown": "0.2",
                "avgPriceMins": 5,
            },
            {
                "filterType": "TRAILING_DELTA",
                "minTrailingAboveDelta": 10,
                "maxTrailingAboveDelta": 2000,
                "minTrailingBelowDelta": 10,
                "maxTrailingBelowDelta": 2000,
            },
            {"filterType": "MAX_NUM_ORDERS", "maxNumOrders": 200},
            {"filterType": "MAX_NUM_ALGO_ORDERS", "maxNumAlgoOrders": 5},
        ],
    }


DEFAULT_SYMBOLS = [
    symbol_json("BTCUSDT", "BTC", tick="0.01000000", step="0.00001000", min_qty="0.00001000"),
    symbol_json("ETHUSDT", "ETH", tick="0.01000000", step="0.00010000", min_qty="0.00010000"),
    symbol_json(
        "SHIBUSDT",
        "SHIB",
        tick="0.00000001",
        step="1.00000000",
        min_qty="1.00000000",
        max_qty="9000000000.00000000",
    ),
    symbol_json("XRPUSDT", "XRP", tick="0.00000100", step="0.10000000", min_qty="0.10000000"),
    symbol_json(
        "OLDUSDT",
        "OLD",
        tick="0.01000000",
        step="0.01000000",
        min_qty="0.01000000",
        status="BREAK",
    ),
]


@dataclass
class Fault:
    """A scripted failure: the next `times` requests whose path contains `path` get it."""

    path: str
    method: str | None = None
    times: int = 1
    status: int = 500
    body: dict[str, Any] | str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    raises: Exception | None = None
    """Raised from the transport instead of returning a response (a timeout)."""


@dataclass
class _Order:
    symbol: str
    order_id: int
    client_id: str
    side: str
    type: str
    quantity: Decimal
    time: datetime
    price: Decimal = Decimal(0)
    stop_price: Decimal | None = None
    trailing_delta: int | None = None
    quote_qty: Decimal | None = None
    list_id: int | None = None
    status: str = "NEW"
    executed: Decimal = Decimal(0)
    cumulative_quote: Decimal = Decimal(0)
    fills: list[dict[str, str]] = field(default_factory=list)
    peak: Decimal | None = None
    update_time: datetime | None = None

    def json(self, *, place: bool, full: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "symbol": self.symbol,
            "orderId": self.order_id,
            "orderListId": -1 if self.list_id is None else self.list_id,
            "clientOrderId": self.client_id,
            "price": f"{self.price:f}",
            "origQty": f"{self.quantity:f}",
            "executedQty": f"{self.executed:f}",
            "cummulativeQuoteQty": f"{self.cumulative_quote:f}",
            "status": self.status,
            "timeInForce": "GTC",
            "type": self.type,
            "side": self.side,
        }
        if self.stop_price is not None:
            out["stopPrice"] = f"{self.stop_price:f}"
        if self.trailing_delta is not None:
            out["trailingDelta"] = self.trailing_delta
        if place:
            out["transactTime"] = to_ms(self.time)
        else:
            out["time"] = to_ms(self.time)
            out["updateTime"] = to_ms(self.update_time or self.time)
        if full:
            out["fills"] = list(self.fills)
        return out


@dataclass
class _OrderList:
    list_id: int
    client_id: str
    symbol: str
    legs: list[_Order]
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


class FakeTestnet:
    """The handler: pass an instance as `httpx.MockTransport(fake)`.

    Attributes tests use: `requests` (every request, in order), `prices` (last price per
    symbol), `balances` (`asset -> [free, locked]`), `orders`, `order_lists`, `trades`,
    `skew` (server clock minus the client's clock), `rate_limited` (429s it sent),
    `max_weight_1m` and `max_orders_10s` (highest usage it saw), and the limits
    `weight_limit`, `order_limit_10s`, above which it answers 429 / -1015.
    """

    def __init__(
        self,
        clock: FakeClock | None = None,
        *,
        api_key: str = API_KEY,
        api_secret: str = API_SECRET,
        skew: timedelta = timedelta(0),
        weight_limit: int | None = None,
        order_limit_10s: int | None = None,
        symbols: list[dict[str, Any]] | None = None,
    ) -> None:
        self.clock = clock or FakeClock()
        self.api_key = api_key
        self.api_secret = api_secret
        self.skew = skew
        self.weight_limit = weight_limit
        self.order_limit_10s = order_limit_10s
        self.symbols = symbols if symbols is not None else DEFAULT_SYMBOLS
        self.requests: list[httpx.Request] = []
        self.faults: list[Fault] = []
        self.prices: dict[str, Decimal] = {
            "BTCUSDT": Decimal("30000.00"),
            "ETHUSDT": Decimal("2000.00"),
            "SHIBUSDT": Decimal("0.00001000"),
            "XRPUSDT": Decimal("0.500000"),
        }
        self.balances: dict[str, list[Decimal]] = {
            "USDT": [Decimal("10000"), Decimal(0)],
            "BTC": [Decimal(0), Decimal(0)],
            "ETH": [Decimal(0), Decimal(0)],
        }
        self.orders: dict[str, _Order] = {}
        self.order_lists: dict[str, _OrderList] = {}
        self.trades: list[dict[str, Any]] = []
        self.rate_limited = 0
        self.max_weight_1m = 0
        self.max_orders_10s = 0
        self._weight: dict[datetime, int] = defaultdict(int)
        self._order_times: list[datetime] = []
        self._next_order_id = 1000
        self._next_list_id = 1
        self._next_trade_id = 1

    # -- test controls ---------------------------------------------------------------------------

    @property
    def server_now(self) -> datetime:
        return self.clock.now() + self.skew

    def fail(
        self,
        path: str,
        *,
        method: str | None = None,
        status: int = 500,
        code: int | None = None,
        msg: str = "",
        times: int = 1,
        headers: dict[str, str] | None = None,
        body: str | None = None,
    ) -> None:
        """Answer the next `times` requests to paths containing `path` with an error."""
        payload: dict[str, Any] | str | None = body
        if payload is None and code is not None:
            payload = {"code": code, "msg": msg}
        self.faults.append(Fault(path, method, times, status, payload, headers or {}))

    def timeout(self, path: str, *, method: str | None = None, times: int = 1) -> None:
        """Make the next `times` requests to `path` time out (no response)."""
        self.faults.append(Fault(path, method, times, raises=httpx.ReadTimeout("fake timeout")))

    def ban(self) -> None:
        """Answer every request from now on with HTTP 418."""
        self.fail("", status=418, code=-1003, msg="Way too many requests; banned.", times=10**9)

    def set_price(self, symbol: str, price: Decimal) -> None:
        """Move the market and fill whatever resting orders the move reaches."""
        self.prices[symbol] = price
        for order in list(self.orders.values()):
            if order.symbol == symbol and order.status == "NEW":
                self._try_fill_resting(order, price)

    def requests_to(self, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == path]

    # -- the handler -----------------------------------------------------------------------------

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for fault in self.faults:
            if (
                fault.times > 0
                and fault.path in request.url.path
                and fault.method in (None, request.method)
            ):
                fault.times -= 1
                if fault.raises is not None:
                    raise fault.raises
                return self._fault_response(fault)
        key = (request.method, request.url.path)
        if key not in WEIGHTS:
            return self._error(404, -1102, "No such endpoint.")
        limited = self._count_weight(WEIGHTS[key])
        if limited is not None:
            return limited
        try:
            params = self._authenticate(request) if key not in _PUBLIC else {}
            if key in _PUBLIC:
                params = dict(parse_qsl(request.url.query.decode()))
            body = self._ROUTES[key](self, params)
        except _ApiError as err:
            return self._error(err.status, err.code, err.msg)
        headers = self._usage_headers(order=key in _ORDER_ENDPOINTS)
        return httpx.Response(200, json=body, headers=headers)

    # -- plumbing --------------------------------------------------------------------------------

    def _fault_response(self, fault: Fault) -> httpx.Response:
        headers = {**fault.headers, **self._usage_headers(order=False)}
        if isinstance(fault.body, str):
            return httpx.Response(fault.status, text=fault.body, headers=headers)
        return httpx.Response(fault.status, json=fault.body, headers=headers)

    def _error(self, status: int, code: int, msg: str) -> httpx.Response:
        return httpx.Response(
            status, json={"code": code, "msg": msg}, headers=self._usage_headers(order=False)
        )

    def _minute(self) -> datetime:
        return self.server_now.replace(second=0, microsecond=0)

    def _count_weight(self, weight: int) -> httpx.Response | None:
        minute = self._minute()
        if self.weight_limit is not None and self._weight[minute] + weight > self.weight_limit:
            self.rate_limited += 1
            return httpx.Response(
                429,
                json={"code": -1003, "msg": "Too much request weight used."},
                headers={"Retry-After": "30", **self._usage_headers(order=False)},
            )
        self._weight[minute] += weight
        self.max_weight_1m = max(self.max_weight_1m, self._weight[minute])
        return None

    def _orders_10s(self) -> int:
        cutoff = self.server_now - timedelta(seconds=10)
        return sum(1 for t in self._order_times if t > cutoff)

    def _orders_1d(self) -> int:
        return sum(1 for t in self._order_times if t.date() == self.server_now.date())

    def _usage_headers(self, *, order: bool) -> dict[str, str]:
        headers = {"X-MBX-USED-WEIGHT-1M": str(self._weight[self._minute()])}
        if order:
            headers["X-MBX-ORDER-COUNT-10S"] = str(self._orders_10s())
            headers["X-MBX-ORDER-COUNT-1D"] = str(self._orders_1d())
        return headers

    def _count_orders(self, n: int) -> None:
        if self.order_limit_10s is not None and self._orders_10s() + n > self.order_limit_10s:
            self.rate_limited += 1
            raise _ApiError(429, -1015, "Too many new orders; current limit is 50 orders per 10s.")
        self._order_times.extend([self.server_now] * n)
        self.max_orders_10s = max(self.max_orders_10s, self._orders_10s())

    def _authenticate(self, request: httpx.Request) -> dict[str, str]:
        supplied = request.headers.get("X-MBX-APIKEY")
        if supplied is None:
            raise _ApiError(401, -2014, "API-key format invalid.")
        raw = request.url.query.decode()
        head, sep, signature = raw.rpartition("&signature=")
        if not sep:
            raise _ApiError(400, -1102, "Mandatory parameter 'signature' was not sent.")
        params = dict(parse_qsl(head))
        if "timestamp" not in params:
            raise _ApiError(400, -1102, "Mandatory parameter 'timestamp' was not sent.")
        expected = hmac.new(self.api_secret.encode(), head.encode(), hashlib.sha256).hexdigest()
        if supplied != self.api_key:
            raise _ApiError(401, -2015, "Invalid API-key, IP, or permissions for action.")
        if not hmac.compare_digest(expected, signature):
            raise _ApiError(400, -1022, "Signature for this request is not valid.")
        timestamp = int(params["timestamp"])
        window = int(params.get("recvWindow", "5000"))
        server = to_ms(self.server_now)
        if not (timestamp < server + 1000 and server - timestamp <= window):
            raise _ApiError(400, -1021, "Timestamp for this request is outside of the recvWindow.")
        return params

    # -- rules -----------------------------------------------------------------------------------

    def _symbol(self, name: str) -> dict[str, Any]:
        for s in self.symbols:
            if s["symbol"] == name:
                return s
        raise _ApiError(400, -1121, "Invalid symbol.")

    def _filter(self, symbol: str, kind: str) -> dict[str, Any]:
        for f in self._symbol(symbol)["filters"]:
            if f["filterType"] == kind:
                return dict(f)
        return {}

    def _check_filters(
        self, symbol: str, quantity: Decimal | None, price: Decimal | None, notional: Decimal
    ) -> None:
        if self._symbol(symbol)["status"] != "TRADING":
            raise _ApiError(400, -1013, "Market is closed.")
        if price is not None:
            pf = self._filter(symbol, "PRICE_FILTER")
            tick = Decimal(pf["tickSize"])
            if price < Decimal(pf["minPrice"]) or (tick and price % tick != 0):
                raise _ApiError(400, -1013, "Filter failure: PRICE_FILTER")
        if quantity is not None:
            lot = self._filter(symbol, "LOT_SIZE")
            step = Decimal(lot["stepSize"])
            if (
                quantity < Decimal(lot["minQty"])
                or quantity > Decimal(lot["maxQty"])
                or (step and quantity % step != 0)
            ):
                raise _ApiError(400, -1013, "Filter failure: LOT_SIZE")
        if notional < Decimal(self._filter(symbol, "NOTIONAL")["minNotional"]):
            raise _ApiError(400, -1013, "Filter failure: NOTIONAL")

    # -- balances and fills ----------------------------------------------------------------------

    def _bal(self, asset: str) -> list[Decimal]:
        return self.balances.setdefault(asset, [Decimal(0), Decimal(0)])

    def _lock(self, asset: str, amount: Decimal) -> None:
        bal = self._bal(asset)
        if bal[0] < amount:
            raise _ApiError(400, -2010, "Account has insufficient balance for requested action.")
        bal[0] -= amount
        bal[1] += amount

    def _unlock(self, asset: str, amount: Decimal) -> None:
        bal = self._bal(asset)
        bal[1] -= amount
        bal[0] += amount

    def _base(self, symbol: str) -> str:
        return str(self._symbol(symbol)["baseAsset"])

    def _fill(self, order: _Order, quantity: Decimal, price: Decimal, *, locked: bool) -> None:
        """Execute `quantity` at `price`, settle balances, record the fill and the trade."""
        base = self._base(order.symbol)
        value = quantity * price
        if order.side == "BUY":
            fee, fee_asset = quantity * COMMISSION_RATE, base
            if locked:
                self._bal("USDT")[1] -= value
            else:
                self._lock("USDT", value)
                self._bal("USDT")[1] -= value
            self._bal(base)[0] += quantity - fee
        else:
            fee, fee_asset = value * COMMISSION_RATE, "USDT"
            if locked:
                self._bal(base)[1] -= quantity
            else:
                self._lock(base, quantity)
                self._bal(base)[1] -= quantity
            self._bal("USDT")[0] += value - fee
        trade_id = self._next_trade_id
        self._next_trade_id += 1
        order.executed += quantity
        order.cumulative_quote += value
        order.update_time = self.server_now
        order.fills.append(
            {
                "price": f"{price:f}",
                "qty": f"{quantity:f}",
                "commission": f"{fee:f}",
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
                "price": f"{price:f}",
                "qty": f"{quantity:f}",
                "quoteQty": f"{value:f}",
                "commission": f"{fee:f}",
                "commissionAsset": fee_asset,
                "time": to_ms(self.server_now),
                "isBuyer": order.side == "BUY",
                "isMaker": order.type == "LIMIT_MAKER",
                "isBestMatch": True,
            }
        )
        if order.executed >= order.quantity or order.quote_qty is not None:
            order.status = "FILLED"

    def _market_fill(self, order: _Order, *, locked: bool = False) -> None:
        """A market fill at the symbol's price, in two fills (the second one tick worse)."""
        price = self.prices[order.symbol]
        tick = Decimal(self._filter(order.symbol, "PRICE_FILTER")["tickSize"])
        step = Decimal(self._filter(order.symbol, "LOT_SIZE")["stepSize"])
        sign = 1 if order.side == "BUY" else -1
        half = (order.quantity / 2 // step) * step if step else order.quantity / 2
        if half > 0 and half < order.quantity:
            self._fill(order, half, price, locked=locked)
            self._fill(order, order.quantity - half, price + sign * tick, locked=locked)
        else:
            self._fill(order, order.quantity, price, locked=locked)
        order.status = "FILLED"

    def _try_fill_resting(self, order: _Order, price: Decimal) -> None:
        trigger: Decimal | None = None
        if order.type == "LIMIT_MAKER":
            if (order.side == "SELL" and price >= order.price) or (
                order.side == "BUY" and price <= order.price
            ):
                self._fill(order, order.quantity, order.price, locked=True)
                self._finish(order)
            return
        if order.type == "STOP_LOSS":
            if order.trailing_delta is not None:
                order.peak = max(order.peak or price, price)
                trigger = order.peak * (1 - Decimal(order.trailing_delta) / 10000)
                if order.stop_price is not None:
                    trigger = max(trigger, order.stop_price)
            else:
                trigger = order.stop_price
            if trigger is not None and price <= trigger:
                self._fill(order, order.quantity, price, locked=True)
                self._finish(order)

    def _finish(self, order: _Order) -> None:
        """After a resting order filled: close its OCO list and cancel the other leg."""
        if order.list_id is None:
            return
        listed = next(ls for ls in self.order_lists.values() if ls.list_id == order.list_id)
        listed.status, listed.status_type = "ALL_DONE", "ALL_DONE"
        for leg in listed.legs:
            if leg is not order and leg.status == "NEW":
                leg.status = "EXPIRED"
                leg.update_time = self.server_now

    # -- endpoints -------------------------------------------------------------------------------

    def _time(self, params: dict[str, str]) -> Any:
        return {"serverTime": to_ms(self.server_now)}

    def _exchange_info(self, params: dict[str, str]) -> Any:
        return {
            "timezone": "UTC",
            "serverTime": to_ms(self.server_now),
            "rateLimits": [],
            "symbols": self.symbols,
        }

    def _account(self, params: dict[str, str]) -> Any:
        return {
            "canTrade": True,
            "updateTime": to_ms(self.server_now),
            "balances": [
                {"asset": a, "free": f"{b[0]:f}", "locked": f"{b[1]:f}"}
                for a, b in sorted(self.balances.items())
            ],
        }

    def _new_order(self, params: dict[str, str]) -> Any:
        symbol, side, kind = params["symbol"], params["side"], params["type"]
        client_id = params.get("newClientOrderId", f"auto-{self._next_order_id}")
        self._check_new_id(client_id)
        quantity = Decimal(params["quantity"]) if "quantity" in params else None
        quote_qty = Decimal(params["quoteOrderQty"]) if "quoteOrderQty" in params else None
        price = Decimal(params["price"]) if "price" in params else None
        stop = Decimal(params["stopPrice"]) if "stopPrice" in params else None
        trailing = int(params["trailingDelta"]) if "trailingDelta" in params else None
        last = self.prices[symbol]
        self._symbol(symbol)
        if kind == "MARKET":
            if quote_qty is not None:
                step = Decimal(self._filter(symbol, "LOT_SIZE")["stepSize"])
                quantity = (quote_qty / last // step) * step
                self._check_filters(symbol, quantity, None, quantity * last)
            elif quantity is not None:
                self._check_filters(symbol, quantity, None, quantity * last)
        elif kind == "LIMIT_MAKER":
            assert quantity is not None
            assert price is not None
            self._check_filters(symbol, quantity, price, quantity * price)
            if (side == "BUY" and price >= last) or (side == "SELL" and price <= last):
                raise _ApiError(400, -2010, "Order would immediately match and take.")
        elif kind == "STOP_LOSS":
            assert quantity is not None
            if stop is None and trailing is None:
                raise _ApiError(400, -1102, "Mandatory parameter 'stopPrice' was not sent.")
            self._check_filters(symbol, quantity, stop, quantity * (stop or last))
            if stop is not None and (
                (side == "SELL" and stop >= last) or (side == "BUY" and stop <= last)
            ):
                raise _ApiError(400, -2021, "Order would trigger immediately.")
        else:
            raise _ApiError(400, -1116, "Invalid orderType.")
        assert quantity is not None
        self._count_orders(1)
        order = self._create(symbol, client_id, side, kind, quantity, price, stop, trailing)
        order.quote_qty = quote_qty
        if kind == "MARKET":
            if side == "BUY":
                need = quantity * last
                if self._bal("USDT")[0] < need:
                    raise _ApiError(
                        400, -2010, "Account has insufficient balance for requested action."
                    )
            elif self._bal(self._base(symbol))[0] < quantity:
                raise _ApiError(
                    400, -2010, "Account has insufficient balance for requested action."
                )
            self.orders[client_id] = order
            self._market_fill(order)
        else:
            self._reserve(order)
            self.orders[client_id] = order
        return order.json(place=True, full=params.get("newOrderRespType") == "FULL")

    def _create(
        self,
        symbol: str,
        client_id: str,
        side: str,
        kind: str,
        quantity: Decimal,
        price: Decimal | None,
        stop: Decimal | None,
        trailing: int | None,
    ) -> _Order:
        order = _Order(
            symbol=symbol,
            order_id=self._next_order_id,
            client_id=client_id,
            side=side,
            type=kind,
            quantity=quantity,
            time=self.server_now,
            price=price or Decimal(0),
            stop_price=stop,
            trailing_delta=trailing,
        )
        order.peak = self.prices[symbol]
        self._next_order_id += 1
        return order

    def _reserve(self, order: _Order) -> None:
        """Lock what a resting order needs (base for sells, quote for buys)."""
        if order.side == "SELL":
            self._lock(self._base(order.symbol), order.quantity)
        else:
            self._lock("USDT", order.quantity * (order.price or order.stop_price or Decimal(0)))

    def _release(self, order: _Order) -> None:
        if order.status != "NEW":
            return
        if order.side == "SELL":
            self._unlock(self._base(order.symbol), order.quantity)
        else:
            self._unlock("USDT", order.quantity * (order.price or order.stop_price or Decimal(0)))

    def _check_new_id(self, client_id: str) -> None:
        existing = self.orders.get(client_id)
        if existing is not None and existing.status == "NEW":
            raise _ApiError(400, -2010, "Duplicate order sent.")

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
            raise _ApiError(400, -1013, "Only the SELL LIMIT_MAKER + STOP_LOSS bracket is faked.")
        if above_price is None or (below_stop is None and below_trailing is None):
            raise _ApiError(400, -1102, "Mandatory parameter was not sent.")
        last = self.prices[symbol]
        self._check_filters(symbol, quantity, above_price, quantity * above_price)
        if below_stop is not None:
            self._check_filters(symbol, None, below_stop, quantity * below_stop)
        if not (above_price > last and (below_stop is None or last > below_stop)):
            raise _ApiError(
                400, -1013, "The relationship of the prices for the orders is not correct."
            )
        ids = (
            params.get("aboveClientOrderId", f"{list_client_id}-a"),
            params.get("belowClientOrderId", f"{list_client_id}-b"),
        )
        for client_id in (list_client_id, *ids):
            self._check_new_id(client_id)
        self._count_orders(2)
        self._lock(self._base(symbol), quantity)  # one lock for the pair
        list_id = self._next_list_id
        self._next_list_id += 1
        above = self._create(symbol, ids[0], side, above_type, quantity, above_price, None, None)
        below = self._create(
            symbol, ids[1], side, below_type, quantity, None, below_stop, below_trailing
        )
        above.list_id = below.list_id = list_id
        self.orders[ids[0]], self.orders[ids[1]] = above, below
        listed = _OrderList(list_id, list_client_id, symbol, [above, below], self.server_now)
        self.order_lists[list_client_id] = listed
        return listed.json(reports=True)

    def _find_order(self, params: dict[str, str], *, cancel: bool) -> _Order:
        order = self.orders.get(params.get("origClientOrderId", ""))
        if order is None or order.symbol != params.get("symbol"):
            if cancel:
                raise _ApiError(400, -2011, "Unknown order sent.")
            raise _ApiError(400, -2013, "Order does not exist.")
        return order

    def _cancel_order(self, params: dict[str, str]) -> Any:
        order = self._find_order(params, cancel=True)
        if order.status != "NEW":
            raise _ApiError(400, -2011, "Unknown order sent.")
        if order.list_id is not None:  # cancelling one leg cancels the list
            listed = next(ls for ls in self.order_lists.values() if ls.list_id == order.list_id)
            self._cancel_list(listed)
        else:
            self._release(order)
            order.status = "CANCELED"
            order.update_time = self.server_now
        out = order.json(place=True)
        out["origClientOrderId"] = order.client_id
        out["clientOrderId"] = f"cancel-{order.order_id}"
        return out

    def _cancel_list(self, listed: _OrderList) -> None:
        if listed.status != "EXECUTING":
            raise _ApiError(400, -2011, "Unknown order sent.")
        self._unlock(self._base(listed.symbol), listed.legs[0].quantity)
        for leg in listed.legs:
            leg.status = "CANCELED"
            leg.update_time = self.server_now
        listed.status, listed.status_type = "ALL_DONE", "ALL_DONE"

    def _cancel_order_list(self, params: dict[str, str]) -> Any:
        listed = self._find_list(params.get("listClientOrderId", ""))
        self._cancel_list(listed)
        return listed.json(reports=True)

    def _find_list(self, client_id: str) -> _OrderList:
        listed = self.order_lists.get(client_id)
        if listed is None:
            raise _ApiError(400, -2013, "Order list does not exist.")
        return listed

    def _get_order(self, params: dict[str, str]) -> Any:
        return self._find_order(params, cancel=False).json(place=False)

    def _open_orders(self, params: dict[str, str]) -> Any:
        symbol = params.get("symbol")
        return [
            o.json(place=False)
            for o in self.orders.values()
            if o.status == "NEW" and symbol in (None, o.symbol)
        ]

    def _open_order_lists(self, params: dict[str, str]) -> Any:
        return [
            ls.json(reports=False) for ls in self.order_lists.values() if ls.status == "EXECUTING"
        ]

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


_PUBLIC = {("GET", "/api/v3/time"), ("GET", "/api/v3/exchangeInfo")}
_ORDER_ENDPOINTS = {("POST", "/api/v3/order"), ("POST", "/api/v3/orderList/oco")}


class _ApiError(Exception):
    def __init__(self, status: int, code: int, msg: str) -> None:
        super().__init__(msg)
        self.status, self.code, self.msg = status, code, msg


def fake_client_http(fake: FakeTestnet) -> httpx.AsyncClient:
    """An `httpx.AsyncClient` wired to `fake` (never the network)."""
    return httpx.AsyncClient(transport=httpx.MockTransport(fake))


def make_client(fake: FakeTestnet, **kwargs: Any) -> SignedClient:
    """A `SignedClient` on `fake`'s transport and clock, with the fake's keys (overridable)."""
    options: dict[str, Any] = {
        "http": fake_client_http(fake),
        "now": fake.clock.now,
        "sleep": fake.clock.sleep,
        "base_url": "https://testnet.fake",
    }
    options.update(kwargs)
    return SignedClient(
        options.pop("api_key", SecretStr(fake.api_key)),
        options.pop("api_secret", SecretStr(fake.api_secret)),
        **options,
    )


__all__ = [
    "API_KEY",
    "API_SECRET",
    "T0",
    "FakeClock",
    "FakeTestnet",
    "fake_client_http",
    "make_client",
]
