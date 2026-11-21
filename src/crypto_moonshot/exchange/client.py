"""Signed async client for the Binance spot REST API (built for the testnet).

Signing, the server-time offset, pacing and retries live here; order logic does not. Rules:

- Every signed request carries `timestamp` (local clock plus the server-time offset) and
  `recvWindow`, and a `signature`: the hex HMAC-SHA256 of the exact query string that is sent.
- Read-only requests are retried on 5xx and network errors. An order-placing or cancelling
  request is never retried: a timeout or 5xx raises `OrderStatusUnknown` and the caller queries
  the order by its client order ID.
- HTTP 418 bans the client for good (`Banned` on every later call). HTTP 429 pauses all requests
  for `Retry-After`.
- Keys, signatures and query strings never reach a log line, an exception or a repr.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import TracebackType
from typing import Any, Literal, Self
from urllib.parse import urlencode

import httpx
from pydantic import SecretStr

from crypto_moonshot.config import Settings
from crypto_moonshot.data.binance import DEFAULT_WEIGHT_BUDGET, Clock, Sleep
from crypto_moonshot.exchange.errors import (
    Banned,
    ExchangeError,
    InvalidTimestamp,
    OrderStatusUnknown,
    RateLimited,
    error_from_response,
)
from crypto_moonshot.exchange.models import (
    Account,
    ExchangeOrder,
    OcoLeg,
    OrderList,
    Trade,
    from_ms,
    to_ms,
)
from crypto_moonshot.exchange.pacing import (
    DEFAULT_ORDER_LIMIT_1D,
    DEFAULT_ORDER_LIMIT_10S,
    Pacer,
)
from crypto_moonshot.exchange.rules import ExchangeInfo, SymbolRules, format_decimal

TESTNET_REST_URL = "https://testnet.binance.vision"
DEFAULT_RECV_WINDOW_MS = 5000
API_KEY_HEADER = "X-MBX-APIKEY"
_MAX_BACKOFF_S = 60.0

Side = Literal["BUY", "SELL"]
Param = str | int | bool | Decimal | None

logger = logging.getLogger(__name__)


class _SignatureFilter(logging.Filter):
    """Drops httpx's request log lines for signed URLs: they would print the signature."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "signature=" not in record.getMessage()


# httpx logs "HTTP Request: POST <full url>" at INFO; the full URL of a signed request holds the
# signature and every parameter. Install the filter once, whenever this module is imported.
logging.getLogger("httpx").addFilter(_SignatureFilter())


def sign(secret: str, query: str) -> str:
    """Hex HMAC-SHA256 of `query` with `secret`, as Binance expects in `signature`."""
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _fmt(value: Param) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        return format_decimal(value)
    return str(value)


class SignedClient:
    """Async Binance spot REST client with signed endpoints.

    Args:
        api_key: API key, sent only in the `X-MBX-APIKEY` header.
        api_secret: Secret, used only to sign.
        base_url: REST host (default the spot testnet).
        http: The `httpx.AsyncClient` to use; tests pass one with a `MockTransport`. When
            omitted the client creates (and closes) its own.
        recv_window: `recvWindow` in ms sent with every signed request.
        now, sleep: Injectable clock (aware UTC) and sleep, so tests do not wait.
        weight_budget: Used weight per minute to stay under.
        order_limit_10s, order_limit_1d: Order-count limits never to exceed.
        max_attempts: Attempts for a read-only request on 5xx, network errors and 429.
        backoff_base: Seconds before the first retry; doubles each time.
    """

    def __init__(
        self,
        api_key: SecretStr,
        api_secret: SecretStr,
        *,
        base_url: str = TESTNET_REST_URL,
        http: httpx.AsyncClient | None = None,
        recv_window: int = DEFAULT_RECV_WINDOW_MS,
        now: Clock = _utc_now,
        sleep: Sleep = asyncio.sleep,
        weight_budget: int = DEFAULT_WEIGHT_BUDGET,
        order_limit_10s: int = DEFAULT_ORDER_LIMIT_10S,
        order_limit_1d: int = DEFAULT_ORDER_LIMIT_1D,
        max_attempts: int = 5,
        backoff_base: float = 1.0,
    ) -> None:
        if not 1 <= recv_window <= 60_000:
            msg = "recv_window must be between 1 and 60000 ms"
            raise ValueError(msg)
        if max_attempts < 1:
            msg = "max_attempts must be at least 1"
            raise ValueError(msg)
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url.rstrip("/")
        self._owns_http = http is None
        self._http = http if http is not None else httpx.AsyncClient(timeout=30.0)
        self._recv_window = recv_window
        self._now = now
        self._sleep = sleep
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._pacer = Pacer(
            now,
            sleep,
            weight_budget=weight_budget,
            order_limit_10s=order_limit_10s,
            order_limit_1d=order_limit_1d,
        )
        self._offset = timedelta(0)
        self._synced = False
        self._banned: Banned | None = None
        self.requests = 0
        """HTTP requests sent, including retries."""

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> Self:
        """Build a client for the configured testnet URL and keys (which must be set)."""
        key, secret = settings.binance_testnet_api_key, settings.binance_testnet_api_secret
        if key is None or secret is None:
            msg = "MOONSHOT_BINANCE_TESTNET_API_KEY and _SECRET must both be set"
            raise ValueError(msg)
        return cls(key, secret, base_url=settings.binance_testnet_rest_url, **kwargs)

    def __repr__(self) -> str:
        return f"SignedClient(base_url={self._base_url!r})"

    __str__ = __repr__

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

    @property
    def max_used_weight(self) -> int:
        return self._pacer.max_used_weight

    @property
    def time_offset(self) -> timedelta:
        """Server time minus local time, as of the last sync."""
        return self._offset

    @property
    def banned(self) -> bool:
        return self._banned is not None

    # -- market and account endpoints ----------------------------------------------------------

    async def server_time(self) -> datetime:
        """`GET /api/v3/time` (weight 1)."""
        body = await self._request("GET", "/api/v3/time", weight=1)
        return from_ms(int(body["serverTime"]))

    async def sync_time(self) -> timedelta:
        """Measure the server-time offset (assuming the reply was sent mid-flight) and keep it."""
        before = self._now()
        server = await self.server_time()
        after = self._now()
        self._offset = server - (before + (after - before) / 2)
        self._synced = True
        return self._offset

    async def exchange_info(self) -> ExchangeInfo:
        """`GET /api/v3/exchangeInfo` (weight 20): every symbol's rules."""
        body = await self._request("GET", "/api/v3/exchangeInfo", weight=20)
        symbols = [SymbolRules.from_json(s) for s in body.get("symbols", ())]
        return ExchangeInfo(from_ms(int(body["serverTime"])), {s.symbol: s for s in symbols})

    async def ticker_price(self, symbol: str) -> Decimal:
        """`GET /api/v3/ticker/price` (weight 2): the symbol's last traded price."""
        body = await self._request("GET", "/api/v3/ticker/price", {"symbol": symbol}, weight=2)
        return Decimal(str(body["price"]))

    async def account(self) -> Account:
        """`GET /api/v3/account` (weight 20)."""
        return Account.from_json(await self._signed("GET", "/api/v3/account", {}, weight=20))

    # -- orders --------------------------------------------------------------------------------

    async def new_order(
        self,
        symbol: str,
        side: Side,
        order_type: Literal["MARKET", "LIMIT_MAKER", "STOP_LOSS"],
        *,
        client_order_id: str,
        quantity: Decimal | None = None,
        quote_order_qty: Decimal | None = None,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        trailing_delta: int | None = None,
    ) -> ExchangeOrder:
        """`POST /api/v3/order`. Never retried: a timeout or 5xx raises `OrderStatusUnknown`.

        - `MARKET`: exactly one of `quantity` (base) and `quote_order_qty` (quote, buys); the
          response is `FULL`, so `fills` hold prices and commissions.
        - `LIMIT_MAKER`: `quantity` and `price`.
        - `STOP_LOSS`: `quantity` and `stop_price` and/or `trailing_delta` (basis points).
        """
        if order_type == "MARKET":
            if (quantity is None) == (quote_order_qty is None):
                msg = "a MARKET order needs exactly one of quantity and quote_order_qty"
                raise ValueError(msg)
        elif quantity is None:
            msg = f"a {order_type} order needs a quantity"
            raise ValueError(msg)
        if order_type == "LIMIT_MAKER" and price is None:
            msg = "a LIMIT_MAKER order needs a price"
            raise ValueError(msg)
        if order_type == "STOP_LOSS" and stop_price is None and trailing_delta is None:
            msg = "a STOP_LOSS order needs a stop_price and/or a trailing_delta"
            raise ValueError(msg)
        params: dict[str, Param] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "quoteOrderQty": quote_order_qty,
            "price": price,
            "stopPrice": stop_price,
            "trailingDelta": trailing_delta,
            "newClientOrderId": client_order_id,
            "newOrderRespType": "FULL" if order_type == "MARKET" else "RESULT",
        }
        body = await self._signed("POST", "/api/v3/order", params, weight=1, orders=1)
        return ExchangeOrder.from_json(body)

    async def new_oco(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal,
        above: OcoLeg,
        below: OcoLeg,
        *,
        list_client_order_id: str,
    ) -> OrderList:
        """`POST /api/v3/orderList/oco` (counts as two orders). Never retried."""
        params: dict[str, Param] = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "listClientOrderId": list_client_order_id,
            "aboveType": above.order_type,
            "aboveClientOrderId": above.client_order_id,
            "abovePrice": above.price,
            "aboveStopPrice": above.stop_price,
            "aboveTrailingDelta": above.trailing_delta,
            "belowType": below.order_type,
            "belowClientOrderId": below.client_order_id,
            "belowPrice": below.price,
            "belowStopPrice": below.stop_price,
            "belowTrailingDelta": below.trailing_delta,
            "newOrderRespType": "RESULT",
        }
        body = await self._signed("POST", "/api/v3/orderList/oco", params, weight=1, orders=2)
        return OrderList.from_json(body)

    async def cancel_order(self, symbol: str, client_order_id: str) -> ExchangeOrder:
        """`DELETE /api/v3/order` by client order ID. Cancelling an OCO leg cancels the list."""
        params: dict[str, Param] = {"symbol": symbol, "origClientOrderId": client_order_id}
        return ExchangeOrder.from_json(
            await self._signed("DELETE", "/api/v3/order", params, weight=1)
        )

    async def cancel_order_list(self, symbol: str, list_client_order_id: str) -> OrderList:
        """`DELETE /api/v3/orderList` by list client order ID."""
        params: dict[str, Param] = {"symbol": symbol, "listClientOrderId": list_client_order_id}
        return OrderList.from_json(
            await self._signed("DELETE", "/api/v3/orderList", params, weight=1)
        )

    async def get_order(self, symbol: str, client_order_id: str) -> ExchangeOrder:
        """`GET /api/v3/order` by client order ID (weight 4); `UnknownOrder` if there is none."""
        params: dict[str, Param] = {"symbol": symbol, "origClientOrderId": client_order_id}
        return ExchangeOrder.from_json(await self._signed("GET", "/api/v3/order", params, weight=4))

    async def open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        """`GET /api/v3/openOrders` (weight 6 for a symbol, 80 for all)."""
        params: dict[str, Param] = {"symbol": symbol}
        body = await self._signed("GET", "/api/v3/openOrders", params, weight=6 if symbol else 80)
        return [ExchangeOrder.from_json(o) for o in body]

    async def open_order_lists(self) -> list[OrderList]:
        """`GET /api/v3/openOrderList` (weight 6)."""
        body = await self._signed("GET", "/api/v3/openOrderList", {}, weight=6)
        return [OrderList.from_json(o) for o in body]

    async def get_order_list(self, list_client_order_id: str) -> OrderList:
        """`GET /api/v3/orderList` by list client order ID (weight 4)."""
        params: dict[str, Param] = {"origClientOrderId": list_client_order_id}
        return OrderList.from_json(await self._signed("GET", "/api/v3/orderList", params, weight=4))

    async def my_trades(self, symbol: str, order_id: int) -> list[Trade]:
        """`GET /api/v3/myTrades` for one order (weight 5)."""
        params: dict[str, Param] = {"symbol": symbol, "orderId": order_id}
        body = await self._signed("GET", "/api/v3/myTrades", params, weight=5)
        return [Trade.from_json(t) for t in body]

    # -- internals -----------------------------------------------------------------------------

    async def _signed(
        self, method: str, path: str, params: Mapping[str, Param], *, weight: int, orders: int = 0
    ) -> Any:
        return await self._request(method, path, params, weight=weight, orders=orders, signed=True)

    def _query(self, params: Mapping[str, Param], signed: bool) -> str:
        items = [(k, _fmt(v)) for k, v in params.items() if v is not None]
        if not signed:
            return urlencode(items)
        timestamp = to_ms(self._now() + self._offset)
        items += [("recvWindow", str(self._recv_window)), ("timestamp", str(timestamp))]
        query = urlencode(items)
        signature = sign(self._api_secret.get_secret_value(), query)
        return f"{query}&signature={signature}"

    async def _request(
        self,
        method: str,
        path: str,
        params: Mapping[str, Param] | None = None,
        *,
        weight: int,
        orders: int = 0,
        signed: bool = False,
    ) -> Any:
        read_only = method == "GET"
        headers = {API_KEY_HEADER: self._api_key.get_secret_value()} if signed else {}
        resynced = False
        attempt = 0
        while True:
            if self._banned is not None:
                raise self._banned
            if signed and not self._synced:
                await self.sync_time()
            await self._pacer.reserve(weight, orders)
            url = httpx.URL(self._base_url + path, query=self._query(params or {}, signed).encode())
            response: httpx.Response | None = None
            try:
                self.requests += 1
                response = await self._http.request(method, url, headers=headers)
            except httpx.TransportError as exc:
                failure = type(exc).__name__  # never the exception text: it may name the URL
            finally:
                self._pacer.release(weight, response.headers if response else None)

            if response is None:
                if not read_only:
                    msg = f"{method} {path}: no response ({failure}); outcome unknown"
                    raise OrderStatusUnknown(msg) from None
                error: ExchangeError = ExchangeError(f"{method} {path}: network error ({failure})")
            else:
                logger.debug("%s %s -> %d", method, path, response.status_code)
                if response.status_code == 200:
                    return self._parse(response)
                error = error_from_response(
                    response.status_code,
                    response.text,
                    retry_after=response.headers.get("Retry-After"),
                    order_endpoint=not read_only,
                )

            if isinstance(error, Banned):
                self._banned = error
                raise error
            if isinstance(error, InvalidTimestamp) and not resynced:
                resynced = True
                await self.sync_time()
                continue
            attempt += 1
            retryable = read_only and attempt < self._max_attempts
            if isinstance(error, RateLimited):
                self._pacer.pause(error.retry_after)
                if retryable:
                    continue
            elif response is None or response.status_code >= 500:
                if retryable:
                    await self._sleep(min(self._backoff_base * 2 ** (attempt - 1), _MAX_BACKOFF_S))
                    continue
            raise error

    @staticmethod
    def _parse(response: httpx.Response) -> Any:
        try:
            return response.json()
        except json.JSONDecodeError:
            msg = f"unreadable response (HTTP {response.status_code})"
            raise ExchangeError(msg, status=response.status_code) from None
