"""Typed exceptions for Binance REST failures, mapped from HTTP statuses, error codes and
messages (see `docs/binance-spot-api.md`, "Errors the broker handles").

No exception here ever carries a key, a signature or a query string: only Binance's own code and
message text.
"""

from __future__ import annotations

import json
import re
from typing import Any

DEFAULT_RETRY_AFTER_S = 60.0


class ExchangeError(Exception):
    """A Binance request failed. `code` and `msg` are Binance's, when it sent them."""

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


class InvalidTimestamp(ExchangeError):
    """-1021: the request timestamp is outside the recvWindow (the clock offset is off)."""


class AuthError(ExchangeError):
    """-1022 (bad signature) or -2014 / -2015 (bad key, IP or permissions). Fatal: stop trading."""


class FilterRejected(ExchangeError):
    """-1013: rejected before the matching engine, usually by a symbol filter."""

    def __init__(
        self,
        message: str,
        *,
        filter_name: str | None = None,
        status: int | None = None,
        code: int | None = None,
        msg: str | None = None,
    ) -> None:
        super().__init__(message, status=status, code=code, msg=msg)
        self.filter_name = filter_name
        """The failed filter (`LOT_SIZE`, `NOTIONAL`, ...), or None if the message names none."""


class OrderNotAccepted(ExchangeError):
    """-2010 with a reason the broker does not special-case."""


class InsufficientBalance(OrderNotAccepted):
    """-2010: the account cannot cover the order."""


class DuplicateOrder(OrderNotAccepted):
    """-2010: the client order ID is already used by an open order (query it)."""


class WouldTake(OrderNotAccepted):
    """-2010: a `LIMIT_MAKER` would have matched at once and taken liquidity."""


class WouldTrigger(OrderNotAccepted):
    """-2021: a stop order would trigger immediately (the stop price is already crossed)."""


class UnknownOrder(ExchangeError):
    """-2011 / -2013 (and a missing order list): no such order. It filled, was cancelled, or was
    never placed: query before concluding."""


class OrderStatusUnknown(ExchangeError):
    """The outcome of an order request is unknown (-1007, -1006, HTTP 5xx or a timeout on an
    order endpoint). Never retry blind: query the order by its client order ID."""


class ServerError(ExchangeError):
    """HTTP 5xx on a read-only endpoint, after the client gave up retrying."""


class RateLimited(ExchangeError):
    """HTTP 429 or -1003 / -1015: back off for `retry_after` seconds."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float = DEFAULT_RETRY_AFTER_S,
        status: int | None = None,
        code: int | None = None,
        msg: str | None = None,
    ) -> None:
        super().__init__(message, status=status, code=code, msg=msg)
        self.retry_after = retry_after


class Banned(ExchangeError):
    """HTTP 418: the IP is banned. The client refuses every later request."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        status: int | None = 418,
        code: int | None = None,
        msg: str | None = None,
    ) -> None:
        super().__init__(message, status=status, code=code, msg=msg)
        self.retry_after = retry_after


_FILTER = re.compile(r"Filter failure:\s*(\w+)", re.IGNORECASE)


def parse_error_body(text: str) -> tuple[int | None, str | None]:
    """Binance's `{"code": -1021, "msg": "..."}` from a response body, or (None, None)."""
    try:
        body: Any = json.loads(text)
    except ValueError:
        return None, None
    if not isinstance(body, dict):
        return None, None
    raw_code = body.get("code")
    code = raw_code if isinstance(raw_code, int) and not isinstance(raw_code, bool) else None
    msg = body.get("msg")
    return code, msg if isinstance(msg, str) else None


def _retry_after(value: str | None, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def error_from_response(
    status: int,
    text: str,
    *,
    retry_after: str | None = None,
    order_endpoint: bool = False,
) -> ExchangeError:
    """The exception for a failed response: HTTP `status`, body `text`, `Retry-After` header.

    `order_endpoint` marks requests that place or cancel orders: there, a timeout or a 5xx means
    the outcome is unknown (`OrderStatusUnknown`) rather than a plain server error.
    """
    code, api_msg = parse_error_body(text)
    shown = api_msg if api_msg is not None else text[:200]
    message = f"HTTP {status}" + (f", code {code}" if code is not None else "") + f": {shown}"
    kw: dict[str, Any] = {"status": status, "code": code, "msg": api_msg}

    if status == 418:
        return Banned(message, retry_after=_retry_after(retry_after, None), **kw)
    if status == 429 or code in (-1003, -1015):
        return RateLimited(
            message, retry_after=_retry_after(retry_after, DEFAULT_RETRY_AFTER_S) or 0.0, **kw
        )
    if code == -1021:
        return InvalidTimestamp(message, **kw)
    if code in (-1022, -2014, -2015):
        return AuthError(message, **kw)
    if code == -1013:
        found = _FILTER.search(api_msg or "")
        return FilterRejected(message, filter_name=found.group(1) if found else None, **kw)
    if code == -2010:
        lowered = (api_msg or "").lower()
        if "insufficient balance" in lowered:
            return InsufficientBalance(message, **kw)
        if "duplicate" in lowered:
            return DuplicateOrder(message, **kw)
        if "immediately match" in lowered:
            return WouldTake(message, **kw)
        return OrderNotAccepted(message, **kw)
    if code == -2021:
        return WouldTrigger(message, **kw)
    if code in (-2011, -2013) or "list does not exist" in (api_msg or "").lower():
        return UnknownOrder(message, **kw)
    if code in (-1006, -1007) or (status >= 500 and order_endpoint):
        return OrderStatusUnknown(message, **kw)
    if status >= 500:
        return ServerError(message, **kw)
    return ExchangeError(message, **kw)
