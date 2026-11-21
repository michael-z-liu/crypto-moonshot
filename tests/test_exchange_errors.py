import json

import pytest

from crypto_moonshot.exchange.errors import (
    AuthError,
    Banned,
    DuplicateOrder,
    ExchangeError,
    FilterRejected,
    InsufficientBalance,
    InvalidTimestamp,
    OrderNotAccepted,
    OrderStatusUnknown,
    RateLimited,
    ServerError,
    UnknownOrder,
    WouldTake,
    WouldTrigger,
    error_from_response,
    parse_error_body,
)


def body(code: int, msg: str) -> str:
    return json.dumps({"code": code, "msg": msg})


@pytest.mark.parametrize(
    ("status", "code", "msg", "expected"),
    [
        (400, -1021, "Timestamp for this request is outside of the recvWindow.", InvalidTimestamp),
        (400, -1022, "Signature for this request is not valid.", AuthError),
        (401, -2015, "Invalid API-key, IP, or permissions for action.", AuthError),
        (401, -2014, "API-key format invalid.", AuthError),
        (400, -1013, "Filter failure: LOT_SIZE", FilterRejected),
        (400, -2010, "Account has insufficient balance for requested action.", InsufficientBalance),
        (400, -2010, "Duplicate order sent.", DuplicateOrder),
        (400, -2010, "Order would immediately match and take.", WouldTake),
        (400, -2010, "Stop price would trigger immediately.", OrderNotAccepted),
        (400, -2021, "Order would trigger immediately.", WouldTrigger),
        (400, -2011, "Unknown order sent.", UnknownOrder),
        (400, -2013, "Order does not exist.", UnknownOrder),
        (503, -1007, "Timeout waiting for response from backend server.", OrderStatusUnknown),
        (
            409,
            -1006,
            "An unexpected response was received from the message bus.",
            OrderStatusUnknown,
        ),
        (429, -1003, "Too many requests queued.", RateLimited),
        (429, -1015, "Too many new orders; current limit is 50 orders per 10 SECOND.", RateLimited),
        (400, -1100, "Illegal characters found in parameter 'symbol'.", ExchangeError),
    ],
)
def test_binance_codes_map_to_typed_errors(
    status: int, code: int, msg: str, expected: type[ExchangeError]
) -> None:
    error = error_from_response(status, body(code, msg))
    assert type(error) is expected
    assert (error.status, error.code, error.msg) == (status, code, msg)
    assert str(code) in str(error)
    assert msg in str(error)


def test_filter_rejection_names_the_filter() -> None:
    error = error_from_response(400, body(-1013, "Filter failure: NOTIONAL"))
    assert isinstance(error, FilterRejected)
    assert error.filter_name == "NOTIONAL"
    other = error_from_response(400, body(-1013, "Market is closed."))
    assert isinstance(other, FilterRejected)
    assert other.filter_name is None


def test_rate_limit_and_ban_carry_retry_after() -> None:
    limited = error_from_response(429, body(-1003, "Too many requests"), retry_after="12")
    assert isinstance(limited, RateLimited)
    assert limited.retry_after == 12.0
    assert error_from_response(429, "", retry_after=None).retry_after == 60.0  # type: ignore[attr-defined]
    banned = error_from_response(418, "", retry_after="300")
    assert isinstance(banned, Banned)
    assert banned.retry_after == 300.0
    assert error_from_response(418, "").retry_after is None  # type: ignore[attr-defined]


def test_server_errors_depend_on_the_endpoint() -> None:
    read = error_from_response(503, "Service Unavailable")
    assert type(read) is ServerError
    order = error_from_response(503, "Service Unavailable", order_endpoint=True)
    assert type(order) is OrderStatusUnknown
    assert order.status == 503


def test_unparseable_bodies() -> None:
    assert parse_error_body("<html>") == (None, None)
    assert parse_error_body("[1]") == (None, None)
    assert parse_error_body('{"code": "x", "msg": 3}') == (None, None)
    error = error_from_response(400, "<html>bad</html>")
    assert type(error) is ExchangeError
    assert "bad" in str(error)


def test_missing_order_list_maps_to_unknown_order() -> None:
    error = error_from_response(400, body(-2018, "Order list does not exist."))
    assert isinstance(error, UnknownOrder)
