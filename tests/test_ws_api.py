"""The WebSocket API user data stream client against the fake exchange's local server."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from binance_fakes import Clock, fast_sleep, until
from fake_exchange import API_KEY, API_SECRET, FakeExchange
from pydantic import SecretStr

from crypto_moonshot.exchange.client import sign
from crypto_moonshot.exchange.ws_api import (
    SUBSCRIBE_METHOD,
    AccountPosition,
    ExecutionReport,
    ListStatus,
    OtherEvent,
    StreamDown,
    StreamMessage,
    StreamUp,
    UserDataStream,
    WsApiError,
    parse_event,
    signed_params,
)

pytestmark = pytest.mark.anyio

D = Decimal


class Harness:
    def __init__(self, fake: FakeExchange, url: str, secret: str = API_SECRET) -> None:
        self.fake = fake
        self.messages: list[StreamMessage] = []
        self.stream = UserDataStream(
            SecretStr(API_KEY),
            SecretStr(secret),
            self.messages.append,
            url=url,
            now=fake.clock,
            sleep=fast_sleep,
            backoff_base=0.005,
            backoff_max=0.02,
        )

    def of(self, kind: type) -> list[StreamMessage]:
        return [m for m in self.messages if isinstance(m, kind)]


@pytest.fixture
async def fake() -> AsyncIterator[FakeExchange]:
    fake = FakeExchange(Clock())
    yield fake
    await fake.stop_ws()


async def started(fake: FakeExchange, secret: str = API_SECRET) -> Harness:
    url = await fake.start_ws()
    assert url.startswith("ws://127.0.0.1:")
    harness = Harness(fake, url, secret)
    harness.stream.start()
    return harness


def test_signed_params_sign_the_sorted_parameters() -> None:
    params = signed_params(SecretStr("key"), SecretStr("secret"), 1_700_000_000_000, 5000)
    assert list(params) == ["apiKey", "recvWindow", "timestamp", "signature"]
    expected = sign("secret", "apiKey=key&recvWindow=5000&timestamp=1700000000000")
    assert params["signature"] == expected


async def test_it_subscribes_and_passes_parsed_events(fake: FakeExchange) -> None:
    h = await started(fake)
    try:
        await until(lambda: h.stream.connected)
        assert h.messages == [StreamUp(reconnect=False)]
        (request,) = fake.ws_requests
        assert request["method"] == SUBSCRIBE_METHOD
        assert set(request["params"]) == {"apiKey", "recvWindow", "timestamp", "signature"}
        assert h.stream.subscription_id == 1
        fake.set_price("AAAUSDT", D("100"))
        fake.deposit("AAA", D(10))
        fake._new_order(
            {
                "symbol": "AAAUSDT",
                "side": "SELL",
                "type": "MARKET",
                "quantity": "1",
                "newClientOrderId": "cms-x",
            }
        )
        await until(lambda: len(h.of(AccountPosition)) == 1)
        new, trade = h.of(ExecutionReport)
        assert isinstance(new, ExecutionReport)
        assert isinstance(trade, ExecutionReport)
        assert (new.execution_type, new.status, new.order_client_id) == ("NEW", "NEW", "cms-x")
        assert (trade.execution_type, trade.status) == ("TRADE", "FILLED")
        assert trade.last_quantity == D(1)
        assert trade.cumulative_quote == D(100)
        assert trade.commission == D("0.1")
        assert trade.commission_asset == "USDT"
        assert trade.trade_id == 1
        (account,) = h.of(AccountPosition)
        assert isinstance(account, AccountPosition)
        assert account.balances["AAA"] == (D(9), D(0))
    finally:
        await h.stream.aclose()


async def test_requests_are_answered_by_id(fake: FakeExchange) -> None:
    h = await started(fake)
    try:
        await until(lambda: h.stream.connected)
        assert await h.stream.request("ping") == {}
        with pytest.raises(WsApiError, match="-1102") as caught:
            await h.stream.request("no.such.method", {"x": 1})
        assert caught.value.code == -1102
    finally:
        await h.stream.aclose()
    with pytest.raises(ConnectionError):
        await h.stream.request("ping")


async def test_a_dropped_connection_reconnects_and_says_so(fake: FakeExchange) -> None:
    h = await started(fake)
    try:
        await until(lambda: h.stream.connected)
        await fake.drop_connections()
        await until(lambda: len(h.of(StreamUp)) == 2)
        assert h.messages[1:] == [StreamDown("connection closed"), StreamUp(reconnect=True)]
        assert h.stream.connects == 2
    finally:
        await h.stream.aclose()


async def test_event_stream_terminated_resubscribes_on_the_same_connection(
    fake: FakeExchange,
) -> None:
    h = await started(fake)
    try:
        await until(lambda: h.stream.connected)
        fake.terminate_stream()
        await until(lambda: len(h.of(StreamUp)) == 2)
        assert h.messages[1:] == [StreamDown("event stream terminated"), StreamUp(True)]
        assert h.stream.connects == 1
        assert h.stream.subscription_id == 2
    finally:
        await h.stream.aclose()


async def test_server_shutdown_reconnects(fake: FakeExchange) -> None:
    h = await started(fake)
    try:
        await until(lambda: h.stream.connected)
        fake.server_shutdown()
        await until(lambda: len(h.of(StreamUp)) == 2)
        assert h.messages[1] == StreamDown("server shutdown")
        assert h.stream.connects == 2
    finally:
        await h.stream.aclose()


async def test_a_bad_signature_keeps_retrying_without_leaking_the_secret(
    fake: FakeExchange, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    h = await started(fake, secret="wrong-secret-not-real")
    try:
        await until(lambda: h.stream.connects >= 3)
        assert not h.stream.connected
        assert h.messages == []
        assert h.stream.last_error is not None
        assert "-1022" in h.stream.last_error
        signatures = [m["params"]["signature"] for m in fake.ws_requests]
        for secret in ("wrong-secret-not-real", API_KEY, *signatures):
            assert secret not in caplog.text
            assert secret not in repr(h.stream)
            assert secret not in h.stream.last_error
    finally:
        await h.stream.aclose()


def test_parse_event_types() -> None:
    listed = parse_event(
        {
            "e": "listStatus",
            "s": "AAAUSDT",
            "g": 7,
            "c": "OCO",
            "l": "ALL_DONE",
            "L": "ALL_DONE",
            "C": "cmx-list",
            "T": 1_700_000_000_000,
            "O": [{"s": "AAAUSDT", "i": 1, "c": "cmx-a"}, {"s": "AAAUSDT", "i": 2, "c": "cmx-b"}],
        }
    )
    assert isinstance(listed, ListStatus)
    assert listed.list_order_status == "ALL_DONE"
    assert listed.orders == (("AAAUSDT", 1, "cmx-a"), ("AAAUSDT", 2, "cmx-b"))
    cancel = parse_event(
        {
            "e": "executionReport",
            "s": "AAAUSDT",
            "c": "cancel-1",
            "C": "cmx-a",
            "x": "CANCELED",
            "X": "CANCELED",
            "i": 1,
            "g": 7,
            "T": 1,
        }
    )
    assert isinstance(cancel, ExecutionReport)
    assert cancel.order_client_id == "cmx-a"
    assert cancel.order_list_id == 7
    other = parse_event({"e": "balanceUpdate", "a": "USDT", "d": "5"})
    assert other == OtherEvent("balanceUpdate", {"e": "balanceUpdate", "a": "USDT", "d": "5"})
