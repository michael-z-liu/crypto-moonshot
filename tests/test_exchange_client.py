import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qsl

import httpx
import pytest
from exchange_fakes import API_KEY, API_SECRET, T0, FakeClock, FakeTestnet, make_client
from pydantic import SecretStr

from crypto_moonshot.config import Settings
from crypto_moonshot.exchange.client import SignedClient, sign
from crypto_moonshot.exchange.errors import (
    AuthError,
    Banned,
    DuplicateOrder,
    ExchangeError,
    FilterRejected,
    InsufficientBalance,
    InvalidTimestamp,
    OrderStatusUnknown,
    RateLimited,
    ServerError,
    UnknownOrder,
    WouldTake,
    WouldTrigger,
)
from crypto_moonshot.exchange.models import OcoLeg

pytestmark = pytest.mark.anyio

D = Decimal


def split_query(request: httpx.Request) -> tuple[str, str, list[tuple[str, str]]]:
    """The query string as sent, the signature, and the parameters before the signature."""
    raw = request.url.query.decode()
    head, sep, signature = raw.rpartition("&signature=")
    assert sep, "signature must be the last parameter"
    return head, signature, parse_qsl(head)


def setup(**fake_kwargs: object) -> tuple[FakeTestnet, SignedClient]:
    fake = FakeTestnet(**fake_kwargs)  # type: ignore[arg-type]
    return fake, make_client(fake)


# -- signing -----------------------------------------------------------------------------------


def test_documented_signature_vector() -> None:
    """The HMAC example from Binance's docs (a documented key, not a real one)."""
    secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
    query = (
        "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1"
        "&recvWindow=5000&timestamp=1499827319559"
    )
    expected = "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"
    assert sign(secret, query) == expected


async def test_signed_request_signs_exactly_the_query_it_sends() -> None:
    fake, client = setup()
    await client.new_order(
        "BTCUSDT", "BUY", "MARKET", client_order_id="cid-1", quote_order_qty=D("100")
    )
    request = fake.requests_to("/api/v3/order")[0]
    head, signature, params = split_query(request)
    assert signature == sign(API_SECRET, head)
    assert request.headers["X-MBX-APIKEY"] == API_KEY
    assert request.method == "POST"
    # Parameter order is the order we built: our params, then recvWindow, timestamp.
    assert [k for k, _ in params] == [
        "symbol",
        "side",
        "type",
        "quoteOrderQty",
        "newClientOrderId",
        "newOrderRespType",
        "recvWindow",
        "timestamp",
    ]
    assert dict(params)["recvWindow"] == "5000"
    assert dict(params)["quoteOrderQty"] == "100"
    assert dict(params)["newOrderRespType"] == "FULL"


async def test_public_requests_are_unsigned_and_send_no_key() -> None:
    fake, client = setup()
    await client.server_time()
    await client.exchange_info()
    for request in fake.requests:
        assert "signature" not in request.url.query.decode()
        assert "X-MBX-APIKEY" not in request.headers


async def test_decimals_are_sent_plainly() -> None:
    fake, client = setup()
    fake.set_price("XRPUSDT", D("0.5"))
    fake.balances["XRP"] = [D("1000"), D(0)]
    await client.new_order(
        "XRPUSDT",
        "SELL",
        "LIMIT_MAKER",
        client_order_id="c",
        quantity=D("1E+2"),
        price=D("0.60000100"),
    )
    params = dict(split_query(fake.requests_to("/api/v3/order")[0])[2])
    assert params["quantity"] == "100"
    assert params["price"] == "0.600001"


async def test_recv_window_is_configurable() -> None:
    fake = FakeTestnet()
    client = make_client(fake, recv_window=1234)
    await client.account()
    assert dict(split_query(fake.requests_to("/api/v3/account")[0])[2])["recvWindow"] == "1234"
    with pytest.raises(ValueError, match="recv_window"):
        make_client(fake, recv_window=60001)


async def test_bad_secret_and_bad_key() -> None:
    fake = FakeTestnet()
    wrong_secret = make_client(fake, api_secret=SecretStr("another-fake-secret"))
    with pytest.raises(AuthError) as info:
        await wrong_secret.account()
    assert info.value.code == -1022
    wrong_key = make_client(fake, api_key=SecretStr("another-fake-key"))
    with pytest.raises(AuthError) as info2:
        await wrong_key.account()
    assert info2.value.code == -2015


# -- time sync ---------------------------------------------------------------------------------


async def test_server_time_offset_is_applied() -> None:
    # The server runs 10 s behind: unadjusted timestamps would be "in the future" (-1021).
    fake = FakeTestnet(skew=timedelta(seconds=-10))
    client = make_client(fake)
    await client.account()
    assert client.time_offset == timedelta(seconds=-10)
    assert len(fake.requests_to("/api/v3/time")) == 1
    timestamp = int(dict(split_query(fake.requests_to("/api/v3/account")[0])[2])["timestamp"])
    assert timestamp == int((fake.clock.now() - timedelta(seconds=10)).timestamp() * 1000)
    await client.account()
    assert len(fake.requests_to("/api/v3/time")) == 1  # synced once, at the first signed call


async def test_explicit_sync_returns_the_offset() -> None:
    fake = FakeTestnet(skew=timedelta(seconds=3))
    client = make_client(fake)
    assert await client.sync_time() == timedelta(seconds=3)
    assert await client.server_time() == fake.clock.now() + timedelta(seconds=3)


async def test_timestamp_error_resyncs_and_retries_once() -> None:
    fake, client = setup()
    await client.account()
    fake.skew = timedelta(seconds=-30)  # the clocks drift apart
    account = await client.account()
    assert account.balance("USDT").free == D("10000")
    assert len(fake.requests_to("/api/v3/time")) == 2
    assert len(fake.requests_to("/api/v3/account")) == 3  # ok, -1021, ok
    assert client.time_offset == timedelta(seconds=-30)


async def test_second_timestamp_error_raises() -> None:
    fake, client = setup()
    await client.account()
    fake.fail(
        "/api/v3/account", status=400, code=-1021, msg="Timestamp outside recvWindow.", times=2
    )
    with pytest.raises(InvalidTimestamp):
        await client.account()
    assert len(fake.requests_to("/api/v3/account")) == 3
    assert len(fake.requests_to("/api/v3/time")) == 2


async def test_orders_are_retried_after_a_timestamp_error_only() -> None:
    fake, client = setup()
    await client.account()
    fake.skew = timedelta(seconds=-30)
    order = await client.new_order(
        "BTCUSDT", "BUY", "MARKET", client_order_id="cid-ts", quote_order_qty=D("50")
    )
    assert order.status == "FILLED"
    assert len(fake.orders) == 1  # placed once: -1021 rejects before processing


# -- pacing and retries ------------------------------------------------------------------------


async def test_weight_budget_is_respected() -> None:
    fake = FakeTestnet(weight_limit=100)
    client = make_client(fake, weight_budget=100)
    for _ in range(30):  # 30 x weight 20 = 600 in a 100/min budget
        await client.account()
    assert fake.rate_limited == 0
    assert fake.max_weight_1m <= 100
    assert fake.clock.sleeps, "the client had to wait for the next minute"
    assert client.max_used_weight <= 100


async def test_weight_waits_for_the_next_minute_boundary() -> None:
    fake = FakeTestnet(FakeClock(datetime(2026, 1, 1, 12, 0, 30, tzinfo=UTC)))
    client = make_client(fake, weight_budget=45)
    await client.account()  # time sync 1 + 20
    await client.account()  # 41
    await client.account()  # would be 61 > 45: waits until 12:01:00 + margin
    assert fake.clock.sleeps == [pytest.approx(30.5)]


async def test_order_count_limit_10s_is_respected() -> None:
    fake = FakeTestnet(order_limit_10s=5)
    client = make_client(fake, order_limit_10s=5)
    for i in range(12):
        await client.new_order(
            "BTCUSDT", "BUY", "MARKET", client_order_id=f"cid-{i}", quote_order_qty=D("10")
        )
    assert fake.rate_limited == 0
    assert fake.max_orders_10s <= 5
    assert fake.clock.sleeps
    assert len(fake.orders) == 12


async def test_oco_counts_as_two_orders() -> None:
    fake = FakeTestnet(order_limit_10s=3)
    client = make_client(fake, order_limit_10s=3)
    fake.balances["BTC"] = [D("10"), D(0)]
    for i in range(3):
        await client.new_oco(
            "BTCUSDT",
            "SELL",
            D("1"),
            OcoLeg("LIMIT_MAKER", f"a{i}", price=D("33000")),
            OcoLeg("STOP_LOSS", f"b{i}", stop_price=D("27000")),
            list_client_order_id=f"l{i}",
        )
    assert fake.rate_limited == 0
    assert fake.max_orders_10s <= 3


async def test_order_count_headers_are_honoured() -> None:
    """Orders the client did not place (same account, other process) still count."""
    fake = FakeTestnet(order_limit_10s=5)
    client = make_client(fake, order_limit_10s=5)
    fake._order_times.extend([fake.server_now] * 4)  # four orders placed by someone else
    await client.new_order(
        "BTCUSDT", "BUY", "MARKET", client_order_id="cid-a", quote_order_qty=D("10")
    )  # header now says 5
    await client.new_order(
        "BTCUSDT", "BUY", "MARKET", client_order_id="cid-b", quote_order_qty=D("10")
    )
    assert fake.rate_limited == 0
    assert fake.clock.sleeps
    assert fake.max_orders_10s <= 5


async def test_daily_order_limit_raises_instead_of_waiting() -> None:
    fake = FakeTestnet()
    client = make_client(fake, order_limit_1d=3)
    for i in range(3):
        await client.new_order(
            "BTCUSDT", "BUY", "MARKET", client_order_id=f"cid-{i}", quote_order_qty=D("10")
        )
    with pytest.raises(RateLimited) as info:
        await client.new_order(
            "BTCUSDT", "BUY", "MARKET", client_order_id="cid-x", quote_order_qty=D("10")
        )
    assert info.value.retry_after == pytest.approx(12 * 3600)  # to UTC midnight from 12:00
    assert len(fake.orders) == 3
    assert not fake.clock.sleeps


async def test_429_waits_for_retry_after_then_retries_reads() -> None:
    fake, client = setup()
    await client.sync_time()
    fake.fail(
        "/api/v3/account",
        status=429,
        code=-1003,
        msg="Too many requests",
        headers={"Retry-After": "7"},
    )
    account = await client.account()
    assert account.can_trade
    assert fake.clock.sleeps == [7.0]
    assert len(fake.requests_to("/api/v3/account")) == 2


async def test_429_pauses_every_later_request() -> None:
    fake, client = setup()
    await client.sync_time()
    fake.fail(
        "/api/v3/order",
        method="POST",
        status=429,
        code=-1015,
        msg="Too many new orders",
        headers={"Retry-After": "9"},
    )
    with pytest.raises(RateLimited) as info:
        await client.new_order(
            "BTCUSDT", "BUY", "MARKET", client_order_id="cid-1", quote_order_qty=D("10")
        )
    assert info.value.retry_after == 9.0
    assert len(fake.requests_to("/api/v3/order")) == 1  # not retried
    await client.account()
    assert fake.clock.sleeps == [9.0]  # the next request waited


async def test_persistent_429_gives_up_on_reads() -> None:
    fake, client = setup()
    fake.fail("/api/v3/account", status=429, code=-1003, msg="x", times=99)
    with pytest.raises(RateLimited):
        await client.account()
    assert len(fake.requests_to("/api/v3/account")) == 5


async def test_418_bans_every_later_call() -> None:
    fake, client = setup()
    await client.account()
    fake.ban()
    with pytest.raises(Banned):
        await client.account()
    seen = len(fake.requests)
    with pytest.raises(Banned):
        await client.account()
    with pytest.raises(Banned):
        await client.new_order(
            "BTCUSDT", "BUY", "MARKET", client_order_id="cid-1", quote_order_qty=D("10")
        )
    with pytest.raises(Banned):
        await client.server_time()
    assert len(fake.requests) == seen  # nothing was sent after the ban
    assert client.banned


async def test_reads_are_retried_on_5xx_with_backoff() -> None:
    fake, client = setup()
    await client.sync_time()
    fake.fail("/api/v3/account", status=503, body="Service Unavailable", times=2)
    await client.account()
    assert fake.clock.sleeps == [1.0, 2.0]
    assert len(fake.requests_to("/api/v3/account")) == 3


async def test_reads_are_retried_after_a_network_error() -> None:
    fake, client = setup()
    await client.sync_time()
    fake.timeout("/api/v3/openOrders")
    assert await client.open_orders("BTCUSDT") == []
    assert len(fake.requests_to("/api/v3/openOrders")) == 2


async def test_reads_give_up_after_max_attempts() -> None:
    fake, client = setup()
    await client.sync_time()
    fake.fail("/api/v3/account", status=502, body="bad gateway", times=99)
    with pytest.raises(ServerError):
        await client.account()
    assert len(fake.requests_to("/api/v3/account")) == 5
    fake.faults.clear()
    fake.timeout("/api/v3/account", times=99)
    with pytest.raises(ExchangeError, match="network error"):
        await client.account()


@pytest.mark.parametrize("kind", ["5xx", "-1007", "timeout"])
async def test_order_posts_are_never_retried(kind: str) -> None:
    fake, client = setup()
    await client.sync_time()
    if kind == "5xx":
        fake.fail("/api/v3/order", status=502, body="bad gateway")
    elif kind == "-1007":
        fake.fail("/api/v3/order", status=503, code=-1007, msg="Timeout waiting for response.")
    else:
        fake.timeout("/api/v3/order")
    with pytest.raises(OrderStatusUnknown):
        await client.new_order(
            "BTCUSDT", "BUY", "MARKET", client_order_id="cid-1", quote_order_qty=D("10")
        )
    assert len(fake.requests_to("/api/v3/order")) == 1
    assert fake.orders == {}
    assert not fake.clock.sleeps


async def test_oco_and_cancels_are_never_retried_either() -> None:
    fake, client = setup()
    await client.sync_time()
    fake.fail("/api/v3/orderList/oco", status=504, body="gateway timeout")
    with pytest.raises(OrderStatusUnknown):
        await client.new_oco(
            "BTCUSDT",
            "SELL",
            D("1"),
            OcoLeg("LIMIT_MAKER", "a", price=D("33000")),
            OcoLeg("STOP_LOSS", "b", stop_price=D("27000")),
            list_client_order_id="l",
        )
    fake.timeout("/api/v3/order", method="DELETE")
    with pytest.raises(OrderStatusUnknown):
        await client.cancel_order("BTCUSDT", "cid-1")
    assert len(fake.requests_to("/api/v3/orderList/oco")) == 1
    assert len(fake.requests_to("/api/v3/order")) == 1


# -- error mapping through the client ----------------------------------------------------------


async def test_order_errors_are_typed() -> None:
    fake, client = setup()
    fake.balances["BTC"] = [D("1"), D(0)]

    async def place(order_type: str, cid: str, **kw: Any) -> None:
        await client.new_order(
            "BTCUSDT",
            "BUY",
            order_type,  # type: ignore[arg-type]
            client_order_id=cid,
            **kw,
        )

    with pytest.raises(FilterRejected) as lot:
        await place("MARKET", "c1", quantity=D("0.000001"))
    assert lot.value.filter_name == "LOT_SIZE"
    with pytest.raises(FilterRejected) as notional:
        await place("MARKET", "c2", quote_order_qty=D("1"))
    assert notional.value.filter_name == "NOTIONAL"
    with pytest.raises(FilterRejected) as price:
        await place("LIMIT_MAKER", "c3", quantity=D("1"), price=D("29000.001"))
    assert price.value.filter_name == "PRICE_FILTER"
    with pytest.raises(InsufficientBalance):
        await place("MARKET", "c4", quantity=D("50"))
    with pytest.raises(WouldTake):
        await place("LIMIT_MAKER", "c5", quantity=D("0.1"), price=D("31000"))
    await place("LIMIT_MAKER", "c6", quantity=D("0.1"), price=D("29000"))
    with pytest.raises(DuplicateOrder):
        await place("LIMIT_MAKER", "c6", quantity=D("0.1"), price=D("29000"))
    with pytest.raises(WouldTrigger):
        await client.new_order(
            "BTCUSDT",
            "SELL",
            "STOP_LOSS",
            client_order_id="c7",
            quantity=D("0.1"),
            stop_price=D("31000"),
        )
    with pytest.raises(UnknownOrder):
        await client.cancel_order("BTCUSDT", "never-placed")
    with pytest.raises(UnknownOrder):
        await client.get_order("BTCUSDT", "never-placed")
    with pytest.raises(UnknownOrder):
        await client.get_order_list("never-placed")
    with pytest.raises(ExchangeError) as bad:
        await client.my_trades("NOPEUSDT", 1)
    assert bad.value.code == -1121


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order_type": "MARKET"},
        {"order_type": "MARKET", "quantity": D(1), "quote_order_qty": D(1)},
        {"order_type": "LIMIT_MAKER", "quantity": D(1)},
        {"order_type": "LIMIT_MAKER", "price": D(1)},
        {"order_type": "STOP_LOSS", "quantity": D(1)},
    ],
)
async def test_incomplete_orders_are_refused_before_sending(kwargs: dict[str, object]) -> None:
    fake, client = setup()
    with pytest.raises(ValueError, match="needs"):
        await client.new_order("BTCUSDT", "BUY", client_order_id="c", **kwargs)  # type: ignore[arg-type]
    assert fake.requests == []


# -- typed methods against the fake ------------------------------------------------------------


async def test_exchange_info_and_account() -> None:
    fake, client = setup()
    info = await client.exchange_info()
    assert info.server_time == fake.clock.now()
    assert set(info.symbols) == {"BTCUSDT", "ETHUSDT", "SHIBUSDT", "XRPUSDT", "OLDUSDT"}
    assert info.symbols["BTCUSDT"].tick_size == D("0.01")
    account = await client.account()
    assert account.balance("USDT").free == D("10000")
    assert account.balance("USDT").total == D("10000")
    assert account.balance("DOGE").free == D(0)
    assert account.update_time == fake.clock.now()
    assert (await client.server_time()).tzinfo is UTC


async def test_market_buy_by_quote_returns_full_fills_with_commission() -> None:
    fake, client = setup()
    order = await client.new_order(
        "BTCUSDT", "BUY", "MARKET", client_order_id="buy-1", quote_order_qty=D("1000")
    )
    assert order.status == "FILLED"
    assert not order.is_open
    assert order.client_order_id == "buy-1"
    assert order.order_list_id is None
    assert order.time == fake.clock.now()
    assert order.time is not None
    assert order.time.tzinfo is UTC
    assert order.quantity == D("0.03333")  # 1000 / 30000 rounded down to the step
    assert order.executed_quantity == D("0.03333")
    assert len(order.fills) == 2
    assert sum((f.quantity for f in order.fills), D(0)) == order.executed_quantity
    assert {f.price for f in order.fills} == {D("30000"), D("30000.01")}
    assert all(f.commission_asset == "BTC" for f in order.fills)
    assert order.commission("BTC") == order.executed_quantity * D("0.001")
    assert order.commission("BNB") == D(0)
    assert all(isinstance(f.price, Decimal) and f.trade_id for f in order.fills)
    assert order.cumulative_quote_quantity == sum((f.price * f.quantity for f in order.fills), D(0))
    assert order.average_price == order.cumulative_quote_quantity / order.executed_quantity
    account = await client.account()
    assert account.balance("BTC").free == D("0.03333") - order.commission("BTC")
    assert account.balance("USDT").free == D("10000") - order.cumulative_quote_quantity


async def test_market_sell_by_quantity_charges_commission_in_quote() -> None:
    fake, client = setup()
    fake.balances["BTC"] = [D("0.5"), D(0)]
    order = await client.new_order(
        "BTCUSDT", "SELL", "MARKET", client_order_id="sell-1", quantity=D("0.1")
    )
    assert order.status == "FILLED"
    assert order.commission("USDT") == order.cumulative_quote_quantity * D("0.001")
    assert (await client.account()).balance("BTC").free == D("0.4")


async def test_limit_maker_and_stop_loss_orders() -> None:
    fake, client = setup()
    fake.balances["BTC"] = [D("1"), D(0)]
    limit = await client.new_order(
        "BTCUSDT",
        "SELL",
        "LIMIT_MAKER",
        client_order_id="tp",
        quantity=D("0.4"),
        price=D("33000.5"),
    )
    assert (limit.status, limit.price, limit.quantity) == ("NEW", D("33000.5"), D("0.4"))
    assert limit.is_open
    assert limit.fills == ()
    assert limit.average_price is None
    stop = await client.new_order(
        "BTCUSDT",
        "SELL",
        "STOP_LOSS",
        client_order_id="sl",
        quantity=D("0.3"),
        stop_price=D("27000"),
    )
    assert stop.stop_price == D("27000")
    assert stop.trailing_delta is None
    trailing = await client.new_order(
        "BTCUSDT",
        "SELL",
        "STOP_LOSS",
        client_order_id="tr",
        quantity=D("0.3"),
        trailing_delta=500,
    )
    assert trailing.trailing_delta == 500
    assert trailing.stop_price is None
    params = dict(split_query(fake.requests_to("/api/v3/order")[2])[2])
    assert params["trailingDelta"] == "500"
    assert "stopPrice" not in params
    assert params["newOrderRespType"] == "RESULT"
    account = await client.account()
    assert (account.balance("BTC").free, account.balance("BTC").locked) == (D("0"), D("1"))


async def test_oco_lifecycle_take_profit() -> None:
    fake, client = setup()
    fake.balances["BTC"] = [D("1"), D(0)]
    placed = await client.new_oco(
        "BTCUSDT",
        "SELL",
        D("1"),
        OcoLeg("LIMIT_MAKER", "tp-id", price=D("33000")),
        OcoLeg("STOP_LOSS", "sl-id", stop_price=D("27000")),
        list_client_order_id="list-id",
    )
    assert placed.list_client_order_id == "list-id"
    assert placed.is_open
    assert placed.transaction_time == fake.clock.now()
    assert {leg.client_order_id for leg in placed.orders} == {"tp-id", "sl-id"}
    assert {r.client_order_id: r.order_list_id for r in placed.reports} == {
        "tp-id": placed.order_list_id,
        "sl-id": placed.order_list_id,
    }
    params = dict(split_query(fake.requests_to("/api/v3/orderList/oco")[0])[2])
    assert params["aboveType"] == "LIMIT_MAKER"
    assert params["belowType"] == "STOP_LOSS"
    assert params["abovePrice"] == "33000"
    assert params["belowStopPrice"] == "27000"
    assert params["aboveClientOrderId"] == "tp-id"
    assert params["listClientOrderId"] == "list-id"
    assert "belowTrailingDelta" not in params
    assert "belowPrice" not in params

    assert [ol.list_client_order_id for ol in await client.open_order_lists()] == ["list-id"]
    assert {o.client_order_id for o in await client.open_orders("BTCUSDT")} == {"tp-id", "sl-id"}
    assert {o.client_order_id for o in await client.open_orders()} == {"tp-id", "sl-id"}
    assert await client.open_orders("ETHUSDT") == []

    fake.set_price("BTCUSDT", D("33000"))
    tp = await client.get_order("BTCUSDT", "tp-id")
    sl = await client.get_order("BTCUSDT", "sl-id")
    assert (tp.status, sl.status) == ("FILLED", "EXPIRED")
    assert tp.executed_quantity == D("1")
    assert tp.average_price == D("33000")
    assert tp.update_time is not None
    assert tp.time is not None
    done = await client.get_order_list("list-id")
    assert (done.list_order_status, done.is_open) == ("ALL_DONE", False)
    assert await client.open_order_lists() == []
    trades = await client.my_trades("BTCUSDT", tp.order_id)
    assert len(trades) == 1
    assert trades[0].price == D("33000")
    assert trades[0].is_maker
    assert not trades[0].is_buyer
    assert trades[0].commission == D("33")
    assert trades[0].commission_asset == "USDT"
    assert trades[0].time.tzinfo is UTC
    assert trades[0].quote_quantity == D("33000")
    assert await client.my_trades("BTCUSDT", 999) == []


async def test_oco_trailing_leg_triggers_after_a_pullback() -> None:
    fake, client = setup()
    fake.balances["BTC"] = [D("1"), D(0)]
    placed = await client.new_oco(
        "BTCUSDT",
        "SELL",
        D("1"),
        OcoLeg("LIMIT_MAKER", "tp", price=D("40000")),
        OcoLeg("STOP_LOSS", "tr", trailing_delta=500),
        list_client_order_id="l",
    )
    assert placed.reports[1].trailing_delta == 500
    fake.set_price("BTCUSDT", D("32000"))
    assert (await client.get_order("BTCUSDT", "tr")).status == "NEW"
    fake.set_price("BTCUSDT", D("30400"))  # below 32000 * 0.95
    stopped = await client.get_order("BTCUSDT", "tr")
    assert stopped.status == "FILLED"
    assert stopped.average_price == D("30400")


async def test_cancel_order_and_cancel_list() -> None:
    fake, client = setup()
    fake.balances["BTC"] = [D("2"), D(0)]
    await client.new_order(
        "BTCUSDT", "SELL", "LIMIT_MAKER", client_order_id="solo", quantity=D("1"), price=D("33000")
    )
    cancelled = await client.cancel_order("BTCUSDT", "solo")
    assert (cancelled.client_order_id, cancelled.status) == ("solo", "CANCELED")
    assert (await client.get_order("BTCUSDT", "solo")).status == "CANCELED"
    with pytest.raises(UnknownOrder):
        await client.cancel_order("BTCUSDT", "solo")  # already cancelled
    assert (await client.account()).balance("BTC").free == D("2")

    await client.new_oco(
        "BTCUSDT",
        "SELL",
        D("1"),
        OcoLeg("LIMIT_MAKER", "a", price=D("33000")),
        OcoLeg("STOP_LOSS", "b", stop_price=D("27000")),
        list_client_order_id="l1",
    )
    cancelled_list = await client.cancel_order_list("BTCUSDT", "l1")
    assert cancelled_list.list_order_status == "ALL_DONE"
    assert {r.status for r in cancelled_list.reports} == {"CANCELED"}
    assert (await client.account()).balance("BTC").locked == D("0")

    await client.new_oco(
        "BTCUSDT",
        "SELL",
        D("1"),
        OcoLeg("LIMIT_MAKER", "a2", price=D("33000")),
        OcoLeg("STOP_LOSS", "b2", stop_price=D("27000")),
        list_client_order_id="l2",
    )
    await client.cancel_order("BTCUSDT", "a2")  # cancelling a leg cancels the list
    assert (await client.get_order_list("l2")).list_order_status == "ALL_DONE"
    assert (await client.get_order("BTCUSDT", "b2")).status == "CANCELED"


async def test_oco_price_relationship_is_rejected() -> None:
    fake, client = setup()
    fake.balances["BTC"] = [D("1"), D(0)]
    with pytest.raises(FilterRejected):
        await client.new_oco(
            "BTCUSDT",
            "SELL",
            D("1"),
            OcoLeg("LIMIT_MAKER", "a", price=D("29000")),  # below the market
            OcoLeg("STOP_LOSS", "b", stop_price=D("27000")),
            list_client_order_id="l",
        )


# -- secrets -----------------------------------------------------------------------------------


async def test_keys_never_appear_in_logs_errors_or_reprs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    fake, client = setup()
    secrets = [API_KEY, API_SECRET]
    errors: list[ExchangeError] = []

    async def attempt(coro: object) -> None:
        try:
            await coro  # type: ignore[misc]
        except ExchangeError as exc:
            errors.append(exc)

    await client.exchange_info()
    await client.account()
    await client.new_order(
        "BTCUSDT", "BUY", "MARKET", client_order_id="cid-1", quote_order_qty=D("100")
    )
    await attempt(client.cancel_order("BTCUSDT", "missing"))
    await attempt(
        client.new_order("BTCUSDT", "BUY", "MARKET", client_order_id="c", quantity=D("1E-9"))
    )
    fake.timeout("/api/v3/order")
    await attempt(
        client.new_order("BTCUSDT", "BUY", "MARKET", client_order_id="c2", quote_order_qty=D("10"))
    )
    fake.fail("/api/v3/order", status=503, code=-1007, msg="Timeout waiting for response.")
    await attempt(
        client.new_order("BTCUSDT", "BUY", "MARKET", client_order_id="c3", quote_order_qty=D("10"))
    )
    bad_secret = make_client(fake, api_secret=SecretStr("another-fake-secret"))
    await attempt(bad_secret.account())
    secrets.append("another-fake-secret")
    fake.fail("/api/v3/account", status=429, code=-1003, msg="x", times=99)
    await attempt(client.account())
    fake.faults.clear()
    fake.ban()
    await attempt(client.account())
    assert len(errors) == 7

    signatures = {split_query(r)[1] for r in fake.requests if "signature=" in r.url.query.decode()}
    assert signatures
    leaks = [*secrets, *signatures]
    for exc in errors:
        for text in (str(exc), repr(exc), exc.message, repr(exc.args)):
            assert not any(leak in text for leak in leaks), text
            assert "signature=" not in text.lower()
    for text in (repr(client), str(client)):
        assert not any(leak in text for leak in leaks)
    assert not any(leak in caplog.text for leak in leaks)
    assert "signature=" not in caplog.text
    assert "timestamp=" not in caplog.text
    assert "HTTP Request" in caplog.text  # httpx did log: the unsigned requests (time, info)
    assert "/api/v3/time" in caplog.text


def test_settings_keys_are_secret_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MOONSHOT_BINANCE_TESTNET_API_KEY",
        "MOONSHOT_BINANCE_TESTNET_API_SECRET",
        "MOONSHOT_BINANCE_TESTNET_REST_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    empty = Settings()
    assert empty.binance_testnet_api_key is None
    assert empty.binance_testnet_api_secret is None
    assert empty.binance_testnet_rest_url == "https://testnet.binance.vision"
    with pytest.raises(ValueError, match="TESTNET_API_KEY"):
        SignedClient.from_settings(empty)

    monkeypatch.setenv("MOONSHOT_BINANCE_TESTNET_API_KEY", "env-fake-key-123")
    monkeypatch.setenv("MOONSHOT_BINANCE_TESTNET_API_SECRET", "env-fake-secret-456")
    monkeypatch.setenv("MOONSHOT_BINANCE_TESTNET_REST_URL", "http://localhost:1")
    settings = Settings()
    assert settings.binance_testnet_api_key is not None
    assert settings.binance_testnet_api_key.get_secret_value() == "env-fake-key-123"
    assert settings.binance_testnet_rest_url == "http://localhost:1"
    shown = repr(settings) + str(settings)
    assert "env-fake-key-123" not in shown
    assert "env-fake-secret-456" not in shown
    client = SignedClient.from_settings(settings)
    assert "http://localhost:1" in repr(client)
    assert "env-fake" not in repr(client)


def test_clock_helper_is_aware() -> None:
    assert T0.tzinfo is UTC
