from datetime import timedelta
from decimal import Decimal

import pytest
from exchange_fakes import DEFAULT_SYMBOLS, FakeClock, symbol_json

from crypto_moonshot.exchange.errors import ExchangeError
from crypto_moonshot.exchange.rules import (
    ExchangeInfo,
    ExchangeRules,
    SymbolRules,
    format_decimal,
    round_to_step,
)

pytestmark = pytest.mark.anyio

D = Decimal


def rules(name: str) -> SymbolRules:
    raw = next(s for s in DEFAULT_SYMBOLS if s["symbol"] == name)
    return SymbolRules.from_json(raw)


def test_parses_a_realistic_symbol() -> None:
    btc = rules("BTCUSDT")
    assert (btc.status, btc.base_asset, btc.quote_asset) == ("TRADING", "BTC", "USDT")
    assert btc.is_trading
    assert btc.tick_size == D("0.01")
    assert btc.min_price == D("0.01")
    assert btc.max_price == D("1000000")
    assert (btc.lot_step, btc.min_qty, btc.max_qty) == (D("0.00001"), D("0.00001"), D("9000"))
    assert (btc.market_step, btc.market_max_qty) == (D(0), D("100"))
    assert btc.min_notional == D("5")
    assert btc.min_notional_applies_to_market is True
    assert btc.max_notional == D("9000000")
    assert btc.max_notional_applies_to_market is False
    assert (btc.bid_multiplier_up, btc.ask_multiplier_down) == (D("5"), D("0.2"))
    assert (btc.min_trailing_below_delta, btc.max_trailing_below_delta) == (10, 2000)
    assert (btc.min_trailing_above_delta, btc.max_trailing_above_delta) == (10, 2000)
    assert btc.max_num_orders == 200
    assert "LIMIT_MAKER" in btc.order_types
    assert all(isinstance(v, Decimal) for v in (btc.tick_size, btc.lot_step, btc.min_notional))
    assert rules("OLDUSDT").is_trading is False


def test_legacy_min_notional_filter() -> None:
    raw = symbol_json("OLDUSDT", "OLD", tick="0.01", step="0.01", min_qty="0.01")
    raw["filters"] = [f for f in raw["filters"] if f["filterType"] != "NOTIONAL"] + [
        {"filterType": "MIN_NOTIONAL", "minNotional": "10.0", "applyToMarket": False}
    ]
    parsed = SymbolRules.from_json(raw)
    assert parsed.min_notional == D("10")
    assert parsed.min_notional_applies_to_market is False
    # A market order below the minimum passes when the filter does not apply to market orders.
    assert parsed.notional_error(D("1"), D("1"), market=True) is None
    assert parsed.notional_error(D("1"), D("1")) == "NOTIONAL"


def test_symbol_without_filters_has_no_limits() -> None:
    bare = SymbolRules.from_json({"symbol": "XUSDT"})
    assert bare.round_price(D("1.23456789")) == D("1.23456789")
    assert bare.quantity_error(D("0.5")) is None
    assert bare.price_error(D("0.5")) is None


@pytest.mark.parametrize(
    ("value", "mode", "expected"),
    [
        ("100.129", "down", "100.12"),
        ("100.121", "up", "100.13"),
        ("100.12", "up", "100.12"),
        ("100.124", "nearest", "100.12"),
        ("100.125", "nearest", "100.13"),
        ("100.126", "nearest", "100.13"),
        ("0.004", "down", "0.00"),
        ("0.004", "up", "0.01"),
    ],
)
def test_price_rounding_each_direction(value: str, mode: str, expected: str) -> None:
    btc = rules("BTCUSDT")
    assert btc.round_price(D(value), mode) == D(expected)  # type: ignore[arg-type]


def test_rounding_has_no_float_drift() -> None:
    drifted = D(str(0.1 + 0.2))  # 0.30000000000000004
    assert drifted != D("0.3")
    assert round_to_step(drifted, D("0.1"), "down") == D("0.3")
    assert round_to_step(drifted, D("0.1"), "nearest") == D("0.3")
    assert round_to_step(D("0.3"), D("0.1"), "up") == D("0.3")
    assert round_to_step(D("0.1") + D("0.2"), D("0.1"), "down") == D("0.3")
    assert round_to_step(D("0.7"), D("0.1"), "down") == D("0.7")


def test_tiny_ticks_and_steps() -> None:
    xrp = rules("XRPUSDT")
    assert xrp.tick_size == D("0.000001")
    assert xrp.round_price(D("0.5123456"), "down") == D("0.512345")
    assert xrp.round_price(D("0.5123456"), "up") == D("0.512346")
    assert xrp.price_error(D("0.512345")) is None
    assert xrp.price_error(D("0.5123456")) == "PRICE_FILTER"
    shib = rules("SHIBUSDT")
    assert shib.round_price(D("0.0000123456"), "down") == D("0.00001234")
    assert shib.round_quantity(D("1234567.89")) == D("1234567")
    btc = rules("BTCUSDT")
    assert btc.round_quantity(D("0.123456789")) == D("0.12345")
    assert btc.round_quantity(D("0.00000999")) == D("0")


def test_negative_values_floor() -> None:
    assert round_to_step(D("-0.15"), D("0.1"), "down") == D("-0.2")
    assert round_to_step(D("-0.15"), D("0.1"), "up") == D("-0.1")


def test_market_step_overrides_lot_step_when_set() -> None:
    raw = symbol_json("ABCUSDT", "ABC", tick="0.01", step="0.001", min_qty="0.001")
    for f in raw["filters"]:
        if f["filterType"] == "MARKET_LOT_SIZE":
            f.update(stepSize="0.1", minQty="0.1", maxQty="50")
    parsed = SymbolRules.from_json(raw)
    assert parsed.round_quantity(D("1.2345")) == D("1.234")
    assert parsed.round_quantity(D("1.2345"), market=True) == D("1.2")
    assert parsed.quantity_error(D("1.234"), market=True) == "MARKET_LOT_SIZE"
    assert parsed.quantity_error(D("1.2"), market=True) is None
    assert parsed.quantity_error(D("60"), market=True) == "MARKET_LOT_SIZE"


def test_quantity_bounds_and_step() -> None:
    btc = rules("BTCUSDT")
    assert btc.quantity_error(D("0.00001")) is None
    assert btc.quantity_error(D("0.000001")) == "LOT_SIZE"
    assert btc.quantity_error(D("9000.00001")) == "LOT_SIZE"
    assert btc.quantity_error(D("0.123456")) == "LOT_SIZE"
    assert btc.quantity_error(D("0")) == "LOT_SIZE"
    assert btc.quantity_error(D("101"), market=True) == "MARKET_LOT_SIZE"


def test_price_bounds() -> None:
    btc = rules("BTCUSDT")
    assert btc.price_error(D("0.00")) == "PRICE_FILTER"
    assert btc.price_error(D("1000000.01")) == "PRICE_FILTER"
    assert btc.price_error(D("30000.01")) is None


def test_min_notional() -> None:
    btc = rules("BTCUSDT")
    assert btc.notional_error(D("30000"), D("0.00017")) is None  # 5.1
    assert btc.notional_error(D("30000"), D("0.00016")) == "NOTIONAL"  # 4.8
    assert btc.meets_min_notional(D("5"))
    assert not btc.meets_min_notional(D("4.99"))
    assert btc.notional_error(D("30000"), D("400"), market=True) is None  # max not for market
    assert btc.notional_error(D("30000"), D("400")) == "NOTIONAL"


def test_trailing_delta_bounds() -> None:
    btc = rules("BTCUSDT")
    assert btc.trailing_delta_error(500) is None
    assert btc.trailing_delta_error(10) is None
    assert btc.trailing_delta_error(9) == "TRAILING_DELTA"
    assert btc.trailing_delta_error(2000) is None
    assert btc.trailing_delta_error(2001) == "TRAILING_DELTA"
    assert btc.trailing_delta_error(0) == "TRAILING_DELTA"
    assert btc.trailing_delta_error(2001, above=True) == "TRAILING_DELTA"


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (D("1E+2"), "100"),
        (D("0.00000100"), "0.000001"),
        (D("1E-8"), "0.00000001"),
        (D("123.4500"), "123.45"),
        (D("0"), "0"),
        (D("0E-8"), "0"),
        (D("-0"), "0"),
        (D("1234567890.12345678"), "1234567890.12345678"),
        (D("5E+3"), "5000"),
        (D("-0.5"), "-0.5"),
    ],
)
def test_format_decimal(value: Decimal, text: str) -> None:
    assert format_decimal(value) == text
    assert "e" not in text.lower()


def test_format_decimal_refuses_non_finite() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        format_decimal(D("NaN"))


class StubSource:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.calls = 0

    async def exchange_info(self) -> ExchangeInfo:
        self.calls += 1
        from exchange_fakes import T0

        symbols = [s for s in DEFAULT_SYMBOLS if s["symbol"] in self.names]
        return ExchangeInfo(T0, {s["symbol"]: SymbolRules.from_json(s) for s in symbols})


async def test_cache_refreshes_hourly_and_on_filter_rejection() -> None:
    clock = FakeClock()
    source = StubSource(["BTCUSDT", "ETHUSDT"])
    cache = ExchangeRules(source, now=clock.now)
    assert cache.peek("BTCUSDT") is None
    assert (await cache.get("BTCUSDT")).symbol == "BTCUSDT"
    assert source.calls == 1
    await cache.get("ETHUSDT")
    await cache.refresh()
    assert source.calls == 1  # cached
    assert sorted(cache.symbols) == ["BTCUSDT", "ETHUSDT"]

    clock.current += timedelta(minutes=59)
    await cache.get("BTCUSDT")
    assert source.calls == 1
    clock.current += timedelta(minutes=2)
    await cache.get("BTCUSDT")
    assert source.calls == 2  # an hour old

    cache.note_filter_rejection()
    await cache.get("BTCUSDT")
    assert source.calls == 2  # refreshed less than a minute ago: not again
    clock.current += timedelta(minutes=2)
    await cache.get("BTCUSDT")
    assert source.calls == 3
    await cache.get("BTCUSDT")
    assert source.calls == 3  # the stale mark is cleared


async def test_unlisted_symbol_raises() -> None:
    cache = ExchangeRules(StubSource(["BTCUSDT"]))
    with pytest.raises(ExchangeError, match="NOPEUSDT"):
        await cache.get("NOPEUSDT")


async def test_forced_refresh() -> None:
    clock = FakeClock()
    source = StubSource(["BTCUSDT"])
    cache = ExchangeRules(source, now=clock.now)
    await cache.refresh()
    clock.current += timedelta(minutes=5)
    await cache.refresh(force=True)
    assert source.calls == 2
