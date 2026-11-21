"""Symbol trading rules from `GET /api/v3/exchangeInfo`, with exact `Decimal` rounding.

Prices and quantities are `Decimal` throughout: a float tick (0.1 + 0.2) or a tiny step
(0.00000100) would send the exchange a value it rejects. `format_decimal` writes a `Decimal` the
way the API wants it: plain digits, no exponent.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, Protocol

from crypto_moonshot.exchange.errors import ExchangeError

ZERO = Decimal(0)
REFRESH_AFTER = timedelta(hours=1)
"""`exchangeInfo` weighs 20: refresh at most once an hour..."""
MIN_FORCED_REFRESH_GAP = timedelta(minutes=1)
"""...or after a filter rejection, but never more often than this."""

Rounding = Literal["down", "up", "nearest"]


def format_decimal(value: Decimal) -> str:
    """`value` as plain decimal digits without exponent or trailing zeros: `1E+2` -> `100`,
    `0.00000100` -> `0.000001`."""
    if not value.is_finite():
        msg = f"cannot send a non-finite number to the exchange: {value}"
        raise ValueError(msg)
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def round_to_step(value: Decimal, step: Decimal, mode: Rounding = "down") -> Decimal:
    """`value` as a multiple of `step` (exact; a zero step means no constraint)."""
    if step <= 0:
        return value
    multiples = value // step  # exact integer, truncated toward zero
    if multiples * step > value:  # negative values: make it a floor
        multiples -= 1
    low = multiples * step
    if mode == "down" or low == value:
        return low
    if mode == "up":
        return low + step
    return low + step if (value - low) * 2 >= step else low


def _dec(raw: Any, default: Decimal = ZERO) -> Decimal:
    return default if raw is None else Decimal(str(raw))


@dataclass(frozen=True, slots=True)
class SymbolRules:
    """Status, assets and filters of one symbol. A zero tick, step, minimum or maximum means the
    exchange sets no such limit."""

    symbol: str
    status: str
    base_asset: str
    quote_asset: str
    tick_size: Decimal = ZERO
    min_price: Decimal = ZERO
    max_price: Decimal = ZERO
    lot_step: Decimal = ZERO
    min_qty: Decimal = ZERO
    max_qty: Decimal = ZERO
    market_step: Decimal = ZERO
    market_min_qty: Decimal = ZERO
    market_max_qty: Decimal = ZERO
    min_notional: Decimal = ZERO
    min_notional_applies_to_market: bool = True
    max_notional: Decimal = ZERO
    max_notional_applies_to_market: bool = True
    bid_multiplier_up: Decimal = ZERO
    bid_multiplier_down: Decimal = ZERO
    ask_multiplier_up: Decimal = ZERO
    ask_multiplier_down: Decimal = ZERO
    min_trailing_above_delta: int = 0
    max_trailing_above_delta: int = 0
    min_trailing_below_delta: int = 0
    max_trailing_below_delta: int = 0
    max_num_orders: int = 0
    order_types: tuple[str, ...] = ()
    quote_order_qty_market_allowed: bool = True

    @property
    def is_trading(self) -> bool:
        return self.status == "TRADING"

    # -- parsing ----------------------------------------------------------------------------

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> SymbolRules:
        """Parse one `symbols[]` entry of `exchangeInfo`."""
        filters = {f["filterType"]: f for f in raw.get("filters", [])}
        price = filters.get("PRICE_FILTER", {})
        lot = filters.get("LOT_SIZE", {})
        market_lot = filters.get("MARKET_LOT_SIZE", {})
        # Newer symbols carry NOTIONAL, older ones MIN_NOTIONAL (applyToMarket).
        notional = filters.get("NOTIONAL", {})
        legacy = filters.get("MIN_NOTIONAL", {})
        percent = filters.get("PERCENT_PRICE_BY_SIDE", {})
        trailing = filters.get("TRAILING_DELTA", {})
        return cls(
            symbol=str(raw["symbol"]),
            status=str(raw.get("status", "")),
            base_asset=str(raw.get("baseAsset", "")),
            quote_asset=str(raw.get("quoteAsset", "")),
            tick_size=_dec(price.get("tickSize")),
            min_price=_dec(price.get("minPrice")),
            max_price=_dec(price.get("maxPrice")),
            lot_step=_dec(lot.get("stepSize")),
            min_qty=_dec(lot.get("minQty")),
            max_qty=_dec(lot.get("maxQty")),
            market_step=_dec(market_lot.get("stepSize")),
            market_min_qty=_dec(market_lot.get("minQty")),
            market_max_qty=_dec(market_lot.get("maxQty")),
            min_notional=_dec(notional.get("minNotional", legacy.get("minNotional"))),
            min_notional_applies_to_market=bool(
                notional.get("applyMinToMarket", legacy.get("applyToMarket", True))
            ),
            max_notional=_dec(notional.get("maxNotional")),
            max_notional_applies_to_market=bool(notional.get("applyMaxToMarket", True)),
            bid_multiplier_up=_dec(percent.get("bidMultiplierUp")),
            bid_multiplier_down=_dec(percent.get("bidMultiplierDown")),
            ask_multiplier_up=_dec(percent.get("askMultiplierUp")),
            ask_multiplier_down=_dec(percent.get("askMultiplierDown")),
            min_trailing_above_delta=int(trailing.get("minTrailingAboveDelta", 0)),
            max_trailing_above_delta=int(trailing.get("maxTrailingAboveDelta", 0)),
            min_trailing_below_delta=int(trailing.get("minTrailingBelowDelta", 0)),
            max_trailing_below_delta=int(trailing.get("maxTrailingBelowDelta", 0)),
            max_num_orders=int(filters.get("MAX_NUM_ORDERS", {}).get("maxNumOrders", 0)),
            order_types=tuple(str(t) for t in raw.get("orderTypes", ())),
            quote_order_qty_market_allowed=bool(raw.get("quoteOrderQtyMarketAllowed", True)),
        )

    # -- rounding ---------------------------------------------------------------------------

    def round_price(self, price: Decimal, mode: Rounding = "nearest") -> Decimal:
        """`price` on the tick grid (`down`, `up` or `nearest`, halves up)."""
        return round_to_step(price, self.tick_size, mode)

    def round_quantity(self, quantity: Decimal, *, market: bool = False) -> Decimal:
        """`quantity` rounded down to the step (the market step for market orders, if set)."""
        return round_to_step(quantity, self._step(market), "down")

    def _step(self, market: bool) -> Decimal:
        return self.market_step if market and self.market_step > 0 else self.lot_step

    # -- checks: each returns the name of the failed filter, or None --------------------------

    def price_error(self, price: Decimal) -> str | None:
        """`PRICE_FILTER` failure of a limit or stop price, or None."""
        if price <= 0 or (self.min_price > 0 and price < self.min_price):
            return "PRICE_FILTER"
        if self.max_price > 0 and price > self.max_price:
            return "PRICE_FILTER"
        if self.tick_size > 0 and price % self.tick_size != 0:
            return "PRICE_FILTER"
        return None

    def quantity_error(self, quantity: Decimal, *, market: bool = False) -> str | None:
        """`LOT_SIZE` (or `MARKET_LOT_SIZE`) failure of a base quantity, or None."""
        use_market = market and (self.market_step > 0 or self.market_max_qty > 0)
        name = "MARKET_LOT_SIZE" if use_market else "LOT_SIZE"
        low = self.market_min_qty if use_market else self.min_qty
        high = self.market_max_qty if use_market else self.max_qty
        step = self._step(market)
        if quantity <= 0 or (low > 0 and quantity < low) or (high > 0 and quantity > high):
            return name
        if step > 0 and quantity % step != 0:
            return name
        return None

    def notional_error(
        self, price: Decimal, quantity: Decimal, *, market: bool = False
    ) -> str | None:
        """`NOTIONAL` failure of `price * quantity`, or None. For market orders `price` is the
        expected price and the bounds count only if the filter applies to market orders."""
        value = price * quantity
        check_min = not market or self.min_notional_applies_to_market
        check_max = not market or self.max_notional_applies_to_market
        if check_min and self.min_notional > 0 and value < self.min_notional:
            return "NOTIONAL"
        if check_max and self.max_notional > 0 and value > self.max_notional:
            return "NOTIONAL"
        return None

    def meets_min_notional(self, quote_amount: Decimal) -> bool:
        """Whether spending `quote_amount` (e.g. a quote-quantity market buy) meets the minimum."""
        return quote_amount >= self.min_notional

    def trailing_delta_error(self, bips: int, *, above: bool = False) -> str | None:
        """`TRAILING_DELTA` failure of a trailing delta in basis points, or None. `above` is the
        take-profit side of an OCO; a stop-loss sell trails below."""
        low = self.min_trailing_above_delta if above else self.min_trailing_below_delta
        high = self.max_trailing_above_delta if above else self.max_trailing_below_delta
        if bips <= 0 or (low > 0 and bips < low) or (high > 0 and bips > high):
            return "TRAILING_DELTA"
        return None


@dataclass(frozen=True, slots=True)
class ExchangeInfo:
    """A parsed `exchangeInfo` response."""

    server_time: datetime
    symbols: Mapping[str, SymbolRules] = field(default_factory=dict)


class ExchangeInfoSource(Protocol):
    """What `ExchangeRules` needs from a client (`SignedClient` satisfies it)."""

    async def exchange_info(self) -> ExchangeInfo: ...


class ExchangeRules:
    """Cache of every symbol's rules. Refreshed lazily: at most hourly, or after a filter
    rejection (`note_filter_rejection`), never more than once a minute."""

    def __init__(
        self,
        source: ExchangeInfoSource,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_age: timedelta = REFRESH_AFTER,
        min_gap: timedelta = MIN_FORCED_REFRESH_GAP,
    ) -> None:
        self._source = source
        self._now = now
        self._max_age = max_age
        self._min_gap = min_gap
        self._symbols: Mapping[str, SymbolRules] = {}
        self._fetched_at: datetime | None = None
        self._stale = False
        self._lock = asyncio.Lock()

    @property
    def symbols(self) -> Sequence[str]:
        return list(self._symbols)

    async def refresh(self, *, force: bool = False) -> None:
        """Fetch `exchangeInfo` if the cache is empty, older than `max_age` or marked stale.
        `force` refreshes anyway (but still not twice within `min_gap`)."""
        async with self._lock:
            now = self._now()
            if self._fetched_at is not None:
                age = now - self._fetched_at
                if age < self._min_gap or not (force or self._stale or age >= self._max_age):
                    return
            info = await self._source.exchange_info()
            self._symbols = dict(info.symbols)
            self._fetched_at = now
            self._stale = False

    def note_filter_rejection(self) -> None:
        """A `-1013` filter rejection suggests our rules are out of date: refresh on next use."""
        self._stale = True

    async def get(self, symbol: str) -> SymbolRules:
        """Rules of `symbol` (refreshing first if due). Raises `ExchangeError` if unlisted."""
        await self.refresh()
        rules = self._symbols.get(symbol)
        if rules is None:
            msg = f"symbol {symbol} is not listed on the exchange"
            raise ExchangeError(msg)
        return rules

    def peek(self, symbol: str) -> SymbolRules | None:
        """Cached rules without any refresh."""
        return self._symbols.get(symbol)
