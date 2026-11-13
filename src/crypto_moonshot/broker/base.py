"""Broker interface and the order, fill and exit models shared by every broker.

Prices are USDT per unit of the base asset; quantities are units of the base asset. Everything is
long only and quoted in USDT, so a symbol such as `BTCUSDT` holds the base asset `BTC`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, Protocol
from uuid import uuid4

from crypto_moonshot.models import as_utc

QUOTE_ASSET = "USDT"


def base_asset(symbol: str) -> str:
    """Base asset of a USDT-quoted symbol: `BTCUSDT` -> `BTC`."""
    if not symbol.endswith(QUOTE_ASSET) or symbol == QUOTE_ASSET:
        msg = f"only {QUOTE_ASSET}-quoted symbols are supported, got {symbol!r}"
        raise ValueError(msg)
    return symbol.removesuffix(QUOTE_ASSET)


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    """Fills at the open of the first candle at or after the order time, plus costs."""
    BRACKET = "bracket"
    """The sell side of an `ExitPlan`: take profit, stop, trailing stop and time exit as one-
    cancels-the-others orders. Brokers create these when a buy carries an exit plan."""


class ExitReason(StrEnum):
    TAKE_PROFIT = "take_profit"
    STOP = "stop"
    TRAILING_STOP = "trailing_stop"
    TIME = "time"
    SERIES_END = "series_end"
    """The candle series ended (delisting, or the end of the data) before any other exit."""


@dataclass(frozen=True, slots=True)
class ExitPlan:
    """Exits attached to an entry; the first to fill cancels the rest (bracket / OCO).

    Levels are fractions relative to the *entry reference price*: the raw open of the entry
    candle, before slippage and latency (the research convention, where forward paths are
    relative to the open of the entry candle).

    - `take_profit`: sell limit at `ref * (1 + take_profit)`.
    - `stop`: stop at `ref * (1 - stop)`.
    - `trailing`: stop at `peak * (1 - trailing)`, where `peak` is the highest high of the
      candles before the current one since entry, starting from `ref`.
    - `max_hold`: time exit at `entry time + max_hold`, at the close of the last candle that
      opens before it.
    - `tp_fill`: `touch` fills the take profit when a candle's high reaches the target;
      `close` only when a candle closes at or above it (a stricter check that the price really
      traded there, used with 1m candles).
    """

    take_profit: float | None = None
    stop: float | None = None
    trailing: float | None = None
    max_hold: timedelta | None = None
    tp_fill: Literal["touch", "close"] = "touch"

    def __post_init__(self) -> None:
        if self.take_profit is not None and not self.take_profit > 0:
            msg = f"take_profit must be > 0, got {self.take_profit}"
            raise ValueError(msg)
        for name in ("stop", "trailing"):
            value = getattr(self, name)
            if value is not None and not 0 < value < 1:
                msg = f"{name} must be between 0 and 1, got {value}"
                raise ValueError(msg)
        if self.max_hold is not None and self.max_hold <= timedelta(0):
            msg = f"max_hold must be positive, got {self.max_hold}"
            raise ValueError(msg)
        if self.tp_fill not in ("touch", "close"):
            msg = f"tp_fill must be 'touch' or 'close', got {self.tp_fill!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Costs:
    """Trading costs per side, as fractions.

    Fill prices, from a raw price `p` (a candle open or close, or an exit level):

    - market buy: `p * (1 + slippage + latency)`
    - stop, trailing stop, time exit, series end (market orders once triggered):
      `p * (1 - slippage - latency)`
    - take profit (a resting limit): `p * (1 - slippage)`; slippage is kept on the limit side
      for comparability with the research, which charged it on every side.

    The fee is `fee * notional` on each side, paid in USDT. Buying with `cash` USDT gets
    `cash / (buy_price * (1 + fee))` units; selling `q` units returns `q * sell_price * (1 - fee)`.
    So the net return of a trade is `sell_price * (1 - fee) / (buy_price * (1 + fee)) - 1`.
    The research used an additive `gross - 0.003`; the multiplicative form differs slightly
    (about 0.04% on a +20% exit).
    """

    fee: float = 0.001
    slippage: float = 0.0005
    latency: float = 0.0

    def __post_init__(self) -> None:
        for name in ("fee", "slippage", "latency"):
            value = getattr(self, name)
            if not 0 <= value < 1:
                msg = f"{name} must be between 0 and 1, got {value}"
                raise ValueError(msg)

    def buy_price(self, raw: float) -> float:
        """Fill price of a market buy at raw price `raw`."""
        return raw * (1 + self.slippage + self.latency)

    def sell_price(self, raw: float, *, market: bool) -> float:
        """Fill price of a sell at raw price `raw`; `market` adds the latency cost."""
        return raw * (1 - self.slippage - (self.latency if market else 0.0))

    def net_return(self, buy_price: float, sell_price: float) -> float:
        """Net return of a round trip between two fill prices, after both fees."""
        return sell_price * (1 - self.fee) / (buy_price * (1 + self.fee)) - 1


def _new_id() -> str:
    return uuid4().hex


@dataclass(frozen=True, slots=True)
class Order:
    """An order request.

    A market buy spends `quote_amount` USDT, fee included. A market sell sells `quantity` units
    (`None`: the whole holding). A bracket sell is created by the broker from a buy's exit plan.
    `time` is when the order is placed: a simulated market order fills at the open of the first
    candle that opens at or after it.
    """

    symbol: str
    side: Side
    time: datetime
    type: OrderType = OrderType.MARKET
    quote_amount: float | None = None
    quantity: float | None = None
    exit_plan: ExitPlan | None = None
    id: str = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", as_utc(self.time))
        base_asset(self.symbol)
        if self.side is Side.BUY:
            if self.type is not OrderType.MARKET:
                msg = "only market buys are supported"
                raise ValueError(msg)
            if self.quote_amount is None or not self.quote_amount > 0:
                msg = "a buy needs a positive quote_amount"
                raise ValueError(msg)
            if self.quantity is not None:
                msg = "a buy is sized by quote_amount, not quantity"
                raise ValueError(msg)
            return
        if self.quote_amount is not None:
            msg = "a sell is sized by quantity, not quote_amount"
            raise ValueError(msg)
        if self.quantity is not None and not self.quantity > 0:
            msg = "a sell quantity must be positive"
            raise ValueError(msg)
        if (self.type is OrderType.BRACKET) != (self.exit_plan is not None):
            msg = "a bracket sell needs an exit plan, and only a bracket sell takes one"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Fill:
    """An executed order: `price` includes slippage and latency, `fee` is in USDT."""

    order_id: str
    symbol: str
    side: Side
    time: datetime
    price: float
    quantity: float
    fee: float
    reason: ExitReason | None = None
    """Why a sell happened; `None` for buys and plain market sells."""

    @property
    def notional(self) -> float:
        return self.price * self.quantity


class Broker(Protocol):
    """What the engines need from a broker; simulated and exchange brokers implement it."""

    def place_order(self, order: Order) -> Fill | None:
        """Submit an order. Returns its fill if it filled at once, else `None`."""
        ...

    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order; raises `KeyError` if no such order is open."""
        ...

    def balances(self) -> dict[str, float]:
        """Free balance per asset (`USDT` and each base asset held)."""
        ...

    def open_orders(self) -> list[Order]:
        """Orders placed but not yet filled or cancelled."""
        ...
