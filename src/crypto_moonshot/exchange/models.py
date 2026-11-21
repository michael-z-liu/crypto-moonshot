"""Parsed Binance REST responses: `Decimal` prices and quantities, aware UTC datetimes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from crypto_moonshot.models import EPOCH

ZERO = Decimal(0)
OPEN_STATUSES = frozenset({"NEW", "PARTIALLY_FILLED", "PENDING_NEW", "PENDING_CANCEL"})
"""Order statuses of an order that can still fill."""


def from_ms(ms: int) -> datetime:
    return EPOCH + timedelta(milliseconds=int(ms))


def to_ms(dt: datetime) -> int:
    return (dt - EPOCH) // timedelta(milliseconds=1)


def dec(raw: Any, default: Decimal = ZERO) -> Decimal:
    """A Decimal from a Binance string (never via float); `default` if absent."""
    return default if raw is None else Decimal(str(raw))


def _opt_dec(raw: Any) -> Decimal | None:
    return None if raw is None else Decimal(str(raw))


def _opt_time(raw: Any) -> datetime | None:
    return None if raw is None else from_ms(int(raw))


@dataclass(frozen=True, slots=True)
class Balance:
    asset: str
    free: Decimal
    locked: Decimal

    @property
    def total(self) -> Decimal:
        return self.free + self.locked


@dataclass(frozen=True, slots=True)
class Account:
    balances: Mapping[str, Balance]
    can_trade: bool
    update_time: datetime | None

    def balance(self, asset: str) -> Balance:
        """The balance of `asset` (zero if the account holds none)."""
        return self.balances.get(asset, Balance(asset, ZERO, ZERO))

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Account:
        balances = {
            str(b["asset"]): Balance(str(b["asset"]), dec(b.get("free")), dec(b.get("locked")))
            for b in raw.get("balances", [])
        }
        return cls(balances, bool(raw.get("canTrade", True)), _opt_time(raw.get("updateTime")))


@dataclass(frozen=True, slots=True)
class OrderFill:
    """One fill of a `newOrderRespType=FULL` response."""

    price: Decimal
    quantity: Decimal
    commission: Decimal
    commission_asset: str
    trade_id: int | None

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> OrderFill:
        trade_id = raw.get("tradeId")
        return cls(
            price=dec(raw.get("price")),
            quantity=dec(raw.get("qty")),
            commission=dec(raw.get("commission")),
            commission_asset=str(raw.get("commissionAsset", "")),
            trade_id=None if trade_id is None else int(trade_id),
        )


@dataclass(frozen=True, slots=True)
class ExchangeOrder:
    """An order as the exchange reports it. ACK responses leave most numbers at zero."""

    symbol: str
    order_id: int
    client_order_id: str
    order_list_id: int | None
    side: str
    order_type: str
    status: str
    price: Decimal
    quantity: Decimal
    executed_quantity: Decimal
    cumulative_quote_quantity: Decimal
    stop_price: Decimal | None
    trailing_delta: int | None
    time: datetime | None
    """Transaction time (new orders), or the time the order was placed (queries)."""
    update_time: datetime | None
    fills: tuple[OrderFill, ...] = ()

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES

    @property
    def average_price(self) -> Decimal | None:
        """Quote spent or received per base unit, or None before anything executed."""
        if self.executed_quantity <= 0:
            return None
        return self.cumulative_quote_quantity / self.executed_quantity

    def commission(self, asset: str) -> Decimal:
        """Total commission charged in `asset` over the `FULL` response's fills."""
        return sum((f.commission for f in self.fills if f.commission_asset == asset), ZERO)

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> ExchangeOrder:
        list_id = raw.get("orderListId")
        trailing = raw.get("trailingDelta")
        stop = _opt_dec(raw.get("stopPrice"))
        return cls(
            symbol=str(raw["symbol"]),
            order_id=int(raw["orderId"]),
            # Cancel responses carry the cancelled order's ID as `origClientOrderId`.
            client_order_id=str(raw.get("origClientOrderId") or raw["clientOrderId"]),
            order_list_id=None if list_id is None or int(list_id) < 0 else int(list_id),
            side=str(raw.get("side", "")),
            order_type=str(raw.get("type", "")),
            status=str(raw.get("status", "")),
            price=dec(raw.get("price")),
            quantity=dec(raw.get("origQty")),
            executed_quantity=dec(raw.get("executedQty")),
            cumulative_quote_quantity=dec(raw.get("cummulativeQuoteQty")),
            stop_price=stop if stop else None,
            trailing_delta=int(trailing) if trailing else None,
            time=_opt_time(raw.get("transactTime", raw.get("time"))),
            update_time=_opt_time(raw.get("updateTime")),
            fills=tuple(OrderFill.from_json(f) for f in raw.get("fills", ())),
        )


@dataclass(frozen=True, slots=True)
class OrderListLeg:
    symbol: str
    order_id: int
    client_order_id: str


@dataclass(frozen=True, slots=True)
class OrderList:
    """An OCO order list. `reports` (full orders) come with placement and cancel responses."""

    order_list_id: int
    list_client_order_id: str
    symbol: str
    list_status_type: str
    list_order_status: str
    transaction_time: datetime | None
    orders: tuple[OrderListLeg, ...]
    reports: tuple[ExchangeOrder, ...] = ()

    @property
    def is_open(self) -> bool:
        return self.list_order_status == "EXECUTING"

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> OrderList:
        return cls(
            order_list_id=int(raw["orderListId"]),
            list_client_order_id=str(raw["listClientOrderId"]),
            symbol=str(raw["symbol"]),
            list_status_type=str(raw.get("listStatusType", "")),
            list_order_status=str(raw.get("listOrderStatus", "")),
            transaction_time=_opt_time(raw.get("transactionTime")),
            orders=tuple(
                OrderListLeg(str(o["symbol"]), int(o["orderId"]), str(o["clientOrderId"]))
                for o in raw.get("orders", ())
            ),
            reports=tuple(ExchangeOrder.from_json(r) for r in raw.get("orderReports", ())),
        )


@dataclass(frozen=True, slots=True)
class Trade:
    """One fill from `GET /api/v3/myTrades`."""

    trade_id: int
    order_id: int
    symbol: str
    price: Decimal
    quantity: Decimal
    quote_quantity: Decimal
    commission: Decimal
    commission_asset: str
    time: datetime
    is_buyer: bool
    is_maker: bool

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Trade:
        return cls(
            trade_id=int(raw["id"]),
            order_id=int(raw["orderId"]),
            symbol=str(raw["symbol"]),
            price=dec(raw.get("price")),
            quantity=dec(raw.get("qty")),
            quote_quantity=dec(raw.get("quoteQty")),
            commission=dec(raw.get("commission")),
            commission_asset=str(raw.get("commissionAsset", "")),
            time=from_ms(int(raw["time"])),
            is_buyer=bool(raw.get("isBuyer")),
            is_maker=bool(raw.get("isMaker")),
        )


@dataclass(frozen=True, slots=True)
class OcoLeg:
    """One leg of a new OCO list (`new_oco`): its order type, prices and client order ID."""

    order_type: str
    """`LIMIT_MAKER`, `STOP_LOSS`, ... (see the reference)."""
    client_order_id: str
    price: Decimal | None = None
    stop_price: Decimal | None = None
    trailing_delta: int | None = None
    """Trailing delta in basis points."""
