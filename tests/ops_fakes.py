"""Helpers for the operations tests: positions and batches written straight to a state file, a
feed whose health the test controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from engine_fakes import FakeFeed

from crypto_moonshot.broker.base import Order, Side, client_order_id
from crypto_moonshot.live.feed import FeedHealth
from crypto_moonshot.live.state import (
    BatchStatus,
    OrderRecord,
    PositionRecord,
    SignalOutcome,
    SignalRow,
    StateStore,
)
from crypto_moonshot.models import Interval


def add_position(
    store: StateStore,
    strategy: str,
    symbol: str,
    placed: datetime,
    *,
    pnl: float | None = None,
    exit_time: datetime | None = None,
    cost: float = 100.0,
    interval: Interval = Interval.H1,
    quantity: float = 1.0,
    entry_price: float = 100.0,
) -> PositionRecord:
    """A pending position; with `exit_time` it is open, and with `pnl` too it is closed at
    `exit_time` with that P&L."""
    order = Order(
        symbol,
        Side.BUY,
        placed,
        quote_amount=cost,
        id=client_order_id(strategy, symbol, placed - interval.length, Side.BUY),
    )
    store.insert_order(OrderRecord.from_order(order, strategy, interval, None))
    position = store.insert_position(
        strategy, symbol, interval, placed, order.id, cost, signal_info={"score": 1}
    )
    if pnl is None and exit_time is None:
        return position
    store.mark_order_filled(order.id)
    store.open_position(
        position.id,
        entry_time=placed,
        entry_ref=entry_price,
        entry_price=entry_price,
        quantity=quantity,
        cost=cost,
    )
    if pnl is None:
        opened = store.position(position.id)
        assert opened is not None
        return opened
    assert exit_time is not None
    return store.close_position(
        position.id,
        exit_time=exit_time,
        exit_price=entry_price * (1 + pnl / cost),
        exit_reason="take_profit" if pnl >= 0 else "stop",
        proceeds=cost + pnl,
    )


def add_batch(
    store: StateStore,
    strategy: str,
    candle: datetime,
    status: BatchStatus,
    *,
    interval: Interval = Interval.H1,
    lateness: float | None = None,
    signals: list[SignalRow] | None = None,
) -> int:
    """A batch (received `lateness` seconds after its candle closed unless missed)."""
    delay = None if lateness is None else timedelta(seconds=lateness)
    record = store.record_batch(
        strategy,
        interval,
        candle,
        status,
        received_at=None if delay is None else candle + interval.length + delay,
        lateness=delay,
        signals=signals or [],
    )
    return record.id


def set_outcome(store: StateStore, signal_id: int, outcome: SignalOutcome, reason: str) -> None:
    store.set_signal_outcome(signal_id, outcome, reason)


@dataclass
class HealthFeed(FakeFeed):
    """A `FakeFeed` whose health the test sets."""

    connected: bool = True
    last_message_at: datetime | None = None

    def health(self) -> FeedHealth:
        return FeedHealth(self.connected, 1, self.last_message_at, 0, None, {})
