"""Request pacing for the signed client: weight per minute and order counts per 10 s / day.

Weight follows `X-MBX-USED-WEIGHT-1M` as in the public `BinanceClient`; order counts follow
`X-MBX-ORDER-COUNT-10S` / `-1D` plus the orders this client has sent itself, so the limits hold
even when the headers lag. All waiting uses the injected clock and sleep.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
from datetime import datetime, timedelta

from crypto_moonshot.data.binance import (
    DEFAULT_WEIGHT_BUDGET,
    USED_WEIGHT_HEADER,
    Clock,
    Sleep,
)
from crypto_moonshot.exchange.errors import RateLimited

ORDER_COUNT_10S_HEADER = "X-MBX-ORDER-COUNT-10S"
ORDER_COUNT_1D_HEADER = "X-MBX-ORDER-COUNT-1D"
DEFAULT_ORDER_LIMIT_10S = 50
DEFAULT_ORDER_LIMIT_1D = 160_000
"""The documented testnet examples (the real limits are per account; see the reference)."""
_MINUTE_MARGIN_S = 0.5
_ORDER_WINDOW = timedelta(seconds=10)
_ORDER_MARGIN_S = 0.05


def _int_header(headers: Mapping[str, str], name: str) -> int | None:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


class Pacer:
    """Waits until a request fits the weight budget and the order-count limits."""

    def __init__(
        self,
        now: Clock,
        sleep: Sleep,
        *,
        weight_budget: int = DEFAULT_WEIGHT_BUDGET,
        order_limit_10s: int = DEFAULT_ORDER_LIMIT_10S,
        order_limit_1d: int = DEFAULT_ORDER_LIMIT_1D,
    ) -> None:
        if min(weight_budget, order_limit_10s, order_limit_1d) < 1:
            msg = "weight_budget and the order limits must be at least 1"
            raise ValueError(msg)
        self._now = now
        self._sleep = sleep
        self.weight_budget = weight_budget
        self.order_limit_10s = order_limit_10s
        self.order_limit_1d = order_limit_1d
        self.max_used_weight = 0
        self._lock = asyncio.Lock()
        self._inflight = 0
        self._used_weight = 0
        self._used_minute: datetime | None = None
        self._pause_until: datetime | None = None
        self._sent: deque[tuple[datetime, int]] = deque()
        """Orders this client sent in the last 10 s: (time, count)."""
        self._hdr_10s: tuple[int, datetime] | None = None
        self._hdr_1d: tuple[int, datetime] | None = None
        self._sent_day: list[tuple[datetime, int]] = []

    # -- reserve / release ------------------------------------------------------------------

    def pause(self, seconds: float) -> None:
        """Make every later request wait `seconds` (after an HTTP 429)."""
        until = self._now() + timedelta(seconds=seconds)
        if self._pause_until is None or until > self._pause_until:
            self._pause_until = until

    async def reserve(self, weight: int, orders: int = 0) -> None:
        """Wait until `weight` (and `orders` new orders) fit, then count them as sent.

        Raises `RateLimited` instead of waiting when the daily order limit is used up.
        """
        async with self._lock:
            if self._pause_until is not None:
                remaining = (self._pause_until - self._now()).total_seconds()
                self._pause_until = None
                if remaining > 0:
                    await self._sleep(remaining)
            while True:
                now = self._now()
                used = self._weight_used(now)
                if used + self._inflight + weight <= self.weight_budget:
                    break
                await self._sleep(
                    (self._minute(now) + timedelta(minutes=1) - now).total_seconds()
                    + _MINUTE_MARGIN_S
                )
                self._used_weight, self._used_minute = 0, None
            if orders:
                await self._reserve_orders(orders)
            self._inflight += weight

    async def _reserve_orders(self, orders: int) -> None:
        if orders > self.order_limit_10s:
            msg = f"{orders} orders can never fit the {self.order_limit_10s} per 10 s limit"
            raise ValueError(msg)
        while True:
            now = self._now()
            day = self._orders_today(now)
            if day + orders > self.order_limit_1d:
                midnight = (now + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                msg = f"daily order limit ({self.order_limit_1d}) used up"
                raise RateLimited(msg, retry_after=(midnight - now).total_seconds())
            count, oldest = self._orders_in_window(now)
            if count + orders <= self.order_limit_10s or oldest is None:
                break
            await self._sleep(
                max((oldest + _ORDER_WINDOW - now).total_seconds(), 0.0) + _ORDER_MARGIN_S
            )
        now = self._now()
        self._sent.append((now, orders))
        self._sent_day.append((now, orders))

    def release(self, weight: int, headers: Mapping[str, str] | None) -> None:
        """Account for a finished request: free its in-flight weight, read the usage headers."""
        self._inflight -= weight
        if headers is None:
            return
        now = self._now()
        used = _int_header(headers, USED_WEIGHT_HEADER)
        if used is not None:
            window = self._minute(now)
            if self._used_minute == window:
                used = max(used, self._used_weight)
            self._used_weight, self._used_minute = used, window
            self.max_used_weight = max(self.max_used_weight, used)
        ten = _int_header(headers, ORDER_COUNT_10S_HEADER)
        if ten is not None:
            self._hdr_10s = (ten, now)
        day = _int_header(headers, ORDER_COUNT_1D_HEADER)
        if day is not None:
            self._hdr_1d = (day, now)

    # -- bookkeeping ------------------------------------------------------------------------

    @staticmethod
    def _minute(now: datetime) -> datetime:
        return now.replace(second=0, microsecond=0)

    def _weight_used(self, now: datetime) -> int:
        return self._used_weight if self._used_minute == self._minute(now) else 0

    def _orders_in_window(self, now: datetime) -> tuple[int, datetime | None]:
        """Orders counted against the 10 s limit, and the time of the oldest that counts."""
        while self._sent and now - self._sent[0][0] >= _ORDER_WINDOW:
            self._sent.popleft()
        local = sum(n for _, n in self._sent)
        oldest = self._sent[0][0] if self._sent else None
        if self._hdr_10s is not None:
            count, seen = self._hdr_10s
            if now - seen >= _ORDER_WINDOW:
                self._hdr_10s = None
            else:
                since = sum(n for t, n in self._sent if t > seen)
                if count + since > local:
                    return count + since, seen if oldest is None else min(seen, oldest)
        return local, oldest

    def _orders_today(self, now: datetime) -> int:
        today = now.date()
        self._sent_day = [(t, n) for t, n in self._sent_day if t.date() == today]
        local = sum(n for _, n in self._sent_day)
        if self._hdr_1d is None:
            return local
        count, seen = self._hdr_1d
        if seen.date() != today:
            self._hdr_1d = None
            return local
        return max(local, count + sum(n for t, n in self._sent_day if t > seen))
