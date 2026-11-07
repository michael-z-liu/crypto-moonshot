"""Small synthetic candle series for broker tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np

from crypto_moonshot.broker.candles import Candles, to_us
from crypto_moonshot.models import Interval

T0 = datetime(2024, 1, 1, tzinfo=UTC)

type Bar = tuple[float, float, float, float]


def candles(
    bars: Sequence[Bar | None],
    *,
    interval: Interval = Interval.M5,
    start: datetime = T0,
    ended: bool = True,
) -> Candles:
    """Candles from (open, high, low, close) tuples, one per interval from `start`.

    `None` leaves a hole: that candle is missing.
    """
    length = interval.length
    rows = [(start + k * length, bar) for k, bar in enumerate(bars) if bar is not None]
    prices = np.array([bar for _, bar in rows], dtype=np.float64).reshape(-1, 4)
    return Candles(
        interval=interval,
        open_time=np.array([to_us(t) for t, _ in rows], dtype=np.int64),
        open=prices[:, 0].copy(),
        high=prices[:, 1].copy(),
        low=prices[:, 2].copy(),
        close=prices[:, 3].copy(),
        ended=ended,
    )


def flat(n: int, price: float = 100.0) -> list[Bar | None]:
    """`n` candles that never move from `price`."""
    return [(price, price, price, price)] * n
