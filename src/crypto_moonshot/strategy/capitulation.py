"""Capitulation: buy a volume spike that ends a sharp 4-hour fall while BTC also spikes.

One parameterised rule; the research's two picks are presets of it:

- `F2` ("deep liquid flush", PR #8 `m6_test.py`): spike & `ret_4h <= -10%` &
  `qv_base >= 20,000` & `btc_vol_mult >= 6`; hold 1 hour. Traded with 5 equal slots.
- `R2` (capitulation long shot, cnn-replication `m18_longshot.py`): the same with
  `qv_base >= 1,000`; take profit +20% or 3 days, no stop. Traded with a $50 stake, 200
  positions at most, entries only for signals before `R2_LAST_ENTRY`, on the `all` universe.

A spike is a candle with `vol_mult >= spike_mult` (see `features.spikes`). Priority is `ret_4h`
ascending: the deepest fall first.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import polars as pl

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.models import Interval
from crypto_moonshot.strategy.features import FEATURES, btc_context, spikes

BTC = "BTCUSDT"


@dataclass(frozen=True, slots=True)
class CapitulationRule:
    """A `SignalRule`: spike & `ret_4h <= max_ret_4h` & `qv_base >= min_qv_base` &
    `btc_vol_mult >= min_btc_vol_mult` (no BTC condition when that is `None`)."""

    name: str
    min_qv_base: float
    exit: ExitPlan
    max_ret_4h: float = -0.10
    min_btc_vol_mult: float | None = 6.0
    spike_mult: float = 10.0
    interval: Interval = Interval.M5
    context_symbols: tuple[str, ...] = field(default=(BTC,))

    def key(self) -> dict[str, object]:
        return {
            "rule": "capitulation",
            "interval": self.interval.value,
            "spike_mult": self.spike_mult,
            "min_qv_base": self.min_qv_base,
            "max_ret_4h": self.max_ret_4h,
            "min_btc_vol_mult": self.min_btc_vol_mult,
            "context": list(self.context_symbols),
        }

    def candidates(self, candles: pl.DataFrame) -> pl.DataFrame:
        return spikes(candles, self.spike_mult, self.interval)

    def context(self, series: Mapping[str, pl.DataFrame]) -> pl.DataFrame | None:
        btc = series.get(BTC)
        if btc is None:
            msg = f"the capitulation rule needs {BTC} candles for its context"
            raise ValueError(msg)
        return btc_context(btc, self.interval)

    def select(self, candidates: pl.DataFrame) -> pl.DataFrame:
        mask = (pl.col("ret_4h") <= self.max_ret_4h) & (pl.col("qv_base") >= self.min_qv_base)
        if self.min_btc_vol_mult is not None:
            mask = mask & (pl.col("btc_vol_mult") >= self.min_btc_vol_mult)
        return (
            candidates.filter(mask.fill_null(False))
            .select(
                "symbol", "open_time", pl.col("ret_4h").alias("priority"), *FEATURES, "btc_vol_mult"
            )
            .sort("open_time", "priority", "symbol")
        )


F2 = CapitulationRule("f2", min_qv_base=20_000, exit=ExitPlan(max_hold=timedelta(hours=1)))
R2 = CapitulationRule(
    "r2", min_qv_base=1_000, exit=ExitPlan(take_profit=0.20, max_hold=timedelta(days=3))
)
R2_LAST_ENTRY = datetime(2026, 10, 1, tzinfo=UTC)
"""The research took R2 entries only for signals before this time (`BacktestConfig.end`)."""
