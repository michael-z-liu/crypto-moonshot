"""An hourly store where a loosened capitulation rule fires often, and engine-test helpers.

`write_replay_store` writes `BTCUSDT` and eight coins for `N` hourly candles from `T0`: a random
walk whose candles open at the previous close (so no exit can gap through a level), quiet
volume, and at each of `EVENTS` a BTC volume spike while some coins fall 4-12% over four hours
on an 8x volume spike and stay down. The last `QUIET` candles have constant volume, so no signal
fires near the end and every position exits before it.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from store_fakes import kline_frame, write_series

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.live.config import RunConfig, config_from_dict
from crypto_moonshot.live.plugin import RuleStrategy
from crypto_moonshot.models import Interval
from crypto_moonshot.strategy.capitulation import CapitulationRule

H1 = Interval.H1
HOUR = timedelta(hours=1)
T0 = datetime(2024, 1, 1, tzinfo=UTC)
N = 492
QUIET = 50
COINS = tuple(f"{c * 3}USDT" for c in "ABCDEFGH")
EVENTS = (
    206, 214, 218, 226, 238, 250, 262, 263, 280, 292,
    304, 316, 330, 342, 360, 372, 390, 402, 420, 432,
)  # fmt: skip
START = 200
END = N

RULE = CapitulationRule(
    "cap",
    min_qv_base=1.0,
    exit=ExitPlan(take_profit=0.03, stop=0.04, max_hold=12 * HOUR),
    max_ret_4h=-0.03,
    min_btc_vol_mult=3.0,
    spike_mult=4.0,
    interval=H1,
)


def at(index: int) -> datetime:
    return T0 + index * HOUR


def coin_paths(seed: int = 11) -> dict[str, dict[str, np.ndarray]]:
    """close and volume arrays per symbol (BTC first)."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, np.ndarray]] = {}
    btc_volume = rng.lognormal(np.log(1000.0), 0.12, N)
    btc_volume[N - QUIET :] = 1000.0
    for e in EVENTS:
        btc_volume[e] *= 6.0
    out["BTCUSDT"] = {
        "close": 30_000 * np.exp(np.cumsum(rng.normal(0, 0.004, N))),
        "volume": btc_volume,
    }
    for symbol in COINS:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.009, N)))
        volume = rng.lognormal(np.log(1000.0), 0.12, N)
        volume[N - QUIET :] = 1000.0
        out[symbol] = {"close": close, "volume": volume}
    for e in EVENTS:
        hit = rng.choice(len(COINS), size=int(rng.integers(2, 7)), replace=False)
        for k in hit:
            cols = out[COINS[k]]
            depth = float(rng.uniform(0.04, 0.12))
            cols["close"][e - 3 : e + 1] *= np.linspace(1.0, 1.0 - depth, 4)
            cols["close"][e + 1 :] *= 1.0 - depth
            cols["volume"][e] *= 8.0
    return out


def write_replay_store(
    root: Path,
    *,
    holes: Mapping[str, Sequence[int]] | None = None,
    paths: dict[str, dict[str, np.ndarray]] | None = None,
    first: Mapping[str, int] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Write the store; `holes` drops candles by index, `first` lists a symbol from an index."""
    paths = paths or coin_paths()
    rng = np.random.default_rng(5)
    for symbol, cols in paths.items():
        close = cols["close"]
        open_ = np.concatenate(([close[0]], close[:-1]))
        wick = np.abs(rng.normal(0, 0.004, (2, N)))
        keep = np.ones(N, dtype=bool)
        keep[list((holes or {}).get(symbol, ()))] = False
        keep[: (first or {}).get(symbol, 0)] = False
        idx = np.flatnonzero(keep)
        frame = kline_frame(
            [at(int(k)) for k in idx],
            close[idx],
            cols["volume"][idx],
            open_=open_[idx],
            high=(np.maximum(open_, close) * (1 + wick[0]))[idx],
            low=(np.minimum(open_, close) * (1 - wick[1]))[idx],
        )
        write_series(root, symbol, H1, frame)
    return paths


def strategy_dict(name: str = "cap", **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "source": "tests:not_loaded",
        "symbols": list(COINS),
        "capital": 1000.0,
        "sizing": {"stake": 100.0},
    }
    entry.update(overrides)
    return entry


def run_config(
    state: Path,
    strategies: Sequence[dict[str, Any]] | None = None,
    *,
    initial_cash: float | None = None,
    **account: Any,
) -> RunConfig:
    strategies = list(strategies or [strategy_dict()])
    cash = initial_cash if initial_cash is not None else sum(s["capital"] for s in strategies)
    return config_from_dict(
        {
            "state": str(state),
            "account": {"initial_cash": cash, **account},
            "strategies": strategies,
        }
    )


def rule_strategy(name: str = "cap", rule: CapitulationRule = RULE) -> RuleStrategy:
    return RuleStrategy(rule, name=name, workers=1)


def trades_core(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(
        "symbol",
        "entry_time",
        "exit_time",
        "exit_reason",
        "entry_ref",
        "entry_price",
        "exit_price",
        "quantity",
        "stake",
        "pnl",
    )


RULE_PLUGIN_SOURCE = textwrap.dedent(
    """
    from datetime import timedelta

    from crypto_moonshot.broker.base import ExitPlan
    from crypto_moonshot.live.plugin import RuleStrategy
    from crypto_moonshot.models import Interval
    from crypto_moonshot.strategy.capitulation import CapitulationRule


    def make() -> RuleStrategy:
        rule = CapitulationRule(
            "cap",
            min_qv_base=1.0,
            exit=ExitPlan(take_profit=0.03, stop=0.04, max_hold=timedelta(hours=12)),
            max_ret_4h=-0.03,
            min_btc_vol_mult=3.0,
            spike_mult=4.0,
            interval=Interval.H1,
        )
        return RuleStrategy(rule, workers=1)
    """
)
