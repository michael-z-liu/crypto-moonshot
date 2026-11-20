"""Example strategy plugin: a moving-average cross.

Enters a symbol when its fast moving average of the close crosses above the slow one on the
candle that just closed. Uses only polars, so the same file runs in the daemon's process and as
an external provider (see `docs/strategy-plugins.md`):

    source = "examples/strategies/ma_cross.py:MaCross"          # in a run config (in process)
    moonshot provider run examples/strategies/ma_cross.py:MaCross --name cross --url ...

A plugin is any object with `name`, `interval`, `warmup`, `context_symbols`, `exit` and
`on_candles`; a class whose `__init__` takes the config's `params` is the usual shape.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Final

import polars as pl

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.models import Interval

_SIGNALS: Final = {"symbol": pl.String, "priority": pl.Float64}


class MaCross:
    """Enter when the `fast` moving average crosses above the `slow` one.

    Params (all optional): `interval` ("1h"), `fast` (12) and `slow` (48) candles, and the exit
    plan: `take_profit` (0.05), `stop` (0.03) and `max_hold_hours` (48).
    """

    def __init__(
        self,
        interval: str = "1h",
        fast: int = 12,
        slow: int = 48,
        take_profit: float = 0.05,
        stop: float = 0.03,
        max_hold_hours: int = 48,
    ) -> None:
        if not 1 <= fast < slow:
            msg = f"need 1 <= fast < slow, got fast={fast} slow={slow}"
            raise ValueError(msg)
        self.name = "ma_cross"  # the platform sets this from the config
        self.interval = Interval(interval)
        self.fast = fast
        self.slow = slow
        self.warmup = slow + 2  # candles of history per symbol: two slow averages
        self.context_symbols: tuple[str, ...] = ()  # no extra symbols (BTCUSDT, say) are read
        self.exit: ExitPlan | None = ExitPlan(
            take_profit=take_profit, stop=stop, max_hold=timedelta(hours=max_hold_hours)
        )

    def on_candles(
        self,
        windows: Mapping[str, pl.DataFrame],
        time: datetime,
        context: Mapping[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Signals for the candle that opened at `time`. `windows` maps each traded symbol to
        its last `warmup` closed candles (oldest first, ending at `time`)."""
        symbols: list[str] = []
        priorities: list[float] = []
        for symbol, window in windows.items():
            if window.height < self.slow + 1 or window["open_time"][-1] != time:
                continue  # not enough history, or no candle at `time` for this symbol
            close = window["close"]
            fast = close.rolling_mean(self.fast)
            slow = close.rolling_mean(self.slow)
            crossed = fast[-1] > slow[-1] and fast[-2] <= slow[-2]
            if crossed:
                symbols.append(symbol)
                priorities.append(-(fast[-1] / slow[-1] - 1.0))  # the widest gap first
        return pl.DataFrame({"symbol": symbols, "priority": priorities}, schema=_SIGNALS)
