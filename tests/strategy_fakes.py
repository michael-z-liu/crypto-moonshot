"""A small synthetic store where the capitulation rules fire at known candles.

Every series runs from `T0` for `N` 5m candles at a flat price of 100. At candle `SPIKE` (past
the 7-day warm-up) BTC's volume jumps 10x, and the coins:

- `DIPUSDT`: falls 15% over the previous 4 hours on 20x volume (quote volume base 100k), then
  trades at 90: an F2 and R2 signal;
- `THINUSDT`: the same fall and spike on a thin book (quote volume base ~10k): R2 only;
- `SHALLOWUSDT`: falls only 5%: no signal.

At `LONE` DIPUSDT crashes and spikes again, but BTC is quiet: no signal.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from store_fakes import kline_frame, write_series

from crypto_moonshot.models import Interval

M5 = timedelta(minutes=5)
T0 = datetime(2024, 1, 1, tzinfo=UTC)
DAY = 288
SPIKE = 7 * DAY + DAY + 50
LONE = SPIKE + DAY
N = LONE + 100


def at(index: int) -> datetime:
    return T0 + index * M5


def _coin(base_volume: float, low: float, spikes: tuple[int, ...]) -> dict[str, np.ndarray]:
    close = np.full(N, 100.0)
    volume = np.full(N, base_volume)
    for s in spikes:
        close[s - 47 : s + 1] = np.linspace(100.0, low, 48)  # ret_4h = low / 100 - 1
        close[s + 1 : s + 12 * 6] = 90.0
        close[s + 12 * 6 :] = 100.0
        volume[s] = 20 * base_volume
    return {"close": close, "volume": volume}


def write_capitulation_store(root: Path) -> None:
    times = [at(k) for k in range(N)]
    btc_volume = np.full(N, 10.0)
    btc_volume[SPIKE] = 100.0
    series = {
        "BTCUSDT": {"close": np.full(N, 30_000.0), "volume": btc_volume},
        "DIPUSDT": _coin(1000.0, 85.0, (SPIKE, LONE)),
        "THINUSDT": _coin(100.0, 88.0, (SPIKE,)),
        "SHALLOWUSDT": _coin(1000.0, 95.0, (SPIKE,)),
    }
    for symbol, cols in series.items():
        close = cols["close"]
        open_ = np.concatenate(([close[0]], close[:-1]))
        frame = kline_frame(
            times,
            close,
            cols["volume"],
            open_=open_,
            high=np.maximum(open_, close) * 1.001,
            low=np.minimum(open_, close) * 0.999,
        )
        write_series(root, symbol, Interval.M5, frame)
