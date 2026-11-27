"""E29: capitulation longs on major coins with leverage (perp-like costs 0.3%, liquidation on the
path). Majors wick less, so 2-3x may survive flash crashes that wipe out levered alt longs."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, fmt_years, simulate_yearly

MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
          "LTCUSDT", "TRXUSDT", "BCHUSDT", "NEARUSDT", "MATICUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT", "FILUSDT", "APTUSDT"}  # fmt: skip
m = meta()
yr, sym = m["year"].to_numpy(), m["symbol"].to_numpy()
r4, btc4 = m["ret_4h"].to_numpy(), m["btc_ret_4h"].to_numpy()
pct = scores("day5")
maj = np.isin(sym, list(MAJORS))
Y = (2022, 2023, 2024, 2025, 2026)
sets = {
    "maj r4<=-8% btc4<=-3%": maj & (r4 <= -0.08) & (btc4 <= -0.03),
    "maj r4<=-10%": maj & (r4 <= -0.10),
    "maj day5>=0.98": maj & (pct >= 0.98),
    "maj day5>=0.95": maj & (pct >= 0.95),
}
for name, mask in sets.items():
    idx = np.flatnonzero(mask & (yr >= 2022))
    c = pl.DataFrame({"row": idx, "side": 1, "prio": -r4[idx]})
    for hold, tp, lev in itertools.product((1440, 4320), (0.05, 0.1, None), (1, 2, 3)):
        b = simulate_yearly(c, Rule(hold_min=hold, tp=tp, slots=5, cost=0.003, lev=lev, liq_frac=None), years=Y)
        if b.trades.height == 0:
            continue
        liq = int((b.trades["kind"] == 3).sum())
        print(f"{name:22s} hold {hold // 60:>2}h tp {str(tp):4s} lev {lev}: {b.summary['bp']:+5.0f}bp n={b.summary['trades']:>3} liq={liq} {fmt_years(b, Y)}")
