"""E39: deflated Sharpe ratio of the LQM candidate. The variance of Sharpe ratios across trials
is estimated from a family of related books (sleeve combinations x slots), and the number of
trials is taken at 100 / 1000 / 4000 (this study ran roughly 4000 configurations, most of them
strongly correlated)."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from dsr import daily_returns, dsr
from e25_combined import book, sleeve_listing, sleeve_listing_weak, sleeve_long, sleeve_pump, sleeve_pump_regime

Y = (2022, 2023, 2024, 2025, 2026)
S = {"L": sleeve_long(), "W": sleeve_long(q=0.99, tp=0.15), "P": sleeve_pump(), "Q": sleeve_pump_regime(), "N": sleeve_listing(), "M": sleeve_listing_weak()}
srs = []
cand = None
for combo, slots in itertools.product(("L", "W", "P", "Q", "N", "M", "LP", "LQ", "LN", "LM", "WQ", "WM", "QM", "LQM", "WQM", "LPN", "LPM"), (8, 10, 15, 20)):
    tr: list = []
    book(pl.concat([S[k] for k in combo], how="vertical_relaxed"), slots, trades_out=tr)
    r = daily_returns(pl.DataFrame(tr), Y)
    sr = r.mean() / r.std(ddof=1)
    srs.append(sr)
    if combo == "LQM" and slots == 10:
        cand = r
var_sr = float(np.var(srs, ddof=1))
print(f"{len(srs)} related books; annualised Sharpe mean {np.mean(srs) * np.sqrt(365):.2f}, sd {np.std(srs) * np.sqrt(365):.2f}")
assert cand is not None
for n in (100, 1000, 4000):
    print(n, {k: round(v, 3) for k, v in dsr(cand, n, var_sr).items()})
