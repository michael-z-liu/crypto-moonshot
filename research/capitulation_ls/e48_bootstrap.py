"""E48: block bootstrap (7-day blocks) of the LMC book's daily returns: distribution of a year's
return and of the average trade, to show how wide the plausible range is."""

from __future__ import annotations

import numpy as np
import polars as pl
from dsr import daily_returns
from e25_combined import book, sleeve_crash_short, sleeve_listing_weak, sleeve_long

Y = (2022, 2023, 2024, 2025, 2026)
tr: list = []
b = book(pl.concat([sleeve_long(), sleeve_listing_weak(), sleeve_crash_short()], how="vertical_relaxed"), 10, trades_out=tr)
r = daily_returns(pl.DataFrame(tr), Y)
rng = np.random.default_rng(0)
blk, n = 7, 365
yrs = []
for _ in range(5000):
    starts = rng.integers(0, len(r) - blk, size=n // blk + 1)
    path = np.concatenate([r[s : s + blk] for s in starts])[:n]
    yrs.append(np.prod(1 + path) - 1)
yrs = np.array(yrs)
print("bootstrapped one-year return: p5 {:+.0%}, p25 {:+.0%}, median {:+.0%}, p75 {:+.0%}".format(*np.quantile(yrs, [0.05, 0.25, 0.5, 0.75])))
print(f"P(year < 0) = {np.mean(yrs < 0):.1%}, P(year >= +50%) = {np.mean(yrs >= 0.5):.1%}")
nets = np.array([t["pnl"] for t in tr])  # $ per trade; per-trade % from the book table
ex25 = r[np.r_[0:365 * 3, 365 * 4 : len(r)]]
yrs2 = []
for _ in range(5000):
    starts = rng.integers(0, len(ex25) - blk, size=n // blk + 1)
    yrs2.append(np.prod(1 + np.concatenate([ex25[s : s + blk] for s in starts])[:n]) - 1)
yrs2 = np.array(yrs2)
print("without 2025: p5 {:+.0%}, median {:+.0%}, P(>= +50%) {:.1%}".format(np.quantile(yrs2, 0.05), np.median(yrs2), np.mean(yrs2 >= 0.5)))
