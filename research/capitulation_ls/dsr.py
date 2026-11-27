"""Deflated Sharpe ratio (Bailey & Lopez de Prado) for a book's daily returns, given the number
of trials and the variance of the trials' Sharpe ratios. Daily returns come from realised P&L by
exit day over equity at the start of that day (each year a fresh $10k book)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.stats import kurtosis, norm, skew


def daily_returns(trades: pl.DataFrame, years: tuple[int, ...]) -> np.ndarray:
    """trades: t_out (epoch s), pnl ($), year; returns one value per calendar day in `years`."""
    out = []
    for y in years:
        g = trades.filter(pl.col("year") == y)
        days = pl.date_range(pl.date(y, 1, 1), pl.date(y, 12, 31), "1d", eager=True)
        if y == 2026:
            days = pl.date_range(pl.date(y, 1, 1), pl.date(2026, 11, 1), "1d", eager=True)
        pnl = (
            g.with_columns(d=pl.from_epoch("t_out").dt.date())
            .group_by("d")
            .agg(p=pl.col("pnl").sum())
        )
        s = pl.DataFrame({"d": days}).join(pnl, on="d", how="left").fill_null(0.0)
        eq = 10_000 + s["p"].cum_sum().shift(1, fill_value=0.0)
        out.append((s["p"] / eq).to_numpy())
    return np.concatenate(out)


def dsr(r: np.ndarray, n_trials: int, var_sr: float) -> dict[str, float]:
    sr = r.mean() / r.std(ddof=1)  # per day
    t = len(r)
    g, k = skew(r), kurtosis(r, fisher=False)
    emc = 0.5772156649
    sr0 = math.sqrt(var_sr) * (
        (1 - emc) * norm.ppf(1 - 1 / n_trials) + emc * norm.ppf(1 - 1 / (n_trials * math.e))
    )
    z = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(1 - g * sr + (k - 1) / 4 * sr**2)
    return {
        "sr_annual": sr * math.sqrt(365),
        "sr0_annual": sr0 * math.sqrt(365),
        "skew": g,
        "kurt": k,
        "dsr": float(norm.cdf(z)),
        "days": t,
    }
