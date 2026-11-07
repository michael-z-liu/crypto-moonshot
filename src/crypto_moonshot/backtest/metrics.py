"""Backtest metrics, overall and per calendar year.

Two families, measured differently on purpose:

- **Trade metrics** come from the trade list and are grouped by the year of the *entry* (as the
  research reports them): trades, mean net return per trade in basis points, win rate, P&L in
  USDT (the sum of trade P&L), worst trade in basis points and average holding time in hours.
- **Equity metrics** come from the mark-to-market equity curve: total return measured from the
  initial capital (so the first day's P&L counts), CAGR over the curve's span, max drawdown
  (the initial capital counts as a peak), Sharpe on the curve's period returns (daily by
  default) annualised with 365 days, time in market (share of the curve's span with at least
  one position open), and each calendar year's return from the curve.

Undefined values (e.g. a mean over no trades, a Sharpe with fewer than two returns) are NaN.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import nan, sqrt

import numpy as np
import polars as pl

from crypto_moonshot.broker.candles import to_us

_YEAR = timedelta(days=365.25)
_ANNUAL = timedelta(days=365)

TRADE_METRICS = ("trades", "mean_bp", "win_rate", "pnl", "worst_bp", "avg_hold_hours")
EQUITY_METRICS = (
    "final_equity",
    "total_return",
    "cagr",
    "max_drawdown",
    "sharpe",
    "time_in_market",
)


def _trade_aggregates() -> list[pl.Expr]:
    hold = (pl.col("exit_time") - pl.col("entry_time")).dt.total_microseconds() / 3.6e9
    return [
        pl.len().cast(pl.Float64).alias("trades"),
        (pl.col("net_return").mean() * 1e4).alias("mean_bp"),
        (pl.col("net_return") > 0).mean().alias("win_rate"),
        pl.col("pnl").sum().alias("pnl"),
        (pl.col("net_return").min() * 1e4).alias("worst_bp"),
        hold.mean().alias("avg_hold_hours"),
    ]


def _floats(row: dict[str, object]) -> dict[str, float]:
    return {k: nan if v is None else float(v) for k, v in row.items()}  # type: ignore[arg-type]


def trade_metrics(trades: pl.DataFrame) -> dict[str, float]:
    """Trade metrics over the whole trade list."""
    if trades.height == 0:
        return {"trades": 0.0, "mean_bp": nan, "win_rate": nan, "pnl": 0.0} | {
            "worst_bp": nan,
            "avg_hold_hours": nan,
        }
    return _floats(trades.select(_trade_aggregates()).row(0, named=True))


def _arrays(equity: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    times = equity["time"].dt.epoch("us").to_numpy().astype(np.int64)
    return times, equity["equity"].to_numpy().astype(np.float64)


def time_in_market(trades: pl.DataFrame, start: datetime, end: datetime) -> float:
    """Share of `[start, end)` with at least one position open (`entry <= t < exit`)."""
    span = to_us(end) - to_us(start)
    if trades.height == 0 or span <= 0:
        return 0.0
    order = trades.select(
        pl.col("entry_time").dt.epoch("us").alias("a"),
        pl.col("exit_time").dt.epoch("us").alias("b"),
    ).sort("a")
    a = np.clip(order["a"].to_numpy(), to_us(start), to_us(end))
    b = np.clip(order["b"].to_numpy(), to_us(start), to_us(end))
    # Merge overlapping intervals: a new block starts where an entry follows every earlier exit.
    reach = np.maximum.accumulate(b)
    new = np.concatenate(([True], a[1:] > reach[:-1]))
    block = np.cumsum(new) - 1
    starts = a[new]
    ends = np.zeros(len(starts), dtype=np.int64)
    np.maximum.at(ends, block, b)
    return float((ends - starts).sum() / span)


def equity_metrics(
    equity: pl.DataFrame,
    trades: pl.DataFrame,
    initial_capital: float,
    mark_every: timedelta = timedelta(days=1),
) -> dict[str, float]:
    """Equity metrics from the mark-to-market curve (see the module docstring)."""
    if equity.height == 0:
        return {
            "final_equity": initial_capital,
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": nan,
            "time_in_market": 0.0,
        }
    _, e = _arrays(equity)
    final = float(e[-1])
    total = final / initial_capital - 1
    first, last = equity["time"][0], equity["time"][-1]
    years = (last - first) / _YEAR
    cagr = (max(final, 0.0) / initial_capital) ** (1 / years) - 1 if years > 0 else nan
    peak = np.maximum.accumulate(np.concatenate(([initial_capital], e)))[1:]
    max_dd = float((e / peak - 1).min())
    returns = e[1:] / e[:-1] - 1
    sharpe = nan
    if len(returns) >= 2 and returns.std(ddof=1) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=1) * sqrt(_ANNUAL / mark_every))
    return {
        "final_equity": final,
        "total_return": total,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "time_in_market": time_in_market(trades, first, last),
    }


def _equity_at(times: np.ndarray, e: np.ndarray, t: datetime) -> float | None:
    """Equity of the last curve point at or before `t` (None if the curve starts later)."""
    i = int(np.searchsorted(times, to_us(t), side="right")) - 1
    return None if i < 0 else float(e[i])


def yearly_metrics(
    trades: pl.DataFrame, equity: pl.DataFrame, initial_capital: float
) -> pl.DataFrame:
    """One row per calendar year: trade metrics by entry year, and the year's return from the
    equity curve (equity at the year's end over equity at its start, the initial capital for
    the curve's first year)."""
    years: set[int] = set()
    by_year = pl.DataFrame()
    if trades.height:
        by_year = (
            trades.group_by(pl.col("entry_time").dt.year().alias("year"))
            .agg(_trade_aggregates())
            .with_columns(pl.col("year").cast(pl.Int32))
        )
        years.update(by_year["year"].to_list())
    times, e = _arrays(equity)
    if equity.height:
        years.update(range(equity["time"][0].year, equity["time"][-1].year + 1))
    rows = []
    for year in sorted(years):
        start = _equity_at(times, e, datetime(year, 1, 1, tzinfo=UTC))
        end = _equity_at(times, e, datetime(year + 1, 1, 1, tzinfo=UTC))
        start = initial_capital if start is None else start
        rows.append({"year": year, "return": nan if end is None else end / start - 1})
    out = pl.DataFrame(rows, schema={"year": pl.Int32, "return": pl.Float64})
    if by_year.height:
        out = out.join(by_year, on="year", how="left")
    else:
        out = out.with_columns(pl.lit(None, pl.Float64).alias(m) for m in TRADE_METRICS)
    return out.with_columns(pl.col("trades").fill_null(0.0), pl.col("pnl").fill_null(0.0)).select(
        "year", *TRADE_METRICS, "return"
    )


def summarize(
    trades: pl.DataFrame,
    equity: pl.DataFrame,
    initial_capital: float,
    mark_every: timedelta = timedelta(days=1),
) -> tuple[dict[str, float], pl.DataFrame]:
    """Overall metrics (trade and equity) and the per-year table."""
    overall = trade_metrics(trades) | equity_metrics(equity, trades, initial_capital, mark_every)
    return overall, yearly_metrics(trades, equity, initial_capital)
