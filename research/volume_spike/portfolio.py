"""Slot-limited portfolio simulation over the crash subset (24-hour forward paths)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl
from common import CACHE, split_of
from evalx import COST


@dataclass(frozen=True)
class Exit:
    tp: float | None = None  # take profit, e.g. 0.05 = +5%
    sl: float | None = None  # stop loss, e.g. 0.10 = -10%
    bars: int = 48  # time stop, in 5m bars (max 288)
    trail: float | None = None  # trailing stop from the highest high since entry


def load_long() -> tuple[pl.DataFrame, np.ndarray]:
    events = pl.read_parquet(CACHE / "long.parquet").with_columns(split_of())
    return events, np.load(CACHE / "long_fwd.npy")


def exit_path(fwd: np.ndarray, ex: Exit, cost: float = COST) -> tuple[np.ndarray, np.ndarray]:
    """Net return and exit bar (0-based, within the forward window) for each event.

    Same conventions as evalx.triple_barrier: a stop is assumed to fill before a take profit
    touched in the same bar, stops and targets fill at their level, a time stop at the close.
    The trailing stop is set from the highest high of earlier bars, so it is never lifted by the
    same bar that triggers it.
    """
    hi = fwd[:, 0, : ex.bars]
    lo = fwd[:, 1, : ex.bars]
    n, bars = hi.shape
    never = bars + 1

    def first(hit: np.ndarray) -> np.ndarray:
        return np.where(hit.any(1), hit.argmax(1), never)

    t_tp = first(hi >= ex.tp) if ex.tp is not None else np.full(n, never)
    t_sl = first(lo <= -ex.sl) if ex.sl is not None else np.full(n, never)
    t_tr = np.full(n, never)
    tr_level = np.zeros(n)
    if ex.trail is not None:
        peak = np.maximum.accumulate(np.concatenate([np.zeros((n, 1)), hi[:, :-1]], 1), axis=1)
        level = (1 + peak) * (1 - ex.trail) - 1
        t_tr = first(lo <= level)
        tr_level = level[np.arange(n), np.minimum(t_tr, bars - 1)]
    ret = fwd[:, 2, bars - 1].astype(np.float64).copy()
    t_exit = np.full(n, bars - 1)
    # the earliest of the three; ties go to the stop-like exits
    t_stop = np.minimum(t_sl, t_tr)
    stop_ret = np.where(t_sl <= t_tr, -(ex.sl or 0.0), tr_level)
    stopped = (t_stop < never) & (t_stop <= t_tp)
    took = (t_tp < never) & (t_tp < t_stop)
    ret[stopped] = stop_ret[stopped]
    t_exit[stopped] = t_stop[stopped]
    if ex.tp is not None:
        ret[took] = ex.tp
        t_exit[took] = t_tp[took]
    return ret - cost, t_exit


@dataclass
class SimResult:
    trades: pl.DataFrame
    equity: pl.DataFrame  # daily, end of day

    def stats(self) -> dict[str, float]:
        tr, eq = self.trades, self.equity
        if tr.height == 0:
            return {"trades": 0}
        e = eq["equity"].to_numpy()
        daily = np.diff(np.log(e))
        years = max(eq.height / 365.25, 1e-9)
        peak = np.maximum.accumulate(e)
        return {
            "trades": tr.height,
            "trades_per_week": tr.height / (years * 52.18),
            "mean_bp": tr["ret"].mean() * 1e4,
            "win": (tr["ret"] > 0).mean(),
            "total_x": e[-1] / e[0],
            "cagr": (e[-1] / e[0]) ** (1 / years) - 1,
            "max_dd": float((e / peak - 1).min()),
            "sharpe": float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0,
            "days_in_market": tr["day"].n_unique(),
        }


def simulate(
    signals: pl.DataFrame,
    fwd: np.ndarray,
    ex: Exit,
    slots: int = 5,
    cost: float = COST,
    priority: str = "ret_4h",
    start: object = None,
    end: object = None,
) -> SimResult:
    """Take signals in time order while a slot is free and the symbol is not already held.

    Each trade gets 1/slots of the equity booked at entry (closed P&L plus open cost basis).
    Signals at the same bar are taken in ascending order of `priority` (deepest drop first).
    """
    ret, t_exit = exit_path(fwd, ex, cost)
    sig = signals.with_columns(
        ret=pl.Series(ret), exit_bar=pl.Series(t_exit), _j=pl.int_range(pl.len())
    ).sort("open_time", priority)
    equity = 1.0
    open_pos: list[tuple[object, str, float, float]] = []  # (exit time, symbol, stake, ret)
    rows = []
    for r in sig.iter_rows(named=True):
        t = r["open_time"]
        entry_t = t + timedelta(minutes=5)
        # close positions that exited before this entry
        still = []
        for pos in open_pos:
            if pos[0] <= entry_t:
                equity += pos[2] * pos[3]
            else:
                still.append(pos)
        open_pos = still
        if len(open_pos) >= slots or any(p[1] == r["symbol"] for p in open_pos):
            continue
        stake = equity / slots
        exit_t = entry_t + timedelta(minutes=5 * (int(r["exit_bar"]) + 1))
        open_pos.append((exit_t, r["symbol"], stake, r["ret"]))
        rows.append(
            {
                "symbol": r["symbol"], "entry_time": entry_t, "exit_time": exit_t,
                "ret": r["ret"], "stake": stake, "day": entry_t.date(),
            }
        )  # fmt: skip
    trades = pl.DataFrame(rows) if rows else pl.DataFrame()
    if trades.height == 0:
        return SimResult(trades, pl.DataFrame({"day": [], "equity": []}))
    pnl = trades.with_columns(pnl=pl.col("stake") * pl.col("ret"), d=pl.col("exit_time").dt.date())
    first = start or sig["open_time"].min().date()
    last = end or pnl["d"].max()
    days = pl.DataFrame({"day": pl.date_range(first, last, "1d", eager=True)})
    daily = days.join(pnl.group_by("d").agg(pl.col("pnl").sum()), left_on="day", right_on="d",
                      how="left").with_columns(pl.col("pnl").fill_null(0.0))  # fmt: skip
    eq = daily.with_columns(equity=1.0 + pl.col("pnl").cum_sum()).select("day", "equity")
    return SimResult(trades, eq)
