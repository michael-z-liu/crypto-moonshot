"""M7: walk-forward selection of capitulation parameters, measured per crash day.

Every six months, pick the rule with the best mean per-crash-day return over the previous 24
months (at least 15 crash days), then trade it for the next six months. Each crash day is one
bet: an equal-weight basket of every qualifying coin (one trade per coin per holding period).
The equity line commits half of equity to each crash day's basket.
"""

from __future__ import annotations

import itertools
import sys
from datetime import UTC, datetime

import numpy as np
import polars as pl
from common import RESULTS
from evalx import COST, load_events
from m3_rules import non_overlapping

DROPS = [0.08, 0.10, 0.12, 0.15, 0.20]
LIQS = [5_000, 20_000, 50_000]
BTC_VM = [0, 4, 6, 10]
HOLDS = [12, 48]
LOOKBACK_MONTHS = 24
MIN_DAYS = 15
SHARE = 0.5
SCORE = "mean"  # or "t" (t-statistic: prefers rules that trade more often)


def day_returns(ev: pl.DataFrame, fwd: np.ndarray) -> pl.DataFrame:
    """Per crash day basket return for every rule in the grid: (rule, day, r, n)."""
    out = []
    for drop, liq, vm, hold in itertools.product(DROPS, LIQS, BTC_VM, HOLDS):
        cond = (
            pl.col("is_spike") & (pl.col("ret_4h") <= -drop) & (pl.col("qv_base") >= liq)
            & (pl.col("btc_vol_mult") >= vm)
        )  # fmt: skip
        idx = np.flatnonzero(ev.select(cond.fill_null(False).alias("m"))["m"].to_numpy())
        d = (
            ev[idx]
            .select("symbol", "open_time")
            .with_columns(r=pl.Series(np.asarray(fwd[idx, 2, hold - 1])) - COST)
        )
        keep = non_overlapping(d["symbol"].to_numpy(), d["open_time"].to_numpy(), hold)
        days = (
            d.filter(pl.Series(keep))
            .group_by(pl.col("open_time").dt.date().alias("day"))
            .agg(pl.col("r").mean(), pl.len().alias("n"))
            .with_columns(rule=pl.lit(f"drop{drop:.2f}_liq{liq}_btcvm{vm}_hold{hold}"))
        )
        out.append(days)
    return pl.concat(out)


def main() -> None:
    global SCORE, MIN_DAYS
    if len(sys.argv) > 1:
        SCORE, MIN_DAYS = sys.argv[1], int(sys.argv[2])
    ev, fwd = load_events()
    days = day_returns(ev, fwd)
    starts = [datetime(y, m, 1, tzinfo=UTC).date() for y in range(2022, 2027) for m in (1, 7)]
    picks, oos = [], []
    for s in starts:
        lb = s.replace(year=s.year - LOOKBACK_MONTHS // 12)
        e = s.replace(year=s.year + 1, month=1) if s.month == 7 else s.replace(month=7)
        hist = days.filter((pl.col("day") >= lb) & (pl.col("day") < s))
        score = (
            hist.group_by("rule")
            .agg(
                pl.col("r").mean().alias("mean"),
                pl.col("r").std().alias("sd"),
                pl.len().alias("ndays"),
            )
            .filter(pl.col("ndays") >= MIN_DAYS)
            .with_columns(t=pl.col("mean") / pl.col("sd") * pl.col("ndays").sqrt())
            .sort(SCORE, descending=True)
        )
        best = score.row(0, named=True)
        nxt = days.filter(
            (pl.col("rule") == best["rule"]) & (pl.col("day") >= s) & (pl.col("day") < e)
        )
        picks.append({"start": s, "rule": best["rule"], "train_bp": best["mean"] * 1e4,
                      "train_days": best["ndays"], "oos_days": nxt.height,
                      "oos_bp": (nxt["r"].mean() or 0) * 1e4,
                      "oos_win_days": (nxt["r"] > 0).mean() if nxt.height else None})  # fmt: skip
        oos.append(nxt)
    p = pl.DataFrame(picks)
    o = pl.concat(oos).sort("day")
    eq = np.cumprod(1 + SHARE * o["r"].to_numpy())
    peak = np.maximum.accumulate(eq)
    p.write_csv(RESULTS / f"m7_walkforward_{SCORE}_{MIN_DAYS}.csv")
    with pl.Config(tbl_rows=20, tbl_width_chars=200, fmt_str_lengths=50, float_precision=1):
        print(p)
    yrs = (o["day"].max() - o["day"].min()).days / 365.25
    print(
        f"OOS 2022-01..: {o.height} crash days, mean {o['r'].mean() * 1e4:+.0f} bp/day, "
        f"win days {(o['r'] > 0).mean():.2f}, equity x{eq[-1]:.2f} "
        f"(CAGR {eq[-1] ** (1 / yrs) - 1:+.1%}), max DD {(eq / peak - 1).min():.1%}"
    )
    for y in range(2022, 2027):
        oy = o.filter(pl.col("day").dt.year() == y)
        print(
            y, oy.height, f"{oy['r'].mean() * 1e4:+.0f} bp/day", f"win {(oy['r'] > 0).mean():.2f}"
        )


if __name__ == "__main__":
    main()
