"""M2: explainable factors for every panel entry, all years (no outcomes).

    uv run --group research python build_factors.py [--workers 8] [--doc-only]

Writes `cache/factors.parquet` (keys `symbol`, `d_us`, then one Float64 column per factor, null
where the history is too short) and the factor dictionary `results/factors.md` from `FACTORS`.

Every factor uses only candles that closed by `D`, i.e. 5m candles with open time < `D`:

- `daily_bars` turns a coin's 5m candles into dense UTC-day bars (open, high, low, close, quote
  volume, trade count, taker-buy quote volume). A day without candles is a flat bar at the last
  close with zero volume. The bar of day `D - 1` is the last complete one before `D`, and its
  close is `P` (`build` asserts this against the panel).
- `coin_series` computes every coin factor at every bar index `i` with trailing windows only:
  the value at `i` is the factor of the entry at `D = day(i) + 1`. `market_series` does the same
  for BTC and ETH. `breadth` aggregates coin values across the panel's coins on each day, and
  `market_volume` sums every universe coin's daily quote volume.
- These functions take candle frames (or bars built from them) as input, so `check_factors.py`
  recomputes them from candles cut at `D` and from candles changed after `D`.

The "habit" and "moonshot share" factors come from past price windows that had fully ended by
`D` (see `habit_hits`), never from the labels files.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import polars as pl
from common import DAY, F64, FACTORS, I64, PANEL, RESULTS, read_5m, universe
from numpy.lib.stride_tricks import sliding_window_view

BTC = "BTCUSDT"
ETH = "ETHUSDT"
MOON = 0.30  # a "moonshot": the high reaches +30% over the reference close ...
MOON_DAYS = 30  # ... within the next 30 days
HABIT_SPAN = 365  # habit: windows starting in the last 365 days that had ended by D
HABIT_MIN = 120  # fewest ended windows for a habit value


@dataclass(frozen=True)
class Factor:
    name: str
    family: str
    kind: str  # market | coin
    formula: str
    lookback: str


def _f(name: str, family: str, kind: str, formula: str, lookback: str) -> Factor:
    return Factor(name, family, kind, formula, lookback)


M, C = "market", "coin"
REG, MOM, VOL, FLOW, REL, FLAG = (
    "market regime",
    "momentum and position",
    "volatility and compression",
    "volume and flow",
    "relative strength, size, age",
    "reference flag",
)

# `c` is the daily close (c[0] the last complete day, D - 1, so c[0] = P), `h`/`l` the daily
# high/low, `qv` quote volume, `tb` taker-buy quote volume, `tc` trade count, `lr` the daily log
# return. "max h 30" is the highest daily high of the last 30 days (D-30 .. D-1).
FACTORS_LIST: list[Factor] = [
    _f("m_btc_ret1", REG, M, "BTC c[0] / c[1] - 1", "1 d"),
    _f("m_btc_ret7", REG, M, "BTC c[0] / c[7] - 1", "7 d"),
    _f("m_btc_ret30", REG, M, "BTC c[0] / c[30] - 1", "30 d"),
    _f("m_btc_ret90", REG, M, "BTC c[0] / c[90] - 1", "90 d"),
    _f("m_btc_sma50", REG, M, "BTC c[0] / mean(c, 50 d) - 1", "50 d"),
    _f("m_btc_sma200", REG, M, "BTC c[0] / mean(c, 200 d) - 1", "200 d"),
    _f("m_btc_dd90", REG, M, "BTC c[0] / max h 90 - 1", "90 d"),
    _f("m_btc_vol7", REG, M, "BTC std of lr over 7 d (daily, not annualised)", "8 d"),
    _f("m_btc_vol30", REG, M, "BTC std of lr over 30 d", "31 d"),
    _f("m_btc_vol_30_180", REG, M, "BTC std lr 30 d / std lr 180 d", "181 d"),
    _f("m_ethbtc_ret30", REG, M, "(ETH c[0] / c[30]) / (BTC c[0] / c[30]) - 1", "30 d"),
    _f("m_breadth20", REG, M, "share of the day's panel coins with c_sma20 > 0", "20 d"),
    _f("m_breadth50", REG, M, "share of the day's panel coins with c_sma50 > 0", "50 d"),
    _f("m_breadth200", REG, M, "share of the day's panel coins with c_sma200 > 0", "200 d"),
    _f("m_med_ret7", REG, M, "median c_ret7 over the day's panel coins", "7 d"),
    _f("m_disp_ret7", REG, M, "std of log(1 + c_ret7) over the day's panel coins", "7 d"),
    _f("m_med_ret30", REG, M, "median c_ret30 over the day's panel coins", "30 d"),
    _f("m_disp_ret30", REG, M, "std of log(1 + c_ret30) over the day's panel coins", "30 d"),
    _f(
        "m_moon30",
        REG,
        M,
        "share of the day's panel coins whose max h 30 >= 1.3 c[30] (last ended window)",
        "31 d",
    ),
    _f("m_qv_7_90", REG, M, "all universe coins' summed qv: mean 7 d / mean 90 d", "90 d"),
    _f("c_ret1", MOM, C, "c[0] / c[1] - 1", "1 d"),
    _f("c_ret3", MOM, C, "c[0] / c[3] - 1", "3 d"),
    _f("c_ret7", MOM, C, "c[0] / c[7] - 1", "7 d"),
    _f("c_ret14", MOM, C, "c[0] / c[14] - 1", "14 d"),
    _f("c_ret30", MOM, C, "c[0] / c[30] - 1", "30 d"),
    _f("c_ret90", MOM, C, "c[0] / c[90] - 1", "90 d"),
    _f("c_dd30", MOM, C, "c[0] / max h 30 - 1 (drawdown from the high)", "30 d"),
    _f("c_dd90", MOM, C, "c[0] / max h 90 - 1", "90 d"),
    _f("c_ru30", MOM, C, "c[0] / min l 30 - 1 (run-up from the low)", "30 d"),
    _f("c_ru90", MOM, C, "c[0] / min l 90 - 1", "90 d"),
    _f("c_pos30", MOM, C, "(c[0] - min l 30) / (max h 30 - min l 30)", "30 d"),
    _f("c_sma20", MOM, C, "c[0] / mean(c, 20 d) - 1", "20 d"),
    _f("c_sma50", MOM, C, "c[0] / mean(c, 50 d) - 1", "50 d"),
    _f("c_sma100", MOM, C, "c[0] / mean(c, 100 d) - 1", "100 d"),
    _f("c_sma200", MOM, C, "c[0] / mean(c, 200 d) - 1", "200 d"),
    _f("c_sma50_slope", MOM, C, "SMA50 today / SMA50 10 days earlier - 1", "60 d"),
    _f("c_up10", MOM, C, "number of days with c[k] > c[k+1] among the last 10", "11 d"),
    _f("c_ret1_atr", MOM, C, "(c[0] - c[1]) / ATR14", "15 d"),
    _f(
        "c_atr14",
        VOL,
        C,
        "mean true range over 14 d / c[0] (TR = max(h-l, |h-c[1]|, |l-c[1]|))",
        "15 d",
    ),
    _f("c_vol7", VOL, C, "std of lr over 7 d", "8 d"),
    _f("c_vol30", VOL, C, "std of lr over 30 d", "31 d"),
    _f("c_vol_7_30", VOL, C, "c_vol7 / c_vol30", "31 d"),
    _f(
        "c_range_7_30",
        VOL,
        C,
        "(max h 7 - min l 7) / (max h 30 - min l 30) (compression)",
        "30 d",
    ),
    _f(
        "c_habit",
        VOL,
        C,
        "share of past entries b (D-365 .. D-30) whose 30-day window [b, b+30) ended by D "
        f"with max h >= 1.3 c(b-1); null under {HABIT_MIN} windows",
        "366 d",
    ),
    _f("c_maxgain30", VOL, C, "largest c[k] / c[k+1] - 1 over the last 30 days", "31 d"),
    _f("c_qv_1_30", FLOW, C, "qv[0] / mean qv of the 30 days before it", "31 d"),
    _f("c_qv_7_30", FLOW, C, "mean qv 7 d / mean qv 30 d", "30 d"),
    _f("c_qv_30_90", FLOW, C, "mean qv 30 d / mean qv 90 d", "90 d"),
    _f(
        "c_qv_z",
        FLOW,
        C,
        "z-score of log(1+qv[0]) against the 30 days before it",
        "31 d",
    ),
    _f("c_tb1", FLOW, C, "tb[0] / qv[0] (taker-buy share of the last day)", "1 d"),
    _f("c_tb7", FLOW, C, "sum tb 7 d / sum qv 7 d", "7 d"),
    _f("c_tb_chg", FLOW, C, "c_tb7 - (sum tb 30 d / sum qv 30 d)", "30 d"),
    _f("c_tc_7_30", FLOW, C, "mean trade count 7 d / mean trade count 30 d", "30 d"),
    _f(
        "c_ats_7_30",
        FLOW,
        C,
        "average trade size (sum qv / sum tc) 7 d over 30 d",
        "30 d",
    ),
    _f("c_upvol14", FLOW, C, "sum qv on up days (c[k] > c[k+1]) / sum qv, 14 d", "15 d"),
    _f("c_rs7", REL, C, "c_ret7 - m_btc_ret7", "7 d"),
    _f("c_rs30", REL, C, "c_ret30 - m_btc_ret30", "30 d"),
    _f("c_rs90", REL, C, "c_ret90 - m_btc_ret90", "90 d"),
    _f("c_beta60", REL, C, "OLS slope of the coin's lr on BTC's lr over 60 d", "61 d"),
    _f(
        "c_resid30",
        REL,
        C,
        "log(c[0]/c[30]) - c_beta60 * BTC log(c[0]/c[30])",
        "61 d",
    ),
    _f("c_log_liq", REL, C, "log(panel liq) (mean daily qv of the 30 days before D)", "30 d"),
    _f(
        "c_liq_rank",
        REL,
        C,
        "rank of liq among the day's panel coins, (rank - 1) / (n - 1)",
        "30 d",
    ),
    _f("c_log_age", REL, C, "log(panel days_listed)", "all"),
    _f(
        "c_touch_ma",
        FLAG,
        C,
        "1 if l[0] <= SMA n (closes to D-2) for some n in 50/100/200, with l > SMA n on each "
        "of the 10 days before; null without SMA50",
        "211 d",
    ),
    _f("c_below_200d", FLAG, C, "1 if c[0] < mean(c, 200 d), else 0", "200 d"),
]
NAMES = [f.name for f in FACTORS_LIST]
MARKET = [f.name for f in FACTORS_LIST if f.kind == M]
COIN = [f.name for f in FACTORS_LIST if f.kind == C]
BTC_FACTORS = [n for n in MARKET if n.startswith(("m_btc_", "m_ethbtc_"))]
BREADTH_FACTORS = [n for n in MARKET if n not in BTC_FACTORS and n != "m_qv_7_90"]
PANEL_FACTORS = ["c_log_liq", "c_liq_rank", "c_log_age"]
SERIES_FACTORS = [n for n in COIN if n not in PANEL_FACTORS]  # from `coin_series`
HELPERS = ["_moon30"]  # per-coin inputs of the breadth factors, not factors themselves


# --- bars --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Bars:
    """Dense daily bars: index k is UTC day number `d0 + k`, from the first candle's day."""

    d0: int
    open: F64
    high: F64
    low: F64
    close: F64
    qv: F64
    tc: F64
    tb: F64

    @property
    def n(self) -> int:
        return len(self.close)


def daily_bars(c5: pl.DataFrame) -> Bars:
    """UTC-day bars from 5m candles with the flow columns (`read_5m(..., flow=True)`)."""
    if c5.height == 0:
        e = np.zeros(0)
        return Bars(0, e, e, e, e, e, e, e)
    g = (
        c5.sort("t")
        .group_by((pl.col("t") // DAY).alias("day"), maintain_order=True)
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("quote_volume").sum().alias("qv"),
            pl.col("trade_count").sum().alias("tc"),
            pl.col("taker_buy_quote_volume").sum().alias("tb"),
        )
    )
    day = g["day"].to_numpy().astype(np.int64)
    d0 = int(day[0])
    n = int(day[-1]) - d0 + 1
    k = day - d0
    have = np.zeros(n, dtype=bool)
    have[k] = True
    close = np.full(n, np.nan)
    close[k] = g["close"].to_numpy()
    last = np.maximum.accumulate(np.where(have, np.arange(n), 0))
    close = close[last]  # forward fill: a day without candles closes at the last close

    def price(col: str) -> F64:
        out = close.copy()
        out[k] = g[col].to_numpy()
        return out

    def volume(col: str) -> F64:
        out = np.zeros(n)
        out[k] = g[col].to_numpy()
        return out

    return Bars(
        d0,
        price("open"),
        price("high"),
        price("low"),
        close,
        volume("qv"),
        volume("tc"),
        volume("tb"),
    )


def bars_from_candles(c5: pl.DataFrame, cut_us: int | None = None) -> Bars:
    """Bars from the candles that opened before `cut_us` (all candles if None)."""
    if cut_us is not None:
        c5 = c5.filter(pl.col("t") < cut_us)
    return daily_bars(c5)


# --- trailing window helpers (value at i uses x[i-n+1 .. i] only) -----------------------------


def nans(n: int) -> F64:
    return np.full(n, np.nan)


def roll(x: F64, n: int, fn: Callable[[F64], F64]) -> F64:
    out = nans(len(x))
    if len(x) >= n:
        out[n - 1 :] = fn(sliding_window_view(x, n))
    return out


def rsum(x: F64, n: int) -> F64:
    return roll(x, n, lambda w: w.sum(axis=1))


def rmean(x: F64, n: int) -> F64:
    return roll(x, n, lambda w: w.mean(axis=1))


def rmax(x: F64, n: int) -> F64:
    return roll(x, n, lambda w: w.max(axis=1))


def rmin(x: F64, n: int) -> F64:
    return roll(x, n, lambda w: w.min(axis=1))


def rstd(x: F64, n: int) -> F64:
    return roll(x, n, lambda w: w.std(axis=1, ddof=1))


def lag(x: F64, k: int) -> F64:
    """x[i - k] at i."""
    out = nans(len(x))
    if k < len(x):
        out[k:] = x[: len(x) - k]
    return out


def ratio(a: F64, b: F64) -> F64:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def ret(c: F64, k: int) -> F64:
    return ratio(c, lag(c, k)) - 1


def log_ret(c: F64) -> F64:
    return np.log(ratio(c, lag(c, 1)))


def align(x: F64, src_d0: int, d0: int, n: int) -> F64:
    """`x` (indexed from day `src_d0`) re-indexed to days `d0 .. d0 + n - 1`, NaN outside."""
    out = nans(n)
    lo = max(d0, src_d0)
    hi = min(d0 + n, src_d0 + len(x))
    if hi > lo:
        out[lo - d0 : hi - d0] = x[lo - src_d0 : hi - src_d0]
    return out


def habit_hits(high: F64, close: F64) -> F64:
    """At index b: 1 if max high of bars b+1 .. b+30 >= 1.3 close[b], else 0.

    That is the "+30% in 30 days" event of the entry whose last complete day is bar b. NaN
    while the window is not complete in the series, so a value at b is only ever used at an
    index i >= b + 30, when every bar of the window had closed.
    """
    n = len(close)
    out = nans(n)
    if n > MOON_DAYS:
        fwd = rmax(high, MOON_DAYS)[MOON_DAYS:]  # max high of bars b+1 .. b+30, b = 0 .. n-31
        out[: n - MOON_DAYS] = (fwd >= (1 + MOON) * close[: n - MOON_DAYS]).astype(np.float64)
    return out


def habit(hits: F64) -> F64:
    """At i: mean of hits[b] for b in [i - 365, i - 30], b >= 0; NaN under HABIT_MIN windows."""
    span = HABIT_SPAN - MOON_DAYS + 1
    ended = lag(hits, MOON_DAYS)  # hits[i - 30]: the last window that had ended by D
    padded = np.concatenate([nans(span - 1), ended])
    w = sliding_window_view(padded, span)
    cnt = np.isfinite(w).sum(axis=1)
    tot = np.nansum(w, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = tot / cnt
    out[cnt < HABIT_MIN] = np.nan
    return np.asarray(out, dtype=np.float64)


def sma_touch(low: F64, close: F64, n: int) -> F64:
    """1 if day i's low touched SMA n (closes to i-1) after 10 days of lows above it."""
    level = lag(rmean(close, n), 1)
    above = np.where(np.isnan(level), np.nan, (low > level).astype(np.float64))
    prior = lag(rmin(above, 10), 1)  # 1 when every low of i-10 .. i-1 was above its level
    out = np.where(np.isnan(prior) | np.isnan(level), np.nan, 0.0)
    out[(prior == 1) & (low <= level)] = 1.0
    return out


# --- factor series -----------------------------------------------------------------------------


def coin_series(b: Bars, btc: Bars) -> dict[str, F64]:
    """Every factor of `SERIES_FACTORS` (and `HELPERS`) at every bar index of `b`."""
    c, h, lo, qv, tb, tc = b.close, b.high, b.low, b.qv, b.tb, b.tc
    bc = align(btc.close, btc.d0, b.d0, b.n)
    lr, blr = log_ret(c), log_ret(bc)
    prev = lag(c, 1)
    up = np.where(np.isnan(prev), np.nan, (c > prev).astype(np.float64))
    tr = np.fmax(h - lo, np.fmax(np.abs(h - prev), np.abs(lo - prev)))
    tr[np.isnan(prev)] = np.nan
    atr = rmean(tr, 14)
    sma = {n: rmean(c, n) for n in (20, 50, 100, 200)}
    hi30, lo30, hi90, lo90 = rmax(h, 30), rmin(lo, 30), rmax(h, 90), rmin(lo, 90)
    vol7, vol30 = rstd(lr, 7), rstd(lr, 30)
    lq = np.log1p(qv)
    tb7 = ratio(rsum(tb, 7), rsum(qv, 7))
    hits = habit_hits(h, c)

    # beta: OLS slope of lr on blr over 60 days
    beta = nans(b.n)
    if b.n >= 60:
        wa, wb = sliding_window_view(lr, 60), sliding_window_view(blr, 60)
        da, db = wa - wa.mean(axis=1, keepdims=True), wb - wb.mean(axis=1, keepdims=True)
        beta[59:] = ratio((da * db).sum(axis=1), (db * db).sum(axis=1))

    touches = [sma_touch(lo, c, n) for n in (50, 100, 200)]
    touch = np.fmax(np.fmax(touches[0], touches[1]), touches[2])
    touch[np.isnan(touches[0])] = np.nan

    out: dict[str, F64] = {
        "c_ret1": ret(c, 1),
        "c_ret3": ret(c, 3),
        "c_ret7": ret(c, 7),
        "c_ret14": ret(c, 14),
        "c_ret30": ret(c, 30),
        "c_ret90": ret(c, 90),
        "c_dd30": ratio(c, hi30) - 1,
        "c_dd90": ratio(c, hi90) - 1,
        "c_ru30": ratio(c, lo30) - 1,
        "c_ru90": ratio(c, lo90) - 1,
        "c_pos30": ratio(c - lo30, hi30 - lo30),
        "c_sma20": ratio(c, sma[20]) - 1,
        "c_sma50": ratio(c, sma[50]) - 1,
        "c_sma100": ratio(c, sma[100]) - 1,
        "c_sma200": ratio(c, sma[200]) - 1,
        "c_sma50_slope": ratio(sma[50], lag(sma[50], 10)) - 1,
        "c_up10": rsum(up, 10),
        "c_ret1_atr": ratio(c - prev, atr),
        "c_atr14": ratio(atr, c),
        "c_vol7": vol7,
        "c_vol30": vol30,
        "c_vol_7_30": ratio(vol7, vol30),
        "c_range_7_30": ratio(rmax(h, 7) - rmin(lo, 7), hi30 - lo30),
        "c_habit": habit(hits),
        "c_maxgain30": rmax(ratio(c, prev) - 1, 30),
        "c_qv_1_30": ratio(qv, lag(rmean(qv, 30), 1)),
        "c_qv_7_30": ratio(rmean(qv, 7), rmean(qv, 30)),
        "c_qv_30_90": ratio(rmean(qv, 30), rmean(qv, 90)),
        "c_qv_z": ratio(lq - lag(rmean(lq, 30), 1), lag(rstd(lq, 30), 1)),
        "c_tb1": ratio(tb, qv),
        "c_tb7": tb7,
        "c_tb_chg": tb7 - ratio(rsum(tb, 30), rsum(qv, 30)),
        "c_tc_7_30": ratio(rmean(tc, 7), rmean(tc, 30)),
        "c_ats_7_30": ratio(ratio(rsum(qv, 7), rsum(tc, 7)), ratio(rsum(qv, 30), rsum(tc, 30))),
        "c_upvol14": ratio(rsum(qv * up, 14), rsum(qv, 14)),
        "c_rs7": ret(c, 7) - ret(bc, 7),
        "c_rs30": ret(c, 30) - ret(bc, 30),
        "c_rs90": ret(c, 90) - ret(bc, 90),
        "c_beta60": beta,
        "c_resid30": np.log(ratio(c, lag(c, 30))) - beta * np.log(ratio(bc, lag(bc, 30))),
        "c_touch_ma": touch,
        "c_below_200d": np.where(np.isnan(sma[200]), np.nan, (c < sma[200]).astype(np.float64)),
        "_moon30": lag(hits, MOON_DAYS),
    }
    assert set(out) == {*SERIES_FACTORS, *HELPERS}
    return out


def market_series(btc: Bars, eth: Bars) -> dict[str, F64]:
    """The BTC and ETH factors at every bar index of `btc`."""
    c, h = btc.close, btc.high
    ec = align(eth.close, eth.d0, btc.d0, btc.n)
    lr = log_ret(c)
    out = {
        "m_btc_ret1": ret(c, 1),
        "m_btc_ret7": ret(c, 7),
        "m_btc_ret30": ret(c, 30),
        "m_btc_ret90": ret(c, 90),
        "m_btc_sma50": ratio(c, rmean(c, 50)) - 1,
        "m_btc_sma200": ratio(c, rmean(c, 200)) - 1,
        "m_btc_dd90": ratio(c, rmax(h, 90)) - 1,
        "m_btc_vol7": rstd(lr, 7),
        "m_btc_vol30": rstd(lr, 30),
        "m_btc_vol_30_180": ratio(rstd(lr, 30), rstd(lr, 180)),
        "m_ethbtc_ret30": ratio(ret(ec, 30) + 1, ret(c, 30) + 1) - 1,
    }
    assert list(out) == BTC_FACTORS
    return out


def at_days(series: dict[str, F64], d0: int, n: int, d_us: I64) -> pl.DataFrame:
    """The series values of the entries at `d_us` (bar of day D - 1), null where NaN."""
    idx = d_us // DAY - 1 - d0
    assert bool(np.all((idx >= 0) & (idx < n))), "an entry without its last complete day"
    df = pl.DataFrame({"d_us": d_us, **{k: v[idx] for k, v in series.items()}})
    return df.with_columns(pl.exclude("d_us").fill_nan(None))


def breadth(rows: pl.DataFrame) -> pl.DataFrame:
    """Breadth factors per day from the coin values of the day's panel entries."""
    out = rows.group_by("d_us").agg(
        m_breadth20=(pl.col("c_sma20") > 0).mean(),
        m_breadth50=(pl.col("c_sma50") > 0).mean(),
        m_breadth200=(pl.col("c_sma200") > 0).mean(),
        m_med_ret7=pl.col("c_ret7").median(),
        m_disp_ret7=pl.col("c_ret7").log1p().std(),
        m_med_ret30=pl.col("c_ret30").median(),
        m_disp_ret30=pl.col("c_ret30").log1p().std(),
        m_moon30=pl.col("_moon30").mean(),
    )
    return out.select("d_us", *BREADTH_FACTORS).cast(dict.fromkeys(BREADTH_FACTORS, pl.Float64))


def market_volume(daily_qv: pl.DataFrame, d_us: I64) -> pl.DataFrame:
    """`m_qv_7_90` at each D from per-coin daily quote volumes (columns `day`, `qv`)."""
    tot = daily_qv.group_by("day").agg(pl.col("qv").sum()).sort("day")
    day = tot["day"].to_numpy().astype(np.int64)
    d0 = int(day[0])
    qv = np.zeros(int(day[-1]) - d0 + 1)
    qv[day - d0] = tot["qv"].to_numpy()
    series = {"m_qv_7_90": ratio(rmean(qv, 7), rmean(qv, 90))}
    return at_days(series, d0, len(qv), d_us)


def panel_factors(panel: pl.DataFrame) -> pl.DataFrame:
    """Factors from the panel's own causal columns (`liq`, `days_listed`)."""
    rank = (pl.col("liq").rank("average").over("d_us") - 1) / (pl.len().over("d_us") - 1)
    return panel.select(
        "symbol",
        "d_us",
        c_log_liq=pl.col("liq").log(),
        c_liq_rank=pl.when(pl.len().over("d_us") > 1).then(rank).otherwise(0.5),
        c_log_age=pl.col("days_listed").cast(pl.Float64).log(),
    )


def coin_rows(symbol: str, c5: pl.DataFrame, btc: Bars, d_us: I64, p: F64) -> pl.DataFrame:
    """The coin factors (and helpers) of one coin's entries at `d_us`, from its 5m candles."""
    b = daily_bars(c5)
    df = at_days(coin_series(b, btc), b.d0, b.n, d_us)
    if len(d_us):
        last_close = b.close[d_us // DAY - 1 - b.d0]
        assert np.array_equal(last_close, p), f"{symbol}: P is not the close of day D - 1"
    return df.with_columns(symbol=pl.lit(symbol))


# --- build -------------------------------------------------------------------------------------

_BTC: Bars | None = None


def _btc() -> Bars:
    global _BTC
    if _BTC is None:
        _BTC = daily_bars(read_5m(BTC, flow=True))
    return _BTC


def coin_job(args: tuple[str, I64, F64]) -> tuple[pl.DataFrame, pl.DataFrame]:
    symbol, d_us, p = args
    c5 = read_5m(symbol, flow=True)
    b = daily_bars(c5)
    qv = pl.DataFrame({"day": b.d0 + np.arange(b.n, dtype=np.int64), "qv": b.qv})
    return coin_rows(symbol, c5, _btc(), d_us, p), qv


def assemble(
    panel: pl.DataFrame,
    coins: pl.DataFrame,
    daily_qv: pl.DataFrame,
    btc: Bars,
    eth: Bars,
) -> pl.DataFrame:
    """The factor table from the coin rows, the coins' daily volumes and the BTC/ETH bars."""
    keys = panel.select("symbol", "d_us")
    rows = keys.join(coins, on=["symbol", "d_us"], how="left")
    days = np.sort(panel["d_us"].unique().to_numpy().astype(np.int64))
    mkt = at_days(market_series(btc, eth), btc.d0, btc.n, days)
    mkt = mkt.join(breadth(rows), on="d_us").join(market_volume(daily_qv, days), on="d_us")
    out = (
        rows.join(mkt, on="d_us", how="left")
        .join(panel_factors(panel), on=["symbol", "d_us"], how="left")
        .select("symbol", "d_us", *NAMES)
        .cast(dict.fromkeys(NAMES, pl.Float64))
    )
    return out.with_columns(pl.col(NAMES).fill_nan(None))


def write_doc() -> None:
    lines = [
        "# Rebound factors: the factor dictionary (M2)",
        "",
        "Built by `build_factors.py` into `cache/factors.parquet`, one row per panel entry",
        "(`symbol`, `d_us`), every year. Every factor uses only 5m candles with open time before",
        "`D`, aggregated to UTC-day bars; the last complete day is `D - 1` and its close is `P`.",
        "Notation: `c[k]` is the daily close `k` days before the last complete day (so",
        "`c[0] = P`), `h`/`l` daily high/low, `qv` quote volume, `tb` taker-buy quote volume,",
        "`tc` trade count, `lr` the daily log return; `max h 30` is the highest daily high of",
        "the 30 days `D-30 .. D-1`. A day without candles is a flat bar at the last close with",
        "zero volume. A factor is null when its lookback reaches before the coin's first 5m",
        "candle (the 5m data start on 2020-01-01, so long lookbacks are null early in 2020).",
        "",
        "*Market* factors are the same for every coin on a day; *coin* factors differ by coin.",
        "Breadth factors use the coins in the panel on that day. Within-day ranks of the coin",
        "factors are computed in M3.",
        "",
        "| # | name | family | kind | formula | lookback |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i, f in enumerate(FACTORS_LIST, 1):
        lines.append(f"| {i} | `{f.name}` | {f.family} | {f.kind} | {f.formula} | {f.lookback} |")
    counts = pl.DataFrame([vars(f) for f in FACTORS_LIST]).group_by("family", "kind").len()
    lines += ["", "## Count by family", "", "| family | kind | factors |", "| --- | --- | --- |"]
    for r in counts.sort("kind", "family", descending=[True, False]).iter_rows(named=True):
        lines.append(f"| {r['family']} | {r['kind']} | {r['len']} |")
    lines.append(f"| **all** | | {len(FACTORS_LIST)} |")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "factors.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--doc-only", action="store_true", help="only write results/factors.md")
    args = parser.parse_args()
    write_doc()
    if args.doc_only:
        return
    t0 = time.perf_counter()
    panel = pl.read_parquet(PANEL)
    by_coin = {
        s[0]: g.sort("d_us") for s, g in panel.select("symbol", "d_us", "P").group_by("symbol")
    }
    jobs = []
    for sym in universe():
        g = by_coin.get(sym)
        d = g["d_us"].to_numpy() if g is not None else np.zeros(0, dtype=np.int64)
        p = g["P"].to_numpy() if g is not None else np.zeros(0)
        jobs.append((sym, d.astype(np.int64), p.astype(np.float64)))
    assert set(by_coin) <= {j[0] for j in jobs}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        parts = list(pool.map(coin_job, jobs, chunksize=2))
    t1 = time.perf_counter()
    coins = pl.concat([p[0] for p in parts])
    daily_qv = pl.concat([p[1] for p in parts])
    btc = daily_bars(read_5m(BTC, flow=True))
    eth = daily_bars(read_5m(ETH, flow=True))
    out = assemble(panel, coins, daily_qv, btc, eth).sort("d_us", "symbol")
    FACTORS.parent.mkdir(parents=True, exist_ok=True)
    tmp = FACTORS.with_name(f".{FACTORS.name}.tmp")
    out.write_parquet(tmp)
    tmp.replace(FACTORS)
    t2 = time.perf_counter()
    print(f"{len(jobs)} coins read and computed in {t1 - t0:.0f}s, assembled in {t2 - t1:.0f}s")
    print(f"factors.parquet: {out.height} rows x {len(NAMES)} factors")


if __name__ == "__main__":
    main()
