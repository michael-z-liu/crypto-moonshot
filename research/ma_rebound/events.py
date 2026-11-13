"""M1 core: bars, SMA levels, candidates, limit fills and exit labels for one coin.

Every function here works on one coin's candles cut at `end_us` (exclusive, µs since the epoch):
only candles that closed by then are read, so a build cut at an earlier time sees exactly what a
live run would have seen then (the truncation test in `check_events.py` relies on this). The
build scripts call these per coin; the definitions are recorded in `results/M1.md`.

Times are int64 microseconds since the Unix epoch, UTC, throughout.

Definitions settled in M1 (the protocol is in `docs/plans/ma-rebound.md`):

- Bars of 1h/4h/1d/1w/1M are aggregated from the 5m candles on UTC boundaries (4h from 00:00,
  weeks from Monday, calendar months); before 2020 the 1d bars come from the downloaded daily
  candles and 1w/1M are aggregated from the 1d bars. A bar exists when it has at least one candle.
- Level in force during bar i: mean close of the n bars before it (the last n bars that have
  candles); null when fewer than n exist or they span more than 1.1 x n bar periods. Computed
  exactly on prices in units of 1e-8, so ties with tick prices are exact.
- `atr_d` in force during a day: mean true range of the 14 daily bars before it (true range uses
  the previous existing bar's close) over the last daily close; null with fewer than 15 bars or a
  span of more than 1.1 x 14 days. `liq`: total quote volume of the 30 calendar days before the
  day, over 30 (days without candles count as zero); null within 30 days of the first daily bar.
- Above (after 5m candle j): the k bars before j's bar each had their low above their own level,
  j's bar has a level, and no 5m low of j's bar up to and including j has reached it. Armed: a 5m
  close >= L(1 + 0.45u) while above. Trigger: the first 5m close <= L(1 + 0.30u) while armed and
  above; the trigger, any candle that is not above, and any candle without u disarm.
- Touch-market: the 5m candle whose low reaches L while above held before it (so the first such
  candle of the bar, after k bars above). Entry at the next 5m candle's open, as a taker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from common import (
    ARM_ZONE,
    ATR_DAYS,
    BUFFERS,
    DAILY_PRE2020,
    EXITS,
    FEE,
    KLINES_1M,
    KLINES_5M,
    LIQUIDITY_DAYS,
    MA_LENGTHS,
    MARKET_SLIPPAGE,
    MAX_HOLD,
    MIN_DAILY_QUOTE_VOLUME,
    PLACEBO_LENGTHS,
    PRIMARY_BUFFER,
    STORE_START,
    STRESS_EXTRA_COST,
    STRESS_THROUGH,
    TIMEFRAMES,
    TRIGGER_ZONE,
)

from crypto_moonshot.broker.base import ExitPlan, ExitReason
from crypto_moonshot.broker.candles import Candles
from crypto_moonshot.broker.simulated import fill_path, resolve_exit
from crypto_moonshot.models import EPOCH, Interval

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]
type Bools = npt.NDArray[np.bool_]

MINUTE = 60_000_000
FIVE = 5 * MINUTE
HOUR = 60 * MINUTE
DAY = 24 * HOUR
STORE_START_US = int(STORE_START.timestamp()) * 1_000_000
SPAN_SLACK = 1.1  # a level (or atr_d) is null if its bars span more than 1.1 x n periods
ENDED_GAP = 2 * DAY  # a series whose last candle closed this long before the data end has ended
FWD_STEPS = 32

TF_NAMES = tuple(TIMEFRAMES)
LENGTHS = {tf: tuple(sorted(set(MA_LENGTHS[tf]) | set(PLACEBO_LENGTHS))) for tf in TF_NAMES}
PRICE_COLS = ("open", "high", "low", "close")


def hold_us(tf: str) -> int:
    return TIMEFRAMES[tf].hold // timedelta(microseconds=1)


def exit_hold_us(tf: str, exit_name: str) -> int:
    """Time limit of an exit: `time` x T, at most 28 days."""
    hold = TIMEFRAMES[tf].hold * EXITS[exit_name].time
    return min(hold, MAX_HOLD) // timedelta(microseconds=1)


# --- loading -------------------------------------------------------------------------------


def _month_stem(us: int) -> str:
    return str(np.datetime64(us, "us").astype("datetime64[M]"))


def read_klines(
    root: Path, symbol: str, end_us: int, step_us: int, quote_volume: bool = False
) -> pl.DataFrame:
    """A symbol's candles that closed by `end_us`, as `t` (open time, µs) and OHLC columns."""
    folder = root / f"symbol={symbol}"
    last = _month_stem(end_us - 1)
    files = [f for f in sorted(folder.glob("*.parquet")) if f.stem <= last]
    cols = ["open_time", *PRICE_COLS, *(["quote_volume"] if quote_volume else [])]
    if not files:
        schema = {"t": pl.Int64, **dict.fromkeys(cols[1:], pl.Float64)}
        return pl.DataFrame(schema=schema)
    df = pl.read_parquet(files, columns=cols)
    df = df.with_columns(t=pl.col("open_time").dt.epoch("us")).select("t", *cols[1:])
    return df.filter(pl.col("t") + step_us <= end_us).sort("t")


def read_5m(symbol: str, end_us: int) -> pl.DataFrame:
    return read_klines(KLINES_5M, symbol, end_us, FIVE, quote_volume=True)


def read_1m(symbol: str, end_us: int) -> pl.DataFrame:
    return read_klines(KLINES_1M, symbol, end_us, MINUTE)


def read_daily_pre2020(symbol: str, end_us: int) -> pl.DataFrame | None:
    path = DAILY_PRE2020 / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pl.read_parquet(path, columns=["open_time", *PRICE_COLS, "quote_volume"])
    df = df.with_columns(t=pl.col("open_time").dt.epoch("us")).select(
        "t", *PRICE_COLS, "quote_volume"
    )
    return df.filter(pl.col("t") < STORE_START_US, pl.col("t") + DAY <= end_us).sort("t")


def to_candles(df: pl.DataFrame, interval: Interval, ended: bool) -> Candles:
    return Candles(
        interval=interval,
        open_time=df["t"].to_numpy().astype(np.int64),
        open=df["open"].to_numpy().astype(np.float64),
        high=df["high"].to_numpy().astype(np.float64),
        low=df["low"].to_numpy().astype(np.float64),
        close=df["close"].to_numpy().astype(np.float64),
        ended=ended,
    )


@dataclass(frozen=True)
class Series:
    """One coin's 5m and 1m candles, cut at `end_us`.

    `ended`: the 5m series stopped at least two days before `end_us` (a delisting); `known_until`
    is the close time of its last 5m candle. Anything that needs candles past `known_until` of a
    series that has not ended is censored (unknown), not closed.
    """

    symbol: str
    end_us: int
    c5: Candles
    c1: Candles
    ended: bool
    known_until: int


def make_series(symbol: str, c5: pl.DataFrame, c1: pl.DataFrame, end_us: int) -> Series:
    last = int(c5["t"][-1]) + FIVE if c5.height else end_us
    ended = last <= end_us - ENDED_GAP
    return Series(
        symbol,
        end_us,
        to_candles(c5, Interval.M5, ended),
        to_candles(c1, Interval.M1, ended),
        ended,
        last,
    )


# --- bars and levels -----------------------------------------------------------------------


def bar_index(t: I64, tf: str) -> I64:
    """Period number of each time on the timeframe's UTC grid."""
    if tf == "1h":
        return t // HOUR
    if tf == "4h":
        return t // (4 * HOUR)
    if tf == "1d":
        return t // DAY
    if tf == "1w":
        return (t // DAY + 3) // 7  # 1970-01-01 was a Thursday; week 0 starts Monday -3
    if tf == "1M":
        return t.astype("datetime64[us]").astype("datetime64[M]").astype(np.int64)
    raise ValueError(tf)


def bar_start(idx: I64, tf: str) -> I64:
    if tf == "1h":
        return idx * HOUR
    if tf == "4h":
        return idx * 4 * HOUR
    if tf == "1d":
        return idx * DAY
    if tf == "1w":
        return (idx * 7 - 3) * DAY
    if tf == "1M":
        return idx.astype("datetime64[M]").astype("datetime64[us]").astype(np.int64)
    raise ValueError(tf)


def aggregate(df: pl.DataFrame, tf: str) -> pl.DataFrame:
    """OHLC and quote volume per bar of `tf`, from finer rows (`t`, OHLC, `quote_volume`).

    numpy segment reductions rather than a polars group-by: their summation order depends only
    on the bar's own rows, so a build cut at an earlier time gives bit-identical bars.
    """
    idx = bar_index(df["t"].to_numpy().astype(np.int64), tf)
    if len(idx) == 0:
        empty = {"idx": pl.Int64, **dict.fromkeys(PRICE_COLS, pl.Float64)}
        return pl.DataFrame(schema={**empty, "quote_volume": pl.Float64, "t": pl.Int64})
    starts = np.flatnonzero(np.concatenate(([True], idx[1:] != idx[:-1])))
    ends = np.concatenate((starts[1:], [len(idx)]))
    out_idx = idx[starts]
    return pl.DataFrame(
        {
            "idx": out_idx,
            "open": df["open"].to_numpy()[starts],
            "high": np.maximum.reduceat(df["high"].to_numpy(), starts),
            "low": np.minimum.reduceat(df["low"].to_numpy(), starts),
            "close": df["close"].to_numpy()[ends - 1],
            "quote_volume": np.add.reduceat(df["quote_volume"].to_numpy(), starts),
            "t": bar_start(out_idx, tf),
        }
    )


PRICE_SCALE = 10**8  # Binance prices have at most 8 decimals


def sma_level(close: F64, idx: I64, n: int) -> F64:
    """Level in force during each bar: mean close of the n bars before it (NaN if undefined).

    Exact: the closes are summed as integers of 1e-8 and the sum divided once (a correctly
    rounded quotient), so a level equal to a tick price compares equal to it, whatever the
    order of summation (the direct-loop check computes it the same way with Python ints).
    """
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return out
    ticks = np.rint(close * PRICE_SCALE).astype(np.int64)
    cs = np.concatenate(([0], np.cumsum(ticks)))
    sums = cs[n:-1] - cs[: len(close) - n]  # for bar i >= n: closes of bars i-n .. i-1
    span = idx[n - 1 : -1] - idx[: len(close) - n] + 1
    level = sums.astype(np.float64) / float(n * PRICE_SCALE)
    out[n:] = np.where(span <= SPAN_SLACK * n, level, np.nan)
    return out


def atr_frac(high: F64, low: F64, close: F64, idx: I64) -> F64:
    """`atr_d` in force during each daily bar (NaN if undefined)."""
    n = ATR_DAYS
    out = np.full(len(close), np.nan)
    if len(close) <= n + 1:
        return out
    prev = close[:-1]
    tr = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - prev), np.abs(low[1:] - prev)])
    # tr[j] belongs to bar j + 1. For bar i >= n + 1: tr of bars i-n..i-1 = tr[i-n-1 : i-1].
    windows = np.lib.stride_tricks.sliding_window_view(tr, n)[:-1]
    i = np.arange(n + 1, len(close))
    span = idx[i - 1] - idx[i - n] + 1
    ok = span <= SPAN_SLACK * n
    out[n + 1 :] = np.where(ok, windows.mean(axis=1) / close[i - 1], np.nan)
    return out


def liquidity(qv: F64, idx: I64) -> F64:
    """Mean daily quote volume of the 30 calendar days before each daily bar (NaN if undefined)."""
    if len(idx) == 0:
        return np.empty(0)
    dense = np.zeros(int(idx[-1] - idx[0]) + 1)
    dense[idx - idx[0]] = qv
    cs = np.concatenate(([0.0], np.cumsum(dense)))
    p = idx - idx[0]
    days = LIQUIDITY_DAYS
    return np.where(p >= days, (cs[p] - cs[np.maximum(p - days, 0)]) / days, np.nan)


def build_bars(c5: pl.DataFrame, pre: pl.DataFrame | None) -> dict[str, pl.DataFrame]:
    """Bars of every timeframe with the level of every study and placebo length."""
    base = c5.select("t", *PRICE_COLS, "quote_volume")
    bars = {"1h": aggregate(base, "1h"), "4h": aggregate(base, "4h")}
    day = aggregate(base, "1d")
    if pre is not None and pre.height:
        first = int(day["idx"][0]) if day.height else 1 << 62
        pre_day = aggregate(pre, "1d").filter(pl.col("idx") < first)
        day = pl.concat([pre_day, day])
    bars["1d"] = day
    bars["1w"] = aggregate(day, "1w")
    bars["1M"] = aggregate(day, "1M")
    out = {}
    for tf, b in bars.items():
        close = b["close"].to_numpy()
        idx = b["idx"].to_numpy()
        cols = [pl.Series(f"L{n}", sma_level(close, idx, n)) for n in LENGTHS[tf]]
        if tf == "1d":
            hi, lo = b["high"].to_numpy(), b["low"].to_numpy()
            cols.append(pl.Series("atr_d", atr_frac(hi, lo, close, idx)))
            cols.append(pl.Series("liq", liquidity(b["quote_volume"].to_numpy(), idx)))
        out[tf] = b.select("idx", "t", *PRICE_COLS, "quote_volume").with_columns(cols)
        out[tf] = out[tf].with_columns(pl.selectors.float().fill_nan(None))
    return out


def level_array(bars: pl.DataFrame, n: int) -> F64:
    return bars[f"L{n}"].fill_null(np.nan).to_numpy().astype(np.float64)


def positions(bars: pl.DataFrame, t: I64, tf: str) -> I64:
    """Position in `bars` of the bar containing each time; -1 where no bar has candles."""
    idx = bars["idx"].to_numpy()
    want = bar_index(t, tf)
    pos = np.searchsorted(idx, want)
    pos_c = np.minimum(pos, len(idx) - 1)
    return np.where((len(idx) > 0) & (idx[pos_c] == want), pos_c, -1).astype(np.int64)


# --- candidates ----------------------------------------------------------------------------


def _last_before(ev: Bools) -> I64:
    """For each j, the index of the last true element before j (-1 if none)."""
    last = np.maximum.accumulate(np.where(ev, np.arange(len(ev)), -1))
    return np.concatenate(([-1], last[:-1])).astype(np.int64)


def _cummin_in_bar(low: F64, pos: I64) -> tuple[F64, F64]:
    """Running min of the 5m lows within each bar, after and before each candle."""
    df = pl.DataFrame({"g": pos, "low": low})
    after = df.select(pl.col("low").cum_min().over("g"))["low"].to_numpy()
    before = df.select(pl.col("low").cum_min().shift(1).over("g"))["low"]
    return after, before.fill_null(np.inf).to_numpy()


def ma_list() -> list[tuple[str, int, bool]]:
    """(timeframe, length, placebo) of every MA built in M1."""
    study = [(tf, n, False) for tf in TF_NAMES for n in MA_LENGTHS[tf]]
    return study + [(tf, n, True) for tf in TF_NAMES for n in PLACEBO_LENGTHS]


def build_candidates(symbol: str, c5: pl.DataFrame, bars: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Both entry families' candidates of every study and placebo MA, after the filters."""
    t = c5["t"].to_numpy().astype(np.int64)
    low = c5["low"].to_numpy()
    close = c5["close"].to_numpy()
    pos_d = positions(bars["1d"], t, "1d")
    atr5 = bars["1d"]["atr_d"].fill_null(np.nan).to_numpy()[pos_d]
    liq5 = bars["1d"]["liq"].fill_null(np.nan).to_numpy()[pos_d]
    frames = []
    for tf in TF_NAMES:
        b = bars[tf]
        pos = positions(b, t, tf)
        assert (pos >= 0).all(), "every 5m candle lies in a bar"
        cm_after, cm_before = _cummin_in_bar(low, pos)
        blow = b["low"].to_numpy()
        spec = TIMEFRAMES[tf]
        k = spec.above_bars
        u5 = atr5 * math.sqrt(spec.hold_days)
        u_ok = np.isfinite(u5) & (u5 > 0)
        nb = len(blow)
        for n, placebo in [(n, False) for n in MA_LENGTHS[tf]] + [
            (n, True) for n in PLACEBO_LENGTHS
        ]:
            lev = level_array(b, n)
            abv = blow > lev  # NaN level: False
            cs = np.concatenate(([0], np.cumsum(abv)))
            bi = np.arange(nb)
            prev_ok = (bi >= k) & (cs[bi] - cs[np.maximum(bi - k, 0)] == k)
            touched = blow <= lev
            last_touch = _last_before(touched)
            L5 = lev[pos]
            ok5 = prev_ok[pos] & np.isfinite(L5)
            above_after = ok5 & (cm_after > L5)
            above_before = ok5 & (cm_before > L5)
            live = above_after & u_ok
            arm = live & (close >= L5 * (1 + ARM_ZONE * u5))
            t0 = live & (close <= L5 * (1 + TRIGGER_ZONE * u5))
            la = _last_before(arm)
            trig = t0 & (la > _last_before(~live)) & (la > _last_before(t0))
            touch = above_before & u_ok & (low <= L5)
            keep = u_ok & (liq5 >= MIN_DAILY_QUOTE_VOLUME)
            for family, ev in (("limit", trig & keep), ("touch", touch & keep)):
                j = np.flatnonzero(ev)
                if len(j) == 0:
                    continue
                bp = pos[j]
                lt = last_touch[bp]
                slope_ref = np.where(bp >= 10, lev[np.maximum(bp - 10, 0)], np.nan)
                u = u5[j]
                L = L5[j]
                frames.append(
                    pl.DataFrame(
                        {
                            "family": family,
                            "placebo": placebo,
                            "symbol": symbol,
                            "tf": tf,
                            "n": n,
                            "trigger_t": t[j],
                            "decision_t": t[j] + FIVE,
                            "L": L,
                            "u": u,
                            "atr_d": atr5[j],
                            "liq": liq5[j],
                            "close": close[j],
                            "low": low[j],
                            "distance": (close[j] / L - 1) / u,
                            "bars_since_touch": np.where(lt >= 0, bp - 1 - lt, -1),
                            "slope10": (L - slope_ref) / L / u,
                        },
                        schema_overrides={"n": pl.Int32},
                    )
                )
    if not frames:
        return empty_candidates()
    df = pl.concat(frames)
    return finish_candidates(df)


def finish_candidates(df: pl.DataFrame) -> pl.DataFrame:
    ts = pl.from_epoch(pl.col("trigger_t"), time_unit="us")
    df = df.with_columns(
        bars_since_touch=pl.when(pl.col("bars_since_touch") >= 0).then(pl.col("bars_since_touch")),
        slope10=pl.col("slope10").fill_nan(None),
        year=ts.dt.year().cast(pl.Int32),
        cid=pl.concat_str(
            pl.col("symbol"),
            pl.col("tf"),
            pl.col("n").cast(pl.String),
            pl.col("family").str.slice(0, 1),
            ts.dt.strftime("%Y%m%d%H%M"),
            separator="-",
        ),
    )
    return df.select(CANDIDATE_COLS).sort("tf", "n", "family", "trigger_t")


CANDIDATE_COLS = [
    "cid",
    "family",
    "placebo",
    "symbol",
    "tf",
    "n",
    "trigger_t",
    "decision_t",
    "L",
    "u",
    "atr_d",
    "liq",
    "close",
    "low",
    "distance",
    "bars_since_touch",
    "slope10",
    "year",
]


def empty_candidates() -> pl.DataFrame:
    schema = {
        "cid": pl.String,
        "family": pl.String,
        "placebo": pl.Boolean,
        "symbol": pl.String,
        "tf": pl.String,
        "n": pl.Int32,
        "trigger_t": pl.Int64,
        "decision_t": pl.Int64,
        "L": pl.Float64,
        "u": pl.Float64,
        "atr_d": pl.Float64,
        "liq": pl.Float64,
        "close": pl.Float64,
        "low": pl.Float64,
        "distance": pl.Float64,
        "bars_since_touch": pl.Int64,
        "slope10": pl.Float64,
        "year": pl.Int32,
    }
    return pl.DataFrame(schema=schema)


# --- fills ---------------------------------------------------------------------------------


def buffers_for(family: str, placebo: bool) -> tuple[float | None, ...]:
    if family == "touch":
        return (None,)
    return (PRIMARY_BUFFER,) if placebo else BUFFERS


def window_us(c: Candles, start: int, end: int) -> Candles:
    """Candles opening in [start, end); `ended` only if the window reaches an ended series' end."""
    lo, hi = np.searchsorted(c.open_time, [start, end])
    return Candles(
        c.interval,
        c.open_time[lo:hi],
        c.open[lo:hi],
        c.high[lo:hi],
        c.low[lo:hi],
        c.close[lo:hi],
        c.ended and hi == len(c),
    )


def _first(hit: Bools) -> int:
    if len(hit) == 0:
        return 0
    i = int(hit.argmax())
    return i if hit[i] else len(hit)


FILL_SCHEMA = {
    "cid": pl.String,
    "buffer": pl.Float64,  # null for touch-market
    "scenario": pl.String,  # base | stress
    "status": pl.String,  # filled | expired | censored
    "kind": pl.String,  # maker | taker (filled only)
    "fill_t": pl.Int64,  # open time of the fill candle
    "fill_len": pl.Int64,  # its length (1m or 5m)
    "price": pl.Float64,  # raw fill price (limit price, or the candle open for a taker)
    "level_at_fill": pl.Float64,  # L in force during the fill candle
}


def build_fills(cands: pl.DataFrame, bars: dict[str, pl.DataFrame], series: Series) -> pl.DataFrame:
    """Entry fills of every candidate: limit fills per buffer (base and stress), touch entries."""
    rows: list[tuple[object, ...]] = []
    c5, c1 = series.c5, series.c1
    lev_cache: dict[tuple[str, int], F64] = {}
    for r in cands.iter_rows(named=True):
        tf, n = r["tf"], r["n"]
        key = (tf, n)
        if key not in lev_cache:
            lev_cache[key] = level_array(bars[tf], n)
        lev = lev_cache[key]
        d = r["decision_t"]
        if r["family"] == "touch":
            i = int(np.searchsorted(c5.open_time, d))
            if i < len(c5):
                t0 = int(c5.open_time[i])
                bp = positions(bars[tf], np.array([t0]), tf)[0]
                lv = float(lev[bp])
                fill = ("filled", "taker", t0, FIVE, float(c5.open[i]), lv)
                for sc in ("base", "stress"):
                    rows.append((r["cid"], None, sc, *fill))
            else:
                st = "expired" if series.ended else "censored"
                for sc in ("base", "stress"):
                    rows.append((r["cid"], None, sc, st, None, None, None, None, None))
            continue
        w_end = d + hold_us(tf) // 2
        path = fill_path(window_us(c5, d, w_end), window_us(c1, d, w_end))
        if len(path):
            lev_path = lev[positions(bars[tf], path.open_time, tf)]
        unknown = "censored" if (not series.ended and w_end > series.known_until) else "expired"
        for b in buffers_for("limit", r["placebo"]):
            assert b is not None
            if len(path) == 0:
                for sc in ("base", "stress"):
                    rows.append((r["cid"], b, sc, unknown, None, None, None, None, None))
                continue
            lim = lev_path * (1 + b * r["u"])
            taker = bool(r["close"] <= lim[0])
            for sc in ("base", "stress"):
                if taker:
                    i, kind, price = 0, "taker", float(path.open[0])
                else:
                    through = lim * (1 - STRESS_THROUGH) if sc == "stress" else lim
                    i = _first(path.low <= through)  # NaN limits never fill
                    kind = "maker"
                    price = float(lim[i]) if i < len(path) else math.nan
                if i < len(path):
                    rows.append(
                        (
                            r["cid"],
                            b,
                            sc,
                            "filled",
                            kind,
                            int(path.open_time[i]),
                            int(path.close_time[i] - path.open_time[i]),
                            price,
                            float(lev_path[i]),
                        )
                    )
                else:
                    rows.append((r["cid"], b, sc, unknown, None, None, None, None, None))
    return pl.DataFrame(rows, schema=FILL_SCHEMA, orient="row")


# --- labels --------------------------------------------------------------------------------


def exits_for(family: str, placebo: bool) -> tuple[str, ...]:
    if placebo:
        return ("E1",) if family == "touch" else ("E1", "E3")
    return tuple(EXITS)


@dataclass(frozen=True)
class EntryWindow:
    """Candles for an exit simulation that starts at a fill.

    `coarse` starts at the 5m candle holding the fill, its open replaced by the entry price (the
    reference price of `resolve_exit`). For a maker fill the fill candle keeps its low and close
    but its high is cut to max(entry, close): the part of the candle before the fill must not
    count. 1m candles of that 5m candle before the fill are flat at the entry price. `offset` is
    fill time minus the 5m candle's open, added to every time limit.
    """

    coarse: Candles
    fine: Candles
    offset: int


def entry_window(
    series: Series, fill_t: int, fill_len: int, price: float, maker: bool, horizon: int
) -> EntryWindow:
    s = fill_t - fill_t % FIVE
    end = fill_t + horizon
    c5, c1 = series.c5, series.c1
    lo5, hi5 = np.searchsorted(c5.open_time, [s, end])
    # 1m candles to the end of the 5m candle holding the end, so that it is fully covered.
    lo1, hi1 = np.searchsorted(c1.open_time, [s, -(-end // FIVE) * FIVE])
    co = c5.open[lo5:hi5].copy()
    ch = c5.high[lo5:hi5].copy()
    fo = c1.open[lo1:hi1].copy()
    fh = c1.high[lo1:hi1].copy()
    fl = c1.low[lo1:hi1].copy()
    fc = c1.close[lo1:hi1].copy()
    ft = c1.open_time[lo1:hi1]
    assert len(co), "the fill candle's 5m candle exists"
    assert c5.open_time[lo5] == s, "the fill candle's 5m candle exists"
    co[0] = price
    if fill_len == MINUTE:
        j = int(np.searchsorted(ft, fill_t))
        assert j < len(ft)
        assert ft[j] == fill_t
        fo[:j] = fh[:j] = fl[:j] = fc[:j] = price
        fo[j] = price
        if maker:
            fh[j] = max(price, fc[j])
    elif maker:
        ch[0] = max(price, float(c5.close[lo5]))
    ended = series.ended and hi5 == len(c5)
    coarse = Candles(
        Interval.M5, c5.open_time[lo5:hi5], co, ch, c5.low[lo5:hi5], c5.close[lo5:hi5], ended
    )
    fine = Candles(Interval.M1, ft, fo, fh, fl, fc, ended)
    return EntryWindow(coarse, fine, fill_t - s)


LABEL_SCHEMA = {
    "cid": pl.String,
    "buffer": pl.Float64,
    "scenario": pl.String,
    "exit": pl.String,
    "status": pl.String,  # done | censored
    "entry_t": pl.Int64,
    "entry_price": pl.Float64,
    "entry_kind": pl.String,
    "exit_t": pl.Int64,
    "exit_price": pl.Float64,
    "reason": pl.String,
}
FWD_SCHEMA = pl.Schema(
    {
        "cid": pl.String(),
        "buffer": pl.Float64(),
        "fwd": pl.Array(pl.Float32, FWD_STEPS),
        "mfe": pl.Float64(),
        "mae": pl.Float64(),
    }
)


def resolve_one(
    series: Series,
    w: EntryWindow,
    fill_t: int,
    price: float,
    level: float,
    u: float,
    tf: str,
    exit_name: str,
) -> tuple[str, int | None, float | None, str | None]:
    """(status, exit time, raw exit price, reason) of one exit from one fill."""
    spec = EXITS[exit_name]
    limit = exit_hold_us(tf, exit_name)
    deadline = fill_t + limit
    stop: float | None = None
    if spec.stop_below_level is not None:
        stop_abs = level * (1 - spec.stop_below_level * u)
        if stop_abs >= price:  # entry at or below the stop: it fills at once, at the entry
            return "done", fill_t, price, ExitReason.STOP.value
        if stop_abs > 0:
            stop = 1 - stop_abs / price
    plan = ExitPlan(
        take_profit=None if spec.take_profit is None else spec.take_profit * u,
        stop=stop,
        max_hold=timedelta(microseconds=limit + w.offset),
    )
    res = resolve_exit(w.coarse, plan, fine=w.fine)
    if res is None or (
        not series.ended and res.reason is ExitReason.TIME and deadline > series.known_until
    ):
        return "censored", None, None, None
    when = (res.time - EPOCH) // timedelta(microseconds=1)
    return "done", int(when), res.raw_price, res.reason.value


def forward_path(
    series: Series, w: EntryWindow, fill_t: int, price: float, u: float, tf: str
) -> tuple[list[float] | None, float | None, float | None]:
    """Closes at 32 steps over T after the fill, and the MFE / MAE over T, in units of u."""
    hold = hold_us(tf)
    if not series.ended and fill_t + hold > series.known_until:
        return None, None, None
    path = fill_path(w.coarse, w.fine)
    taus = fill_t + (hold * np.arange(1, FWD_STEPS + 1)) // FWD_STEPS
    # The close of the last candle opening before each step (the time-exit convention). The
    # flat candles before the fill open earlier, but every step is after the fill.
    idx = np.searchsorted(path.open_time, taus, side="left") - 1
    closes = path.close[np.maximum(idx, 0)]
    m = int(np.searchsorted(path.open_time, fill_t + hold, side="left"))
    fwd = ((closes / price - 1) / u).astype(np.float32).tolist()
    mfe = (float(path.high[:m].max()) / price - 1) / u
    mae = (float(path.low[:m].min()) / price - 1) / u
    return fwd, mfe, mae


def build_labels(
    cands: pl.DataFrame, fills: pl.DataFrame, series: Series
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Exit labels of every filled entry, and the forward paths."""
    info = {
        r["cid"]: r
        for r in cands.select("cid", "family", "placebo", "tf", "u").iter_rows(named=True)
    }
    rows: list[tuple[object, ...]] = []
    fwd_rows: list[tuple[object, ...]] = []
    done: dict[
        tuple[str, int, float, str, str], tuple[str, int | None, float | None, str | None]
    ] = {}
    filled = fills.filter(pl.col("status") == "filled")
    for f in filled.iter_rows(named=True):
        c = info[f["cid"]]
        tf, u, family, placebo = c["tf"], c["u"], c["family"], c["placebo"]
        b = f["buffer"]
        if placebo and family == "limit" and b != PRIMARY_BUFFER:
            continue
        names = exits_for(family, placebo)
        horizon = max(exit_hold_us(tf, e) for e in names)
        maker = f["kind"] == "maker"
        w: EntryWindow | None = None
        for e in names:
            key = (f["cid"], f["fill_t"], f["price"], f["kind"], e)
            if key not in done:
                if w is None:
                    w = entry_window(series, f["fill_t"], f["fill_len"], f["price"], maker, horizon)
                done[key] = resolve_one(
                    series, w, f["fill_t"], f["price"], f["level_at_fill"], u, tf, e
                )
            st, et, ep, reason = done[key]
            rows.append(
                (
                    f["cid"],
                    b,
                    f["scenario"],
                    e,
                    st,
                    f["fill_t"],
                    f["price"],
                    f["kind"],
                    et,
                    ep,
                    reason,
                )
            )
        primary = family == "touch" or b == PRIMARY_BUFFER
        if not placebo and primary and f["scenario"] == "base":
            if w is None:
                w = entry_window(series, f["fill_t"], f["fill_len"], f["price"], maker, horizon)
            fwd, mfe, mae = forward_path(series, w, f["fill_t"], f["price"], u, tf)
            fwd_rows.append((f["cid"], b, fwd, mfe, mae))
    labels = pl.DataFrame(rows, schema=LABEL_SCHEMA, orient="row")
    labels = with_net_returns(labels)
    return labels, pl.DataFrame(fwd_rows, schema=FWD_SCHEMA, orient="row")


def with_net_returns(labels: pl.DataFrame) -> pl.DataFrame:
    """Net return of each label under its scenario's costs (base: fee 0.1% per side, 0.1%
    slippage and latency on market fills; stress: 0.1% more per side)."""
    taker = pl.col("entry_kind") == "taker"
    market_exit = pl.col("reason").is_in(["stop", "time", "series_end", "trailing_stop"])
    extra = pl.when(pl.col("scenario") == "stress").then(STRESS_EXTRA_COST).otherwise(0.0)
    buy = pl.col("entry_price") * (1 + pl.when(taker).then(MARKET_SLIPPAGE).otherwise(0.0) + extra)
    sell = pl.col("exit_price") * (
        1 - pl.when(market_exit).then(MARKET_SLIPPAGE).otherwise(0.0) - extra
    )
    return labels.with_columns(net=sell * (1 - FEE) / (buy * (1 + FEE)) - 1)


def read_bars(symbol: str) -> dict[str, pl.DataFrame]:
    """The bars `build_bars.py` wrote for a coin."""
    from runner import part_path

    return {tf: pl.read_parquet(part_path(f"bars_{tf}", symbol)) for tf in TF_NAMES}
