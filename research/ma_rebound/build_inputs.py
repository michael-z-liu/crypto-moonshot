"""M4: model inputs per study candidate (both families), memory-mapped under `cache/m4/`.

    uv run --group research python build_inputs.py [--workers 8] [--symbols ...] [--force]

Two passes, each coin by coin in parallel processes, resumable (a coin is done when its marker
exists):

1. **Market breadth** on the 5m grid (2020-01-01 .. data end): per coin and 5m candle, whether it
   is in the liquid universe (`liq` of the day >= $1M), its 5m return (close over the previous
   candle's close, when that candle is exactly 5 minutes earlier) and whether its close is above
   its own 1d SMA50 level in force. The parts are summed in universe order into `market.npz`:
   the equal-weight mean return of the liquid coins (as a log index) and the share of liquid
   coins above their 1d SMA50. Every grid value at candle open time t uses only candles that
   closed by t + 5m (checked by `check_inputs.py`).
2. **Inputs**, written straight into preallocated memory maps (one row per study candidate, in
   the order of `rows.parquet`, which is `cache/m3/cands.parquet`'s order: family, trigger time,
   symbol, timeframe, length):
   - `fine.npy` float16 (rows, 14, 288): the last 288 5m candles ending with the trigger candle.
   - `coarse.npy` float16 (rows, 17, 256): the last 255 completed bars of the MA's timeframe and
     the forming bar as it stands at the decision (from the 5m candles up to the trigger; for 1w
     and 1M, the completed days of the bar from the 1d bars plus the forming day from 5m).
   - `scalars.npy` float64 (rows, len(SCALARS)) and `other_ma.npy` float64 (rows, 15).

Channels (`FINE_CH`, `COARSE_CH`): coin prices as `(price / L - 1) / u` (L, u the candidate's);
the level's own path `(L_bar / L - 1) / u`; volume as `log1p(quote volume)` (the model subtracts
the median of the window it uses, so "relative to its window median" holds for any slice); the
taker-buy share of the quote volume (fine only; 0.5 when there is no volume); validity masks
(1 where the candle / bar exists, where the level is defined). BTCUSDT's windows are aligned on
time: its last 288 5m candles closing by the decision, and its bars of the same timeframe with
its forming bar, as returns relative to BTC's last close at the decision in units of BTC's own
`atr_d x sqrt(T)` (BTC's `atr_d` on the trigger day, T the candidate's hold). Breadth: the log
index relative to its value at the trigger candle, and the share above SMA50; in the coarse
window sampled at each bar's last 5m candle (the forming bar: at the trigger), masked before
2020 (no 5m grid). Missing positions are 0 with mask 0. Values are clipped to +-60,000 before the
cast to float16 (`check_inputs.py` measures the precision).

Scalars (raw, the model transforms them): timeframe index, MA length, `distance`,
`bars_since_touch`, `slope10`, `atr_d`, `liq` (NaN for nulls), `u` and BTC's u. `other_ma`: the
signed distance `(close / L_k - 1) / u` from the trigger close to each of the coin's 15 study
levels in force at the trigger (NaN where a level does not exist).

Nothing here reads a candle that closed after the candidate's decision time; the truncation check
in `check_inputs.py` rebuilds sampled candidates from candles cut at their decision.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from common import (
    CACHE,
    KLINES_5M,
    MIN_DAILY_QUOTE_VOLUME,
    STORE_START,
    TIMEFRAMES,
    ma_names,
    universe,
)
from events import (
    FIVE,
    PRICE_COLS,
    TF_NAMES,
    bar_index,
    bar_start,
    build_bars,
    level_array,
    positions,
    read_daily_pre2020,
)
from runner import END_US

type F64 = npt.NDArray[np.float64]
type F32 = npt.NDArray[np.float32]
type F16 = npt.NDArray[np.float16]
type I64 = npt.NDArray[np.int64]
type Bools = npt.NDArray[np.bool_]

M4 = CACHE / "m4"
PARTS = M4 / "parts"
FINE = 288
COARSE = 256
CLIP16 = 60_000.0
BTC = "BTCUSDT"
GRID0 = int(STORE_START.timestamp()) * 1_000_000
GRID_N = (END_US - GRID0) // FIVE

FINE_CH = (
    "open",
    "high",
    "low",
    "close",
    "logqv",
    "taker",
    "mask",
    "btc_open",
    "btc_high",
    "btc_low",
    "btc_close",
    "btc_logqv",
    "mkt_index",
    "mkt_above",
)
COARSE_CH = (
    "open",
    "high",
    "low",
    "close",
    "level",
    "logqv",
    "mask",
    "level_mask",
    "btc_open",
    "btc_high",
    "btc_low",
    "btc_close",
    "btc_logqv",
    "btc_mask",
    "mkt_index",
    "mkt_above",
    "mkt_mask",
)
SCALARS = ("tf", "n", "distance", "bars_since_touch", "slope10", "atr_d", "liq", "u", "btc_u")
ROW_COLS = [
    "cid",
    "family",
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
    "distance",
    "bars_since_touch",
    "slope10",
]


def study_mas() -> list[tuple[str, int]]:
    return ma_names()


# --- reading ---------------------------------------------------------------------------------


def read_5m_full(symbol: str, end_us: int) -> pl.DataFrame:
    """A coin's 5m candles that closed by `end_us`: `t`, OHLC, `quote_volume`, `taker_qv`."""
    folder = KLINES_5M / f"symbol={symbol}"
    last = str(np.datetime64(end_us - 1, "us").astype("datetime64[M]"))
    files = [f for f in sorted(folder.glob("*.parquet")) if f.stem <= last]
    cols = ["open_time", *PRICE_COLS, "quote_volume", "taker_buy_quote_volume"]
    if not files:
        schema = {"t": pl.Int64, **dict.fromkeys([*PRICE_COLS, "quote_volume"], pl.Float64)}
        return pl.DataFrame(schema={**schema, "taker_qv": pl.Float64})
    df = pl.read_parquet(files, columns=cols)
    df = df.select(
        pl.col("open_time").dt.epoch("us").alias("t"),
        *PRICE_COLS,
        "quote_volume",
        pl.col("taker_buy_quote_volume").alias("taker_qv"),
    )
    return df.filter(pl.col("t") + FIVE <= end_us).sort("t")


@dataclass(frozen=True)
class Coin:
    """One coin's candles and bars cut at `end_us` (candles that closed by then)."""

    symbol: str
    c5: pl.DataFrame
    bars: dict[str, pl.DataFrame]
    running: dict[str, dict[str, F64]]  # tf -> forming-bar aggregates after each 5m candle


def load_coin(symbol: str, end_us: int) -> Coin:
    c5 = read_5m_full(symbol, end_us)
    bars = build_bars(c5, read_daily_pre2020(symbol, end_us))
    return Coin(symbol, c5, bars, running_bars(c5, bars))


def running_bars(c5: pl.DataFrame, bars: dict[str, pl.DataFrame]) -> dict[str, dict[str, F64]]:
    """For each 5m candle j and timeframe: the forming bar's open, high, low and quote volume
    after candle j (its close is candle j's close). 1h/4h/1d from the 5m candles of the bar; 1w
    and 1M from the completed 1d bars of the bar before j's day, plus j's forming day."""
    t = c5["t"].to_numpy().astype(np.int64)
    out: dict[str, dict[str, F64]] = {}
    base = c5.select("open", "high", "low", "quote_volume")
    for tf in ("1h", "4h", "1d"):
        g = bar_index(t, tf)
        r = base.with_columns(g=pl.Series(g)).select(
            o=pl.col("open").first().over("g"),
            h=pl.col("high").cum_max().over("g"),
            lo=pl.col("low").cum_min().over("g"),
            q=pl.col("quote_volume").cum_sum().over("g"),
        )
        out[tf] = {k: r[k].to_numpy().astype(np.float64) for k in ("o", "h", "lo", "q")}
    day = bars["1d"]
    pos_d = positions(day, t, "1d")
    dfd = day.select("idx", "t", "open", "high", "low", "quote_volume")
    for tf in ("1w", "1M"):
        g = bar_index(dfd["t"].to_numpy().astype(np.int64), tf)
        # aggregates of the bar's completed days strictly before each day
        prev = dfd.with_columns(g=pl.Series(g)).select(
            o=pl.col("open").first().over("g"),
            first=(pl.int_range(pl.len()).over("g") == 0),
            h=pl.col("high").cum_max().shift(1).over("g"),
            lo=pl.col("low").cum_min().shift(1).over("g"),
            q=pl.col("quote_volume").cum_sum().shift(1).over("g"),
        )
        first = prev["first"].to_numpy()[pos_d]
        po = prev["o"].to_numpy()[pos_d]
        ph = prev["h"].fill_null(-np.inf).to_numpy()[pos_d]
        pq = prev["q"].fill_null(0.0).to_numpy()[pos_d]
        plo = prev["lo"].fill_null(np.inf).to_numpy()[pos_d]
        d = out["1d"]
        out[tf] = {
            "o": np.where(first, d["o"], po),
            "h": np.maximum(ph, d["h"]),
            "lo": np.minimum(plo, d["lo"]),
            "q": pq + d["q"],
        }
    return out


# --- market breadth --------------------------------------------------------------------------


def market_part(symbol: str, end_us: int) -> pl.DataFrame:
    """One coin's breadth contributions on the 5m grid, liquid candles only: grid index `g`,
    `r` (5m return, NaN without the previous candle 5 minutes earlier), `above` (null when the
    1d SMA50 level is undefined)."""
    c5 = read_5m_full(symbol, end_us)
    bars = build_bars(c5, read_daily_pre2020(symbol, end_us))
    t = c5["t"].to_numpy().astype(np.int64)
    close = c5["close"].to_numpy()
    pos_d = positions(bars["1d"], t, "1d")
    liq = bars["1d"]["liq"].fill_null(np.nan).to_numpy()[pos_d]
    l50 = level_array(bars["1d"], 50)[pos_d]
    prev_ok = np.concatenate(([False], np.diff(t) == FIVE))
    prev_close = np.concatenate(([np.nan], close[:-1]))
    r = np.where(prev_ok, close / prev_close - 1, np.nan)
    keep = (liq >= MIN_DAILY_QUOTE_VOLUME) & (t >= GRID0)
    above = np.where(np.isfinite(l50), (close > l50).astype(np.float64), np.nan)
    return pl.DataFrame(
        {"g": (t[keep] - GRID0) // FIVE, "r": r[keep], "above": above[keep]},
        schema={"g": pl.Int64, "r": pl.Float64, "above": pl.Float64},
    )


@dataclass(frozen=True)
class Market:
    """Breadth on the 5m grid: `index[g]` cumulative log of (1 + mean return of the liquid coins)
    up to and including the candle opening at grid point g; `above[g]` share of liquid coins
    above their 1d SMA50 (NaN with none); `n_liquid[g]`."""

    index: F64
    above: F64
    n_liquid: F64

    def cut(self, end_us: int) -> Market:
        """The grid points whose candle closed by `end_us`."""
        k = max(0, (end_us - GRID0) // FIVE)
        return Market(self.index[:k], self.above[:k], self.n_liquid[:k])


def sum_market(parts: list[pl.DataFrame]) -> Market:
    """Sum the coins' contributions in the given order (deterministic)."""
    s_r = np.zeros(GRID_N)
    n_r = np.zeros(GRID_N)
    s_a = np.zeros(GRID_N)
    n_a = np.zeros(GRID_N)
    n_l = np.zeros(GRID_N)
    for p in parts:
        g = p["g"].to_numpy()
        r = p["r"].to_numpy()
        a = p["above"].to_numpy()
        ok = np.isfinite(r)
        s_r[g[ok]] += r[ok]
        n_r[g[ok]] += 1
        oa = np.isfinite(a)
        s_a[g[oa]] += a[oa]
        n_a[g[oa]] += 1
        n_l[g] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_r = np.where(n_r > 0, s_r / np.maximum(n_r, 1), 0.0)
        above = np.where(n_a > 0, s_a / np.maximum(n_a, 1), np.nan)
    return Market(np.cumsum(np.log1p(mean_r)), above, n_l)


@cache
def market() -> Market:
    z = np.load(M4 / "market.npz")
    return Market(z["index"], z["above"], z["n_liquid"])


# --- inputs of one coin ----------------------------------------------------------------------


def _windows(end: I64, length: int) -> tuple[I64, Bools]:
    """Indices end-length+1 .. end per row, and whether each is >= 0."""
    idx = end[:, None] + np.arange(-length + 1, 1)[None, :]
    return np.maximum(idx, 0), idx >= 0


@dataclass(frozen=True)
class Inputs:
    fine: F32  # (k, len(FINE_CH), FINE)
    coarse: F32  # (k, len(COARSE_CH), COARSE)
    scalars: F64  # (k, len(SCALARS))
    other_ma: F64  # (k, 15)


type NDStr = npt.NDArray[np.str_]


def _btc_u(btc: Coin, trigger_t: I64, tf: NDStr) -> F64:
    pos_d = positions(btc.bars["1d"], trigger_t, "1d")
    atr = btc.bars["1d"]["atr_d"].fill_null(np.nan).to_numpy()
    a = np.where(pos_d >= 0, atr[np.maximum(pos_d, 0)], np.nan)
    hold = np.array([math.sqrt(TIMEFRAMES[x].hold_days) for x in tf])
    return a * hold


def coin_inputs(coin: Coin, rows: pl.DataFrame, btc: Coin, mkt: Market) -> Inputs:
    """Inputs of `rows` (candidates of `coin`, `ROW_COLS`), from data cut anywhere at or after
    each row's decision."""
    k = rows.height
    fine = np.zeros((k, len(FINE_CH), FINE), dtype=np.float32)
    coarse = np.zeros((k, len(COARSE_CH), COARSE), dtype=np.float32)
    trig = rows["trigger_t"].to_numpy().astype(np.int64)
    L = rows["L"].to_numpy()
    u = rows["u"].to_numpy()
    tf_arr = rows["tf"].to_numpy()
    n_arr = rows["n"].to_numpy()
    c5 = coin.c5
    t = c5["t"].to_numpy().astype(np.int64)
    j = np.searchsorted(t, trig)
    assert (j < len(t)).all(), "trigger candle"
    assert (t[np.minimum(j, len(t) - 1)] == trig).all(), "trigger candle"
    px = {c: c5[c].to_numpy() for c in PRICE_COLS}
    qv = c5["quote_volume"].to_numpy()
    tq = c5["taker_qv"].to_numpy()

    # fine, coin
    idx, ok = _windows(j, FINE)
    scale = (1 / u)[:, None]
    for ci, c in enumerate(PRICE_COLS):
        fine[:, ci] = (px[c][idx] / L[:, None] - 1) * scale
    fine[:, 4] = np.log1p(qv[idx])
    q = qv[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        fine[:, 5] = np.where(q > 0, tq[idx] / np.where(q > 0, q, 1), 0.5)
    fine[:, 6] = 1.0
    fine[:, :7] *= ok[:, None, :]

    # fine, BTC (aligned on time: its last candles closing by the decision)
    bt = btc.c5["t"].to_numpy().astype(np.int64)
    jb = np.searchsorted(bt, trig, side="right") - 1
    assert (jb >= 0).all(), "BTC has a candle by the decision"
    bpx = {c: btc.c5[c].to_numpy() for c in PRICE_COLS}
    bqv = btc.c5["quote_volume"].to_numpy()
    ub = _btc_u(btc, trig, tf_arr)
    ref = bpx["close"][jb]
    idx_b, ok_b = _windows(jb, FINE)
    bscale = (1 / ub)[:, None]
    for ci, c in enumerate(PRICE_COLS):
        fine[:, 7 + ci] = (bpx[c][idx_b] / ref[:, None] - 1) * bscale
    fine[:, 11] = np.log1p(bqv[idx_b])
    fine[:, 7:12] *= ok_b[:, None, :]

    # fine, breadth (grid points up to the trigger candle)
    g_trig = (trig - GRID0) // FIVE
    gi, gok = _windows(g_trig, FINE)
    gok &= gi < len(mkt.index)
    gi = np.minimum(gi, max(len(mkt.index) - 1, 0))
    assert (g_trig < len(mkt.index)).all(), "breadth known at the trigger"
    m_ref = mkt.index[g_trig]
    fine[:, 12] = np.where(gok, mkt.index[gi] - m_ref[:, None], 0.0)
    fine[:, 13] = np.where(gok, np.nan_to_num(mkt.above[gi], nan=0.0), 0.0)

    # coarse, per timeframe
    for tf in TF_NAMES:
        sel = np.flatnonzero(tf_arr == tf)
        if len(sel) == 0:
            continue
        _coarse_tf(coarse, sel, tf, coin, j[sel], L[sel], u[sel], n_arr[sel], mkt)
        _coarse_btc(coarse, sel, tf, btc, jb[sel], ref[sel], ub[sel])

    scal = np.column_stack(
        [
            np.array([TF_NAMES.index(x) for x in tf_arr], dtype=np.float64),
            n_arr.astype(np.float64),
            rows["distance"].to_numpy(),
            rows["bars_since_touch"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
            rows["slope10"].fill_null(np.nan).to_numpy(),
            rows["atr_d"].to_numpy(),
            rows["liq"].to_numpy(),
            u,
            ub,
        ]
    )
    close_j = px["close"][j]
    other = np.full((k, 15), np.nan)
    for mi, (tf2, n2) in enumerate(study_mas()):
        b2 = coin.bars[tf2]
        p2 = positions(b2, trig, tf2)
        lev = level_array(b2, n2)
        lv = np.where(p2 >= 0, lev[np.maximum(p2, 0)], np.nan)
        other[:, mi] = (close_j / lv - 1) / u
    np.clip(fine, -CLIP16, CLIP16, out=fine)
    np.clip(coarse, -CLIP16, CLIP16, out=coarse)
    return Inputs(fine, coarse, scal, other)


def _bar_window(
    coin: Coin, tf: str, j: I64
) -> tuple[pl.DataFrame, I64, I64, Bools, dict[str, F64]]:
    """Bars table, forming-bar position p per row, completed-bar indices (rows, COARSE - 1) with
    validity, and the forming aggregates at candle j."""
    b = coin.bars[tf]
    t = coin.c5["t"].to_numpy().astype(np.int64)
    p = positions(b, t[j], tf)
    assert (p >= 0).all(), "the trigger's bar exists"
    idx, ok = _windows(p - 1, COARSE - 1)
    run = {key: v[j] for key, v in coin.running[tf].items()}
    return b, p, idx, ok, run


def _coarse_tf(
    out: F32, sel: I64, tf: str, coin: Coin, j: I64, L: F64, u: F64, n: I64, mkt: Market
) -> None:
    b, _, idx, ok, run = _bar_window(coin, tf, j)
    close_j = coin.c5["close"].to_numpy()[j]
    form = {"open": run["o"], "high": run["h"], "low": run["lo"], "close": close_j}
    scale = (1 / u)[:, None]
    for ci, c in enumerate(PRICE_COLS):
        arr = b[c].to_numpy()
        vals = np.concatenate([arr[idx], form[c][:, None]], axis=1)
        out[sel, ci] = (vals / L[:, None] - 1) * scale
    lev = np.full((len(sel), COARSE), np.nan)
    for nn_ in np.unique(n):
        r = np.flatnonzero(n == nn_)
        la = level_array(b, int(nn_))
        lev[r, :-1] = la[idx[r]]
    lev[:, -1] = L
    valid = np.concatenate([ok, np.ones((len(sel), 1), dtype=bool)], axis=1)
    lvl_ok = valid & np.isfinite(lev)
    out[sel, 4] = np.where(lvl_ok, (lev / L[:, None] - 1) * scale, 0.0)
    qv = b["quote_volume"].to_numpy()
    out[sel, 5] = np.log1p(np.concatenate([qv[idx], run["q"][:, None]], axis=1))
    out[sel, 6] = valid
    out[sel, 7] = lvl_ok
    out[sel, :6] *= valid[:, None, :]
    # breadth at each bar's last 5m candle (the forming bar: the trigger candle)
    # the end of bar i is the start of the next period (not the next existing bar)
    bar_end = bar_start(b["idx"].to_numpy().astype(np.int64) + 1, tf)
    g_trig = (coin.c5["t"].to_numpy()[j] - GRID0) // FIVE
    g_last = np.concatenate([(bar_end[idx] - FIVE - GRID0) // FIVE, g_trig[:, None]], axis=1)
    g_ok = valid & (g_last >= 0) & (g_last < len(mkt.index))
    gi = np.clip(g_last, 0, max(len(mkt.index) - 1, 0))
    ref = mkt.index[gi[:, -1]]
    out[sel, 14] = np.where(g_ok, mkt.index[gi] - ref[:, None], 0.0)
    out[sel, 15] = np.where(g_ok, np.nan_to_num(mkt.above[gi], nan=0.0), 0.0)
    out[sel, 16] = g_ok


def _coarse_btc(out: F32, sel: I64, tf: str, btc: Coin, jb: I64, ref: F64, ub: F64) -> None:
    b, _, idx, ok, run = _bar_window(btc, tf, jb)
    close_j = btc.c5["close"].to_numpy()[jb]
    form = {"open": run["o"], "high": run["h"], "low": run["lo"], "close": close_j}
    scale = (1 / ub)[:, None]
    valid = np.concatenate([ok, np.ones((len(sel), 1), dtype=bool)], axis=1)
    for ci, c in enumerate(PRICE_COLS):
        arr = b[c].to_numpy()
        vals = np.concatenate([arr[idx], form[c][:, None]], axis=1)
        out[sel, 8 + ci] = np.where(valid, (vals / ref[:, None] - 1) * scale, 0.0)
    qv = b["quote_volume"].to_numpy()
    out[sel, 12] = np.where(
        valid, np.log1p(np.concatenate([qv[idx], run["q"][:, None]], axis=1)), 0.0
    )
    out[sel, 13] = valid


# --- the build -------------------------------------------------------------------------------


def make_rows() -> pl.DataFrame:
    """Study candidates of both families in `cache/m3/cands.parquet`'s order, row = `cand`."""
    cands = (
        pl.read_parquet(CACHE / "candidates.parquet", columns=[*ROW_COLS, "placebo"])
        .filter(~pl.col("placebo"))
        .drop("placebo")
        .sort("family", "trigger_t", "symbol", "tf", "n")
        .with_row_index("row")
    )
    m3 = CACHE / "m3" / "cands.parquet"
    if m3.exists():
        ref = pl.read_parquet(m3, columns=["cid"])["cid"]
        assert ref.equals(cands["cid"]), "rows follow cache/m3/cands.parquet"
    return cands


@cache
def rows() -> pl.DataFrame:
    return pl.read_parquet(M4 / "rows.parquet")


def open_arrays(mode: str) -> dict[str, np.memmap[tuple[int, ...], np.dtype[np.generic]]]:
    n = rows().height
    shapes: dict[str, tuple[tuple[int, ...], type]] = {
        "fine": ((n, len(FINE_CH), FINE), np.float16),
        "coarse": ((n, len(COARSE_CH), COARSE), np.float16),
        "scalars": ((n, len(SCALARS)), np.float64),
        "other_ma": ((n, 15), np.float64),
    }
    out = {}
    for name, (shape, dtype) in shapes.items():
        path = M4 / f"{name}.npy"
        if mode == "w+" or not path.exists():
            out[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        else:
            out[name] = np.lib.format.open_memmap(path, mode=mode)
            assert out[name].shape == shape, f"{name}: shape {out[name].shape}, expected {shape}"
    return out


@cache
def _btc_full() -> Coin:
    return load_coin(BTC, END_US)


def _done(stage: str, symbol: str) -> Path:
    return PARTS / f"done_{stage}" / symbol


def _market_work(symbol: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    part = market_part(symbol, END_US)
    path = PARTS / "market" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    part.write_parquet(tmp)
    tmp.replace(path)
    return symbol, time.perf_counter() - t0


def _inputs_work(symbol: str) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    r = rows().filter(pl.col("symbol") == symbol)
    if r.height:
        coin = _btc_full() if symbol == BTC else load_coin(symbol, END_US)
        inp = coin_inputs(coin, r, _btc_full(), market())
        arrs = open_arrays("r+")
        k = r["row"].to_numpy()
        order = np.argsort(k)  # write rows in file order
        k = k[order]
        arrs["fine"][k] = inp.fine[order].astype(np.float16)
        arrs["coarse"][k] = inp.coarse[order].astype(np.float16)
        arrs["scalars"][k] = inp.scalars[order]
        arrs["other_ma"][k] = inp.other_ma[order]
        for a in arrs.values():
            a.flush()
    marker = _done("inputs", symbol)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return symbol, r.height, time.perf_counter() - t0


def _pool(fn: object, symbols: list[str], workers: int, label: str) -> None:
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        futures = [pool.submit(fn, s) for s in symbols]  # type: ignore[arg-type]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if i % 25 == 0 or i == len(symbols):
                print(f"{label} [{i}/{len(symbols)}] {res} ({time.perf_counter() - t0:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="M4 model inputs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--symbols", nargs="*", help="only these coins (inputs pass)")
    parser.add_argument("--force", action="store_true", help="rebuild everything")
    args = parser.parse_args()
    M4.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    symbols = universe()

    # 1. market breadth
    if args.force or not (M4 / "market.npz").exists():
        todo = [
            s for s in symbols if args.force or not (PARTS / "market" / f"{s}.parquet").exists()
        ]
        print(f"market: {len(todo)} of {len(symbols)} coins to build", flush=True)
        if todo:
            _pool(_market_work, todo, args.workers, "market")
        parts = [pl.read_parquet(PARTS / "market" / f"{s}.parquet") for s in symbols]
        mk = sum_market(parts)
        np.savez(M4 / "market.npz", index=mk.index, above=mk.above, n_liquid=mk.n_liquid)
        print(f"market.npz written ({time.perf_counter() - t0:.0f}s)", flush=True)

    # 2. inputs
    if args.force or not (M4 / "rows.parquet").exists():
        make_rows().write_parquet(M4 / "rows.parquet")
        rows.cache_clear()
        open_arrays("w+")
        for f in (PARTS / "done_inputs").glob("*"):
            f.unlink()
    else:
        open_arrays("r+")
    with_cands = set(rows()["symbol"].unique().to_list())
    pick = args.symbols or [s for s in symbols if s in with_cands]
    todo = [s for s in pick if not _done("inputs", s).exists()]
    print(f"inputs: {len(todo)} of {len(pick)} coins to build, {rows().height} rows", flush=True)
    if todo:
        _pool(_inputs_work, todo, args.workers, "inputs")
    left = [s for s in with_cands if not _done("inputs", s).exists()]
    print(f"done in {time.perf_counter() - t0:.0f}s; coins still missing: {len(left)}", flush=True)


if __name__ == "__main__":
    main()
