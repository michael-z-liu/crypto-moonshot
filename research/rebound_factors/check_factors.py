"""M2: checks of the factors, and their coverage on discovery entries.

    uv run --group research python check_factors.py [--logic-only] [--workers 8]

1. Logic, on synthetic candles (no data needed): daily bars from 5m candles, hand-computed
   values (returns, SMA distance, taker-buy shares, the habit frequency with its
   window-ended-before-D rule, the moonshot share, breadth, liq rank, market volume, the MA
   touch flag, ETH/BTC), and look-ahead: candles cut at D or changed at/after D change no
   factor.
2. Real entries: a fixed-seed sample of 48 entries across every year (BTC and delisted coins
   included). Every coin and BTC/ETH factor is recomputed from the coin's, BTC's and ETH's
   candles cut at D (open time < D), and again from the full candles changed at and after D;
   both must equal `factors.parquet`. `c_log_liq` and `c_log_age` are recomputed through
   `build_panel.panel_rows` on the cut candles.
3. Real days: for five days, every universe coin's candles are cut at D; the day's panel
   membership, every coin factor of every member, the breadth factors, `c_liq_rank` and
   `m_qv_7_90` are recomputed and compared.
4. The table: keys equal the panel's, Float64 factors, and on discovery rows no factor is
   constant or more than 60% null.
5. Writes `results/m2_coverage.md` and `.csv`: per factor non-null share, mean, std and the
   1/50/99th percentiles on discovery rows (factor values only).

Never opens `cache/sealed/` and reads no label. Exits non-zero on a failure.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import numpy as np
import polars as pl
from build_factors import (
    BREADTH_FACTORS,
    BTC,
    BTC_FACTORS,
    ETH,
    FACTORS_LIST,
    HABIT_SPAN,
    MOON_DAYS,
    NAMES,
    SERIES_FACTORS,
    Bars,
    at_days,
    bars_from_candles,
    breadth,
    coin_rows,
    coin_series,
    daily_bars,
    market_series,
    market_volume,
    panel_factors,
)
from build_panel import panel_rows
from common import (
    DATA_DIR,
    DAY,
    DISCOVERY_END_US,
    F64,
    FACTORS,
    FIVE,
    PANEL,
    RESULTS,
    first_day_pre2020,
    read_5m,
    universe,
    us,
    utc,
)

FAILURES: list[str] = []
REL = 1e-9
SEED = 20261114
MAX_NULL = 0.60

type Val = float | None


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  ' + detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def same(a: Val, b: Val) -> bool:
    a = None if a is None or (isinstance(a, float) and math.isnan(a)) else a
    b = None if b is None or (isinstance(b, float) and math.isnan(b)) else b
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= REL * max(abs(a), abs(b), 1e-12)


def diff_rows(a: dict[str, Val], b: dict[str, Val], cols: list[str]) -> list[str]:
    return [f"{k}: {a.get(k)!r} vs {b.get(k)!r}" for k in cols if not same(a.get(k), b.get(k))]


# --- 1. logic on synthetic candles -------------------------------------------------------------

DAY0 = 18_600  # 2020-12-05, a UTC day number
type Day = tuple[float, float, float, float, float, float, float]  # o, h, l, c, qv, tc, tb


def synth(days: list[Day], day0: int = DAY0) -> pl.DataFrame:
    """Two 5m candles per day whose day bar is exactly the given (o, h, l, c, qv, tc, tb)."""
    rows: list[tuple[int, float, float, float, float, float, float, float]] = []
    for k, (o, h, lo, c, qv, tc, tb) in enumerate(days):
        t0 = (day0 + k) * DAY
        rows.append((t0, o, h, min(o, c), c, 0.25 * qv, 0.5 * tc, 0.25 * tb))
        rows.append((t0 + DAY - FIVE, c, c, lo, c, 0.75 * qv, 0.5 * tc, 0.75 * tb))
    cols = ["t", "open", "high", "low", "close", "quote_volume", "trade_count"]
    df = pl.DataFrame(rows, schema=[*cols, "taker_buy_quote_volume"], orient="row")
    return df.with_columns(pl.col("t").cast(pl.Int64))


def flat_days(n: int, px: float = 100.0, qv: float = 1000.0) -> list[Day]:
    return [(px, px + 1, px - 1, px, qv, 10.0, 0.5 * qv)] * n


def at(series: dict[str, F64], b: Bars, d_us: int) -> dict[str, Val]:
    df = at_days(series, b.d0, b.n, np.array([d_us], dtype=np.int64))
    return df.row(0, named=True)


def expect(name: str, got: dict[str, Val], want: dict[str, Val]) -> None:
    bad = [f"{k}: got {got.get(k)!r} want {v!r}" for k, v in want.items() if not same(got[k], v)]
    check(name, not bad, ", ".join(bad))


def logic_bars() -> None:
    # Daily bars aggregate 5m candles; a missing day is flat at the last close with no volume.
    t0 = DAY0 * DAY
    c5 = pl.DataFrame(
        {
            "t": [t0, t0 + FIVE, t0 + 2 * FIVE, t0 + 2 * DAY + FIVE],
            "open": [10.0, 11.0, 12.0, 20.0],
            "high": [11.0, 15.0, 13.0, 21.0],
            "low": [9.0, 10.5, 8.0, 19.0],
            "close": [11.0, 12.0, 12.5, 20.5],
            "quote_volume": [1.0, 2.0, 3.0, 4.0],
            "trade_count": [1.0, 1.0, 2.0, 5.0],
            "taker_buy_quote_volume": [0.5, 1.0, 2.0, 1.0],
        }
    )
    b = daily_bars(c5)
    ok = (
        b.d0 == DAY0
        and b.n == 3
        and b.open.tolist() == [10.0, 12.5, 20.0]
        and b.high.tolist() == [15.0, 12.5, 21.0]
        and b.low.tolist() == [8.0, 12.5, 19.0]
        and b.close.tolist() == [12.5, 12.5, 20.5]
        and b.qv.tolist() == [6.0, 0.0, 4.0]
        and b.tc.tolist() == [4.0, 0.0, 5.0]
        and b.tb.tolist() == [3.5, 0.0, 1.0]
    )
    check("daily bars from 5m candles (incl. a missing day)", ok, str(b))


def logic_values() -> None:
    rng = np.random.default_rng(SEED)
    n = 420
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    days: list[Day] = []
    for k in range(n):
        o = c[k - 1] if k else c[0]
        vol = float(rng.uniform(1e5, 1e6))
        days.append(
            (
                float(o),
                float(max(o, c[k]) * 1.02),
                float(min(o, c[k]) * 0.98),
                float(c[k]),
                vol,
                float(rng.integers(10, 1000)),
                float(vol * rng.uniform(0.4, 0.6)),
            )
        )
    c5 = synth(days)
    b = daily_bars(c5)
    btc = daily_bars(synth(flat_days(n + 5), DAY0 - 5))
    d = (DAY0 + n) * DAY  # D: the day after the last bar
    got = at(coin_series(b, btc), b, d)
    cl = [x[3] for x in days]
    qv = [x[4] for x in days]
    tb = [x[6] for x in days]
    tc = [x[5] for x in days]
    want: dict[str, Val] = {
        "c_ret1": cl[-1] / cl[-2] - 1,
        "c_ret7": cl[-1] / cl[-8] - 1,
        "c_ret90": cl[-1] / cl[-91] - 1,
        "c_sma20": cl[-1] / (sum(cl[-20:]) / 20) - 1,
        "c_dd30": cl[-1] / max(x[1] for x in days[-30:]) - 1,
        "c_tb1": tb[-1] / qv[-1],
        "c_tb7": sum(tb[-7:]) / sum(qv[-7:]),
        "c_tb_chg": sum(tb[-7:]) / sum(qv[-7:]) - sum(tb[-30:]) / sum(qv[-30:]),
        "c_qv_1_30": qv[-1] / (sum(qv[-31:-1]) / 30),
        "c_ats_7_30": (sum(qv[-7:]) / sum(tc[-7:])) / (sum(qv[-30:]) / sum(tc[-30:])),
        "c_up10": float(sum(cl[-k] > cl[-k - 1] for k in range(1, 11))),
        "c_upvol14": sum(qv[-k] for k in range(1, 15) if cl[-k] > cl[-k - 1]) / sum(qv[-14:]),
        "c_rs7": cl[-1] / cl[-8] - 1,  # BTC flat
        "c_beta60": None,  # BTC flat: zero variance, no beta
        "c_below_200d": float(cl[-1] < sum(cl[-200:]) / 200),
    }
    lr = np.diff(np.log(np.array(cl)))
    want["c_vol7"] = float(np.std(lr[-7:], ddof=1))
    expect("hand values: returns, SMA, drawdown, taker-buy, volume, vol, up days", got, want)

    # Look-ahead on synthetic candles: cut at D, and changed at/after D.
    extra = synth(flat_days(40, 1e6, 1e12), DAY0 + n)
    full = pl.concat([c5, extra])
    a = at(coin_series(bars_from_candles(full), btc), daily_bars(full), d)
    cut = bars_from_candles(full, d)
    trunc = at(coin_series(cut, btc), cut, d)
    bad = diff_rows(got, a, SERIES_FACTORS) + diff_rows(got, trunc, SERIES_FACTORS)
    check("look-ahead: candles at/after D change no coin factor (synthetic)", not bad, str(bad))


def logic_habit() -> None:
    # Flat at 100 with highs at 101; one spike day j with high 140 (+40%).
    n, j = 400, 300
    days = flat_days(n)
    days[j] = (100.0, 140.0, 99.0, 100.0, 1000.0, 10.0, 500.0)
    b = daily_bars(synth(days))
    btc = b
    s = coin_series(b, btc)
    span = HABIT_SPAN - MOON_DAYS + 1  # 336 windows, b in [i-365, i-30]

    def hab(i: int) -> Val:
        return at(s, b, (DAY0 + i + 1) * DAY)["c_habit"]

    def moon(i: int) -> Val:
        return at(s, b, (DAY0 + i + 1) * DAY)["_moon30"]

    # The spike is in the windows of entries b = j-30 .. j-1 (window bars b+1 .. b+30).
    # At i = j (D = the day after the spike) only b = j-30 has ended: 1 hit.
    # At i = j+29 all 30 have ended; at i = j+28, 29 of them. At i = j-1, none.
    # Windows counted at i: b from max(0, i-365) to i-30.
    def cnt(i: int) -> int:
        return i - 30 - max(0, i - 365) + 1

    ok = (
        same(hab(j - 1), 0.0)
        and same(hab(j), 1 / cnt(j))
        and same(hab(j + 28), 29 / cnt(j + 28))
        and same(hab(j + 29), 30 / cnt(j + 29))
        and same(hab(399), 30 / span)
        and same(moon(j - 1), 0.0)
        and same(moon(j), 1.0)
        and same(moon(j + 29), 1.0)
        and same(moon(j + 30), 0.0)
        and hab(140) is None  # 111 ended windows < HABIT_MIN
        and hab(150) is not None
    )
    check(
        "habit: only windows ended by D count (1, 29, 30 hits), moon30 is the last ended one",
        ok,
        f"{hab(j - 1)} {hab(j)} {hab(j + 28)} {hab(j + 29)} {hab(399)} {moon(j)} {hab(140)}",
    )
    # The spike on day D itself (bar j, entry at D = day j) changes nothing.
    cut = daily_bars(synth(days[:j]))
    h_cut = at(coin_series(cut, cut), cut, (DAY0 + j) * DAY)
    h_full = at(s, b, (DAY0 + j) * DAY)
    check(
        "habit: a spike on day D does not count for the entry at D",
        not diff_rows(h_cut, h_full, ["c_habit", "_moon30"]),
    )


def logic_touch() -> None:
    # Rising closes 100 + k, lows 0.5 under the close: lows stay above the SMA50.
    n = 80
    days: list[Day] = [
        (100.0 + k, 100.5 + k, 99.5 + k, 100.0 + k, 1e6, 10.0, 5e5) for k in range(n)
    ]
    sma50_prev = sum(100.0 + k for k in range(n - 51, n - 1)) / 50
    days[-1] = (179.0, 180.0, sma50_prev - 0.01, 179.0, 1e6, 10.0, 5e5)
    b = daily_bars(synth(days))
    r = at(coin_series(b, b), b, (DAY0 + n) * DAY)
    days2 = list(days)
    days2[-5] = (175.0, 176.0, 100.0, 175.0, 1e6, 10.0, 5e5)  # an earlier dip below the SMA
    b2 = daily_bars(synth(days2))
    r2 = at(coin_series(b2, b2), b2, (DAY0 + n) * DAY)
    days3 = list(days)
    days3[-1] = (179.0, 180.0, sma50_prev + 0.01, 179.0, 1e6, 10.0, 5e5)  # just above
    b3 = daily_bars(synth(days3))
    r3 = at(coin_series(b3, b3), b3, (DAY0 + n) * DAY)
    short = daily_bars(synth(days[-55:], DAY0 + n - 55))
    r4 = at(coin_series(short, short), short, (DAY0 + n) * DAY)
    check(
        "touch_ma: touch after 10 days above = 1; earlier dip = 0; no touch = 0; short = null",
        r["c_touch_ma"] == 1.0
        and r2["c_touch_ma"] == 0.0
        and r3["c_touch_ma"] == 0.0
        and r4["c_touch_ma"] is None,
        f"{r['c_touch_ma']} {r2['c_touch_ma']} {r3['c_touch_ma']} {r4['c_touch_ma']}",
    )


def logic_market() -> None:
    # Breadth, median, moonshot share over the day's coins (nulls ignored).
    rows = pl.DataFrame(
        {
            "d_us": [1, 1, 1, 1],
            "c_sma20": [0.1, -0.2, None, 0.3],
            "c_sma50": [0.1, -0.2, -0.1, -0.3],
            "c_sma200": [None, None, None, 0.5],
            "c_ret7": [0.1, -0.1, 0.2, 0.0],
            "c_ret30": [1.0, 0.0, 0.5, -0.5],
            "_moon30": [1.0, 0.0, 0.0, None],
        }
    )
    r = breadth(rows).row(0, named=True)
    lr = np.log1p([0.1, -0.1, 0.2, 0.0])
    expect(
        "breadth shares, median, dispersion, moonshot share",
        r,
        {
            "m_breadth20": 2 / 3,
            "m_breadth50": 0.25,
            "m_breadth200": 1.0,
            "m_med_ret7": 0.05,
            "m_disp_ret7": float(np.std(lr, ddof=1)),
            "m_med_ret30": 0.25,
            "m_moon30": 1 / 3,
        },
    )
    panel = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "d_us": [1, 1, 1, 2],
            "liq": [3e6, 1e6, 2e6, 5e6],
            "days_listed": [100, 40, 50, 30],
        }
    )
    pf = panel_factors(panel).sort("symbol")
    check(
        "liq rank within the day: (rank - 1) / (n - 1), 0.5 for a lone coin",
        pf["c_liq_rank"].to_list() == [1.0, 0.0, 0.5, 0.5],
        str(pf["c_liq_rank"].to_list()),
    )
    # Market volume: two coins at qv 1 for 83 days, then 2 for 7 days.
    qv = [1.0] * 83 + [2.0] * 7
    dq = pl.DataFrame(
        {"day": [DAY0 + k for k in range(90)] * 2, "qv": qv * 2},
        schema={"day": pl.Int64, "qv": pl.Float64},
    )
    mv = market_volume(dq, np.array([(DAY0 + 90) * DAY], dtype=np.int64))
    check("m_qv_7_90 by hand", same(mv["m_qv_7_90"][0], 2 / (97 / 90)), str(mv))
    # BTC up 10% and ETH up 32% over 30 days: ETH/BTC +20%.
    btc = daily_bars(synth([*flat_days(30), *flat_days(1, 110.0)]))
    eth = daily_bars(synth([*flat_days(30, 50.0), *flat_days(1, 66.0)]))
    m = at(market_series(btc, eth), btc, (DAY0 + 31) * DAY)
    expect(
        "BTC return and ETH/BTC return",
        m,
        {"m_btc_ret30": 0.10, "m_btc_ret1": 0.10, "m_ethbtc_ret30": 0.20, "m_btc_ret90": None},
    )


# --- 2. real entries -----------------------------------------------------------------------------


def perturb(c5: pl.DataFrame, d: int) -> pl.DataFrame:
    """The candles with every value at or after D changed."""
    late = pl.col("t") >= d

    def f(col: str, k: float) -> pl.Expr:
        return pl.when(late).then(pl.col(col) * k).otherwise(pl.col(col)).alias(col)

    return c5.with_columns(
        f("open", 1.7),
        f("high", 4.0),
        f("low", 0.2),
        f("close", 2.5),
        f("quote_volume", 50.0),
        f("trade_count", 0.1),
        f("taker_buy_quote_volume", 3.0),
    )


def real_entries(fac: pl.DataFrame, panel: pl.DataFrame) -> None:
    rng = random.Random(SEED)
    delisted = set((DATA_DIR / "delisted.txt").read_text().split())
    picks: list[tuple[str, int]] = []
    for year in range(2020, 2027):
        rows = panel.filter(pl.col("year") == year, pl.col("symbol") != BTC)
        picks += [(r[0], r[1]) for r in rows.select("symbol", "d_us").sample(5, seed=year).rows()]
    btc_days = panel.filter(pl.col("symbol") == BTC)["d_us"].to_list()
    picks += [(BTC, d) for d in rng.sample(btc_days, 4)]
    dl = panel.filter(pl.col("symbol").is_in(sorted(delisted)))
    picks += [(r[0], r[1]) for r in dl.select("symbol", "d_us").sample(9, seed=SEED).rows()]
    n_del = sum(s in delisted for s, _ in picks)
    years = sorted({utc(d).year for _, d in picks})

    btc5 = read_5m(BTC, flow=True)
    eth5 = read_5m(ETH, flow=True)
    cache: dict[str, pl.DataFrame] = {}
    bad_cut: list[str] = []
    bad_alt: list[str] = []
    for sym, d in picks:
        c5 = cache.setdefault(sym, read_5m(sym, flow=True))
        want = fac.filter(pl.col("symbol") == sym, pl.col("d_us") == d).row(0, named=True)
        p = panel.filter(pl.col("symbol") == sym, pl.col("d_us") == d)["P"].to_numpy()
        days = np.array([d], dtype=np.int64)
        # cut at D: the coin's, BTC's and ETH's candles
        cut = c5.filter(pl.col("t") < d)
        btc_cut = bars_from_candles(btc5, d)
        eth_cut = bars_from_candles(eth5, d)
        got = coin_rows(sym, cut, btc_cut, days, p).row(0, named=True)
        got |= at_days(market_series(btc_cut, eth_cut), btc_cut.d0, btc_cut.n, days).row(
            0, named=True
        )
        pr = panel_rows(
            sym,
            cut["t"].to_numpy(),
            cut["close"].to_numpy(),
            cut["quote_volume"].to_numpy(),
            first_day_pre2020(sym),
        ).filter(pl.col("d_us") == d)
        got["c_log_liq"] = math.log(pr["liq"][0]) if pr.height else None
        got["c_log_age"] = math.log(pr["days_listed"][0]) if pr.height else None
        cols = [*SERIES_FACTORS, *BTC_FACTORS, "c_log_liq", "c_log_age"]
        bad_cut += [f"{sym}@{d}: {x}" for x in diff_rows(want, got, cols)]
        # changed at and after D (full series, nothing dropped)
        alt5, btc_alt, eth_alt = perturb(c5, d), perturb(btc5, d), perturb(eth5, d)
        btc_b = daily_bars(btc_alt)
        alt = coin_rows(sym, alt5, btc_b, days, p).row(0, named=True)
        alt |= at_days(market_series(btc_b, daily_bars(eth_alt)), btc_b.d0, btc_b.n, days).row(
            0, named=True
        )
        bad_alt += [
            f"{sym}@{d}: {x}" for x in diff_rows(want, alt, [*SERIES_FACTORS, *BTC_FACTORS])
        ]
    check(
        f"look-ahead: {len(picks)} real entries ({n_del} of delisted coins, BTC 4, years "
        f"{years[0]}-{years[-1]}) recomputed from candles cut at D, {len(cols)} factors each",
        len(picks) >= 40 and not bad_cut,
        str(bad_cut[:10]),
    )
    check(
        f"look-ahead: the same entries with every candle at/after D changed ({len(picks)})",
        not bad_alt,
        str(bad_alt[:10]),
    )


# --- 3. real days, every coin ------------------------------------------------------------------

CHECK_DAYS = [
    datetime(2020, 9, 15, tzinfo=UTC),
    datetime(2021, 11, 10, tzinfo=UTC),
    datetime(2022, 6, 20, tzinfo=UTC),
    datetime(2023, 8, 1, tzinfo=UTC),
    datetime(2025, 4, 7, tzinfo=UTC),
]
_BTC5: pl.DataFrame | None = None


def day_job(args: tuple[str, list[int]]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """One coin at each check day, from its candles cut at D: panel row, factors, daily qv."""
    global _BTC5
    if _BTC5 is None:
        _BTC5 = read_5m(BTC, flow=True)
    sym, days = args
    c5 = read_5m(sym, flow=True)
    pre = first_day_pre2020(sym)
    prs, facs, qvs = [], [], []
    for d in days:
        cut = c5.filter(pl.col("t") < d)
        pr = panel_rows(
            sym, cut["t"].to_numpy(), cut["close"].to_numpy(), cut["quote_volume"].to_numpy(), pre
        ).filter(pl.col("d_us") == d)
        b = daily_bars(cut)
        qvs.append(
            pl.DataFrame(
                {"D": d, "day": b.d0 + np.arange(b.n, dtype=np.int64), "qv": b.qv},
                schema={"D": pl.Int64, "day": pl.Int64, "qv": pl.Float64},
            )
        )
        if pr.height:
            prs.append(pr)
            btc_cut = bars_from_candles(_BTC5, d)
            facs.append(coin_rows(sym, cut, btc_cut, pr["d_us"].to_numpy(), pr["P"].to_numpy()))
    empty = pl.DataFrame()
    return (
        pl.concat(prs) if prs else empty,
        pl.concat(facs) if facs else empty,
        pl.concat(qvs),
    )


def real_days(fac: pl.DataFrame, panel: pl.DataFrame, workers: int, days: list[int]) -> None:
    jobs = [(s, days) for s in universe()]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        parts = list(pool.map(day_job, jobs, chunksize=2))
    prs = pl.concat([p[0] for p in parts if p[0].height])
    facs = pl.concat([p[1] for p in parts if p[1].height])
    qvs = pl.concat([p[2] for p in parts])
    for d in days:
        label = str(pl.from_epoch(pl.Series([d]), time_unit="us").dt.date()[0])
        pr = prs.filter(pl.col("d_us") == d)
        want_panel = panel.filter(pl.col("d_us") == d)
        same_members = pr.sort("symbol").equals(want_panel.sort("symbol"))
        rows = pr.select("symbol", "d_us").join(facs, on=["symbol", "d_us"], how="left")
        got = (
            rows.join(breadth(rows), on="d_us")
            .join(
                market_volume(qvs.filter(pl.col("D") == d), np.array([d], dtype=np.int64)),
                on="d_us",
            )
            .join(panel_factors(pr), on=["symbol", "d_us"])
        )
        want = fac.filter(pl.col("d_us") == d)
        cols = [*SERIES_FACTORS, *BREADTH_FACTORS, "m_qv_7_90", *panel_factors_cols()]
        j = want.join(got, on=["symbol", "d_us"], how="left", suffix="_got")
        bad = []
        for r in j.iter_rows(named=True):
            for k in cols:
                if not same(r[k], r[f"{k}_got"]):
                    bad.append(f"{r['symbol']} {k}: {r[k]!r} vs {r[k + '_got']!r}")
        check(
            f"look-ahead: {label}, every coin cut at D: panel members ({pr.height}), all coin "
            f"factors, breadth, liq rank, m_qv_7_90",
            same_members and want.height == pr.height and not bad,
            f"members_equal={same_members} {bad[:8]}",
        )


def panel_factors_cols() -> list[str]:
    return ["c_log_liq", "c_liq_rank", "c_log_age"]


# --- 4. table, 5. coverage ---------------------------------------------------------------------


def table(fac: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    check(
        "factors keys equal the panel's",
        fac.select("symbol", "d_us")
        .sort(pl.all())
        .equals(panel.select("symbol", "d_us").sort(pl.all())),
    )
    check(
        f"factor columns: {len(NAMES)}, in dictionary order, Float64",
        fac.columns == ["symbol", "d_us", *NAMES]
        and all(fac.schema[n] == pl.Float64 for n in NAMES),
    )
    nonfinite = fac.select(pl.sum_horizontal([pl.col(n).is_infinite().sum() for n in NAMES]))
    nans = fac.select(pl.sum_horizontal([pl.col(n).is_nan().sum() for n in NAMES]))
    check("no inf or NaN (missing is null)", nonfinite.item() == 0 and nans.item() == 0)
    disc = fac.filter(pl.col("d_us") <= DISCOVERY_END_US)
    stats = []
    for f in FACTORS_LIST:
        s = disc[f.name]
        stats.append(
            {
                "factor": f.name,
                "family": f.family,
                "kind": f.kind,
                "non_null": 1 - s.null_count() / len(s),
                "mean": s.mean(),
                "std": s.std(),
                "p1": s.quantile(0.01, "linear"),
                "p50": s.quantile(0.50, "linear"),
                "p99": s.quantile(0.99, "linear"),
                "distinct": s.drop_nulls().n_unique(),
            }
        )
    cov = pl.DataFrame(stats)
    const = cov.filter(pl.col("distinct") < 2)["factor"].to_list()
    sparse = cov.filter(pl.col("non_null") < 1 - MAX_NULL)["factor"].to_list()
    check("discovery: no constant factor", not const, str(const))
    check(f"discovery: no factor over {MAX_NULL:.0%} null", not sparse, str(sparse))
    return cov


def write_coverage(cov: pl.DataFrame, n_rows: int) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cov.write_csv(RESULTS / "m2_coverage.csv", float_precision=6)

    def g(x: float | None) -> str:
        return "" if x is None else f"{x:.4g}"

    lines = [
        "# M2 factor coverage on discovery entries",
        "",
        f"Discovery entries (D 2020-01-31 .. 2023-12-01): {n_rows}. Factor values only (no",
        "outcomes). Built by `check_factors.py` from `cache/factors.parquet`; definitions in",
        "`factors.md`. Percentiles are linear; `distinct` counts distinct non-null values.",
        "",
        "| factor | kind | non-null | mean | std | p1 | p50 | p99 | distinct |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in cov.iter_rows(named=True):
        lines.append(
            f"| `{r['factor']}` | {r['kind']} | {r['non_null']:.1%} | {g(r['mean'])} | "
            f"{g(r['std'])} | {g(r['p1'])} | {g(r['p50'])} | {g(r['p99'])} | {r['distinct']} |"
        )
    (RESULTS / "m2_coverage.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logic-only", action="store_true", help="synthetic checks only")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logic_bars()
    logic_values()
    logic_habit()
    logic_touch()
    logic_market()
    if not args.logic_only:
        fac = pl.read_parquet(FACTORS)
        panel = pl.read_parquet(PANEL)
        cov = table(fac, panel)
        real_entries(fac, panel)
        real_days(fac, panel, args.workers, [us(d) for d in CHECK_DAYS])
        write_coverage(cov, int(fac.filter(pl.col("d_us") <= DISCOVERY_END_US).height))
    print(f"{len(FAILURES)} failed" if FAILURES else "all checks passed")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
