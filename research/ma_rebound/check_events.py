"""M1: look-ahead and invariant checks of the event tables.

    uv run --group research python check_events.py [--quick]

1. Invariants of the combined tables (candidates of every year; fills and labels of 2020-24 only,
   so no 2025-26 outcome is ever read).
2. Truncation: rebuilding several coins with the data cut at random times leaves every bar,
   level, candidate, fill decision, exit and forward path that was fully determined before the
   cut unchanged, and marks the rest censored.
3. Direct loop: for 20 random study candidates with a trigger in 2024 or earlier (two per
   timeframe and family), a deliberately simple loop over the raw 5m/1m candles, written without
   `events.py` or `resolve_exit`, reproduces the level, the trigger (the coin's whole trigger list
   for that MA and family up to 2024), each buffer's fill (time and price, base and stress) and
   every exit.

Prints one pass/fail line per check and nothing about outcomes; exits non-zero on a failure.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import events as ev
import polars as pl
from common import (
    ARM_ZONE,
    CACHE,
    DAILY_PRE2020,
    DATA_END,
    EXITS,
    HOLDOUT_START,
    KLINES_1M,
    KLINES_5M,
    MAX_HOLD,
    MIN_DAILY_QUOTE_VOLUME,
    STRESS_THROUGH,
    TIMEFRAMES,
    TRIGGER_ZONE,
)
from runner import END_US, PARTS, read_part

HOLDOUT_US = int(HOLDOUT_START.timestamp()) * 1_000_000
REL = 1e-9
FAILURES: list[str] = []
USE_PARTS = False  # --parts: read the per-coin part files instead of the combined tables


def table(name: str) -> pl.DataFrame:
    if USE_PARTS:
        return pl.concat([pl.read_parquet(f) for f in sorted((PARTS / name).glob("*.parquet"))])
    return pl.read_parquet(CACHE / f"{name}.parquet")


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  ' + detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= REL * max(abs(a), abs(b), 1e-12)


# --- 1. invariants -------------------------------------------------------------------------


def invariants() -> None:
    cands = table("candidates")
    hold = pl.col("tf").replace_strict({tf: ev.hold_us(tf) for tf in TIMEFRAMES})
    check("candidate ids unique", cands["cid"].n_unique() == cands.height)
    check("decision = trigger close", (cands["decision_t"] == cands["trigger_t"] + ev.FIVE).all())
    check("decisions before the data end", (cands["decision_t"] <= END_US).all())
    check("liquidity floor", (cands["liq"] >= MIN_DAILY_QUOTE_VOLUME).all())
    check("u positive and = atr_d sqrt(T)", (cands["u"] > 0).all())
    u_ok = cands.with_columns(
        want=pl.col("atr_d")
        * pl.col("tf").replace_strict({k: math.sqrt(v.hold_days) for k, v in TIMEFRAMES.items()})
    )
    check("u = atr_d sqrt(T)", ((u_ok["u"] - u_ok["want"]).abs() <= 1e-12 * u_ok["u"]).all())
    lim = cands.filter(pl.col("family") == "limit")
    check("limit triggers in the zone", (lim["distance"] <= TRIGGER_ZONE + 1e-9).all())
    check("limit trigger lows above L", (lim["low"] > lim["L"]).all())
    tch = cands.filter(pl.col("family") == "touch")
    check("touch candles reach L", (tch["low"] <= tch["L"]).all())
    check("bars since touch >= k", _since_touch_ok(lim))

    seen = cands.filter(pl.col("trigger_t") < HOLDOUT_US).select(
        "cid", "family", "placebo", "tf", "u", "decision_t", "close", hold.alias("hold")
    )
    fills = table("fills").join(seen, on="cid", how="inner")
    f = fills.filter(pl.col("status") == "filled")
    check("fills at or after the decision", (f["fill_t"] >= f["decision_t"]).all())
    lf = f.filter(pl.col("family") == "limit")
    check("limit fills within T/2", (lf["fill_t"] < lf["decision_t"] + lf["hold"] // 2).all())
    mk = lf.filter(pl.col("kind") == "maker")
    want = mk["level_at_fill"] * (1 + mk["buffer"] * mk["u"])
    check("maker fills at the limit", ((mk["price"] - want).abs() <= 1e-12 * want).all())
    tk = lf.filter(pl.col("kind") == "taker")
    check("takers only when the close was at or below the limit", _taker_ok(tk))
    wide = f.filter(pl.col("family") == "limit").pivot(
        "scenario", index=["cid", "buffer"], values="fill_t"
    )
    later = wide.filter(pl.col("stress").is_not_null())
    check("stress fills never earlier", (later["stress"] >= later["base"]).all())
    check(
        "stress fills imply base fills",
        wide.filter(pl.col("base").is_null(), pl.col("stress").is_not_null()).height == 0,
    )
    check("fills censored only at the data end", _censored_ok(fills))

    labels = table("labels").join(seen, on="cid", how="inner")
    d = labels.filter(pl.col("status") == "done")
    check("exits not before the entry", (d["exit_t"] >= d["entry_t"]).all())
    lim_e = pl.col("exit").replace_strict(
        {e: x.time for e, x in EXITS.items()}, return_dtype=pl.Float64
    )
    cap = pl.min_horizontal(lim_e * pl.col("hold"), MAX_HOLD // timedelta(microseconds=1))
    d = d.with_columns(cap=cap.cast(pl.Int64))
    # A 5m candle (1m incomplete) that opens before the time limit can stop out at its close.
    check("exits by the time limit", (d["exit_t"] <= d["entry_t"] + d["cap"] + ev.FIVE).all())
    check(
        "labels censored only at the data end",
        labels.filter(pl.col("status") == "censored").height == 0,
    )
    tp = d.filter(pl.col("reason") == "take_profit")
    check("take profits only on E3/E4", tp["exit"].is_in(["E3", "E4"]).all())
    st = d.filter(pl.col("reason") == "stop")
    check("stops only on E3/E4/E5", st["exit"].is_in(["E3", "E4", "E5"]).all())
    combos = labels.group_by("family", "placebo").agg(pl.col("exit").unique().sort())
    want_exits = {
        ("limit", False): sorted(EXITS),
        ("touch", False): sorted(EXITS),
        ("limit", True): ["E1", "E3"],
        ("touch", True): ["E1"],
    }
    got = {(r["family"], r["placebo"]): r["exit"] for r in combos.iter_rows(named=True)}
    check("exit combinations", got == want_exits, str(got))
    fwd = table("fwd").join(seen, on="cid", how="inner")
    check("forward paths for study MAs only", not fwd["placebo"].any())
    check("forward paths complete", fwd["fwd"].is_not_null().all())


def _since_touch_ok(lim: pl.DataFrame) -> bool:
    k = pl.col("tf").replace_strict({tf: s.above_bars for tf, s in TIMEFRAMES.items()})
    bad = lim.filter(pl.col("bars_since_touch").is_not_null(), pl.col("bars_since_touch") < k)
    return bad.height == 0


def _taker_ok(tk: pl.DataFrame) -> bool:
    lim = tk["level_at_fill"] * (1 + tk["buffer"] * tk["u"])
    return bool(
        (tk["close"] <= lim * (1 + 1e-12)).all() and (tk["fill_t"] >= tk["decision_t"]).all()
    )


def _censored_ok(fills: pl.DataFrame) -> bool:
    return fills.filter(pl.col("status") == "censored").height == 0


# --- 2. truncation -------------------------------------------------------------------------


def frames_equal(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    if a.columns != b.columns or a.height != b.height:
        return False
    return a.equals(b)


def truncation(symbols: list[str], cuts_per_coin: int, rng: random.Random) -> None:
    for sym in symbols:
        c5_full = ev.read_5m(sym, END_US)
        t5 = c5_full["t"].to_numpy()
        lo = max(
            int(t5[0]) + 400 * ev.DAY, int(datetime(2021, 1, 1, tzinfo=UTC).timestamp()) * 10**6
        )
        hi = min(int(t5[-1]), HOLDOUT_US - 40 * ev.DAY)
        if hi <= lo:
            continue
        full_bars = ev.read_bars(sym)
        full_c = read_part("candidates", sym)
        full_f = read_part("fills", sym)
        full_l = read_part("labels", sym)
        full_w = read_part("fwd", sym)
        for _ in range(cuts_per_coin):
            for _attempt in range(50):
                cut = rng.randrange(lo, hi) // ev.MINUTE * ev.MINUTE
                i = int(t5.searchsorted(cut))
                if i > 0 and t5[i - 1] + ev.FIVE >= cut - ev.HOUR:  # trading at the cut
                    break
            _truncation_one(sym, cut, full_bars, full_c, full_f, full_l, full_w)


def _truncation_one(
    sym: str,
    cut: int,
    full_bars: dict[str, pl.DataFrame],
    full_c: pl.DataFrame,
    full_f: pl.DataFrame,
    full_l: pl.DataFrame,
    full_w: pl.DataFrame,
) -> None:
    tag = f"{sym} cut {datetime.fromtimestamp(cut / 1e6, UTC):%Y-%m-%d %H:%M}"
    c5 = ev.read_5m(sym, cut)
    c1 = ev.read_1m(sym, cut)
    bars = ev.build_bars(c5, ev.read_daily_pre2020(sym, cut))
    ok = True
    for tf in ev.TF_NAMES:
        tb, fb = bars[tf], full_bars[tf]
        ends = ev.bar_start(tb["idx"].to_numpy() + 1, tf)
        done = tb.filter(pl.Series(ends <= cut))
        ok &= frames_equal(done, fb.head(done.height))
        # The level (and atr_d, liq) of the bar forming at the cut is already known.
        lev_cols = [c for c in tb.columns if c.startswith("L") or c in ("atr_d", "liq")]
        ok &= frames_equal(tb.select("idx", *lev_cols), fb.head(tb.height).select("idx", *lev_cols))
    check(f"truncation bars and levels  {tag}", ok)

    cands = ev.build_candidates(sym, c5, bars)
    want = full_c.filter(pl.col("decision_t") <= cut)
    check(f"truncation candidates ({cands.height})  {tag}", frames_equal(cands, want))

    series = ev.make_series(sym, c5, c1, cut)
    fills = ev.build_fills(cands, bars, series)
    hold = {tf: ev.hold_us(tf) for tf in TIMEFRAMES}
    meta = cands.select("cid", "family", "tf", "decision_t")
    full = full_f.join(meta, on="cid").with_columns(
        w_end=pl.col("decision_t") + pl.col("tf").replace_strict(hold) // 2
    )
    keys = ["cid", "buffer", "scenario"]
    joined = full.join(fills, on=keys, how="full", suffix="_t", nulls_equal=True)
    decided = ((pl.col("status") == "filled") & (pl.col("fill_t") + pl.col("fill_len") <= cut)) | (
        (pl.col("status") == "expired") & (pl.col("family") == "limit") & (pl.col("w_end") <= cut)
    )
    cols = ["status", "kind", "fill_t", "fill_len", "price", "level_at_fill"]
    same = pl.all_horizontal([pl.col(c).eq_missing(pl.col(f"{c}_t")) for c in cols])
    bad_decided = joined.filter(decided & ~same).height
    bad_open = joined.filter(~decided & ~same & (pl.col("status_t") != "censored")).height
    check(
        f"truncation fills ({fills.height}, open {joined.filter(~decided).height})  {tag}",
        bad_decided == 0
        and bad_open == 0
        and joined.filter(pl.col("status").is_null()).height == 0,
        f"{bad_decided} decided differ, {bad_open} undecided not censored",
    )

    labels, fwd = ev.build_labels(cands, fills, series)
    keys = ["cid", "buffer", "scenario", "exit"]
    jl = full_l.join(labels, on=keys, how="inner", suffix="_t", nulls_equal=True)
    cols = ["status", "entry_t", "entry_price", "entry_kind", "exit_t", "exit_price", "reason"]
    same = pl.all_horizontal([pl.col(c).eq_missing(pl.col(f"{c}_t")) for c in cols])
    known = (pl.col("status") == "done") & (pl.col("exit_t") + ev.FIVE <= cut)
    bad_known = jl.filter(known & ~same).height
    bad_open = jl.filter(~known & ~same & (pl.col("status_t") != "censored")).height
    # Every label of the truncated build has its fill in the full build.
    orphans = labels.join(full_l, on=keys, how="anti", nulls_equal=True)
    orphan = orphans.filter(pl.col("status") != "censored").height
    # Every known exit of a fill decided before the cut is in the truncated build.
    lost = full_l.filter(known).join(labels, on=keys, how="anti", nulls_equal=True)
    fill_keys = ["cid", "buffer", "scenario", "fill_t", "fill_len"]
    decided_fills = full_f.filter(pl.col("fill_t") + pl.col("fill_len") <= cut).select(fill_keys)
    gone = lost.join(
        decided_fills.rename({"fill_t": "entry_t"}).drop("fill_len"),
        on=["cid", "buffer", "scenario", "entry_t"],
        nulls_equal=True,
    ).height
    check(
        f"truncation exits ({labels.height}, known {jl.filter(known).height})  {tag}",
        bad_known == 0 and bad_open == 0 and orphan == 0 and gone == 0,
        f"{bad_known} known differ, {bad_open} unknown not censored, {orphan} orphans, "
        f"{gone} missing",
    )
    jw = full_w.join(fwd, on=["cid", "buffer"], how="inner", suffix="_t", nulls_equal=True)
    ent = full_f.filter(pl.col("scenario") == "base").select("cid", "buffer", "fill_t")
    jw = jw.join(ent, on=["cid", "buffer"], nulls_equal=True).join(meta, on="cid")
    known = pl.col("fill_t") + pl.col("tf").replace_strict(hold) <= cut
    same = (
        pl.col("fwd").eq_missing(pl.col("fwd_t"))
        & pl.col("mfe").eq_missing(pl.col("mfe_t"))
        & pl.col("mae").eq_missing(pl.col("mae_t"))
    )
    bad = jw.filter(known & ~same).height + jw.filter(~known & pl.col("fwd_t").is_not_null()).height
    check(f"truncation forward paths ({jw.height})  {tag}", bad == 0, f"{bad} differ")


# --- 3. direct loop ------------------------------------------------------------------------
# Everything below reads the raw parquet files itself and uses plain Python loops and datetimes.


@dataclass
class Bar:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    qv: float


def period_start(t: datetime, tf: str) -> datetime:
    if tf == "1h":
        return t.replace(minute=0, second=0, microsecond=0)
    if tf == "4h":
        return t.replace(hour=t.hour // 4 * 4, minute=0, second=0, microsecond=0)
    day = t.replace(hour=0, minute=0, second=0, microsecond=0)
    if tf == "1d":
        return day
    if tf == "1w":
        return day - timedelta(days=day.weekday())
    return day.replace(day=1)


def period_number(t: datetime, tf: str) -> int:
    if tf == "1M":
        return t.year * 12 + t.month
    size = {"1h": 3600, "4h": 14400, "1d": 86400, "1w": 7 * 86400}[tf]
    return round((t - datetime(2000, 1, 3, tzinfo=UTC)).total_seconds()) // size


@dataclass
class Raw:
    """A coin's raw candles as Python lists, and its bars built by a loop."""

    t5: list[datetime]
    o5: list[float]
    h5: list[float]
    l5: list[float]
    c5: list[float]
    m1: dict[datetime, tuple[float, float, float, float]]
    i5: dict[datetime, int]
    bars: dict[str, list[Bar]] = field(default_factory=dict)
    bar_pos: dict[str, dict[datetime, int]] = field(default_factory=dict)
    ended: bool = False
    last_close: datetime = DATA_END


def load_raw(sym: str) -> Raw:
    df = pl.read_parquet(KLINES_5M / f"symbol={sym}" / "*.parquet").sort("open_time")
    df = df.filter(pl.col("open_time") + timedelta(minutes=5) <= DATA_END)
    m = pl.read_parquet(
        KLINES_1M / f"symbol={sym}" / "*.parquet",
        columns=["open_time", "open", "high", "low", "close"],
    ).filter(pl.col("open_time") + timedelta(minutes=1) <= DATA_END)
    raw = Raw(
        t5=df["open_time"].to_list(),
        o5=df["open"].to_list(),
        h5=df["high"].to_list(),
        l5=df["low"].to_list(),
        c5=df["close"].to_list(),
        m1={r[0]: (r[1], r[2], r[3], r[4]) for r in m.iter_rows()},
        i5={},
    )
    raw.i5 = {t: i for i, t in enumerate(raw.t5)}
    raw.last_close = raw.t5[-1] + timedelta(minutes=5)
    raw.ended = raw.last_close <= DATA_END - timedelta(days=2)
    qv = df["quote_volume"].to_list()
    rows = list(zip(raw.t5, raw.o5, raw.h5, raw.l5, raw.c5, qv, strict=True))
    pre_path = DAILY_PRE2020 / f"{sym}.parquet"
    days: list[tuple[datetime, float, float, float, float, float]] = []
    if pre_path.exists():
        pre = pl.read_parquet(pre_path).sort("open_time")
        first_day = period_start(raw.t5[0], "1d")
        for r in pre.iter_rows(named=True):
            if r["open_time"] < first_day:
                days.append(
                    (r["open_time"], r["open"], r["high"], r["low"], r["close"], r["quote_volume"])
                )
    for tf in ("1h", "4h", "1d", "1w", "1M"):
        src = rows if tf in ("1h", "4h") else days + rows
        out: list[Bar] = []
        for t, o, h, lo, c, v in src:
            s = period_start(t, tf)
            if out and out[-1].start == s:
                b = out[-1]
                b.high, b.low, b.close, b.qv = max(b.high, h), min(b.low, lo), c, b.qv + v
            else:
                out.append(Bar(s, o, h, lo, c, v))
        raw.bars[tf] = out
        raw.bar_pos[tf] = {b.start: i for i, b in enumerate(out)}
    return raw


def ref_level(raw: Raw, tf: str, n: int, i: int) -> float | None:
    bars = raw.bars[tf]
    if i < n:
        return None
    first, last = bars[i - n], bars[i - 1]
    if period_number(last.start, tf) - period_number(first.start, tf) + 1 > 1.1 * n:
        return None
    return sum(round(b.close * 10**8) for b in bars[i - n : i]) / (n * 10**8)


def ref_day_stats(raw: Raw, day: datetime) -> tuple[float | None, float | None]:
    bars = raw.bars["1d"]
    i = raw.bar_pos["1d"][day]
    atr = None
    if i >= 15 and (bars[i - 1].start - bars[i - 14].start).days + 1 <= 15.4:
        trs = []
        for j in range(i - 14, i):
            pc = bars[j - 1].close
            trs.append(
                max(bars[j].high - bars[j].low, abs(bars[j].high - pc), abs(bars[j].low - pc))
            )
        atr = sum(trs) / 14 / bars[i - 1].close
    liq = None
    if (day - bars[0].start).days >= 30:
        total = sum(b.qv for b in bars[:i] if b.start >= day - timedelta(days=30))
        liq = total / 30
    return atr, liq


def level_at(raw: Raw, tf: str, n: int, t: datetime) -> float | None:
    i = raw.bar_pos[tf].get(period_start(t, tf))
    return None if i is None else ref_level(raw, tf, n, i)


def ref_triggers(raw: Raw, tf: str, n: int, family: str, until: datetime) -> list[datetime]:
    """The coin's candidates for one MA and family, by a loop over every 5m candle."""
    k = TIMEFRAMES[tf].above_bars
    sq = math.sqrt(TIMEFRAMES[tf].hold_days)
    bars = raw.bars[tf]
    levels = [ref_level(raw, tf, n, i) for i in range(len(bars))]
    above_bar = [lv is not None and b.low > lv for b, lv in zip(bars, levels, strict=True)]
    day_cache: dict[datetime, tuple[float | None, float | None]] = {}
    out = []
    armed = False
    cur = None
    low_so_far = math.inf
    for j, t in enumerate(raw.t5):
        if t >= until:
            break
        s = period_start(t, tf)
        if s != cur:
            cur, low_so_far = s, math.inf
        i = raw.bar_pos[tf][s]
        lv = levels[i]
        prev_ok = i >= k and all(above_bar[i - k : i])
        before = low_so_far
        low_so_far = min(low_so_far, raw.l5[j])
        day = period_start(t, "1d")
        if day not in day_cache:
            day_cache[day] = ref_day_stats(raw, day)
        atr, liq = day_cache[day]
        u = None if atr is None else atr * sq
        liquid = liq is not None and liq >= MIN_DAILY_QUOTE_VOLUME
        if family == "touch":
            if prev_ok and lv is not None and u and before > lv and raw.l5[j] <= lv and liquid:
                out.append(t)
            continue
        if not (prev_ok and lv is not None and low_so_far > lv and u):
            armed = False
            continue
        c = raw.c5[j]
        if c >= lv * (1 + ARM_ZONE * u):
            armed = True
        elif c <= lv * (1 + TRIGGER_ZONE * u) and armed:
            armed = False
            if liquid:
                out.append(t)
    return out


Candle = tuple[datetime, datetime, float, float, float, float]  # open, close time, OHLC


def ref_path(raw: Raw, start: datetime, end: datetime) -> list[Candle]:
    """Candles opening in [start, end): 1m where a 5m candle has all five, else the 5m one."""
    out: list[Candle] = []
    t = period_start(start, "1h") + timedelta(minutes=(start.minute // 5) * 5)
    while t < end:
        j = raw.i5.get(t)
        if j is not None:
            minutes = [t + timedelta(minutes=m) for m in range(5)]
            if all(x in raw.m1 for x in minutes):
                for x in minutes:
                    if start <= x < end:
                        o, h, lo, c = raw.m1[x]
                        out.append((x, x + timedelta(minutes=1), o, h, lo, c))
            elif start <= t:
                out.append(
                    (t, t + timedelta(minutes=5), raw.o5[j], raw.h5[j], raw.l5[j], raw.c5[j])
                )
        t += timedelta(minutes=5)
    return out


@dataclass
class RefFill:
    t: datetime
    length: timedelta
    price: float
    kind: str
    level: float


def ref_fill(
    raw: Raw, tf: str, n: int, decision: datetime, u: float, close_: float, b: float, stress: bool
) -> RefFill | None:
    end = decision + TIMEFRAMES[tf].hold / 2
    path = ref_path(raw, decision, end)
    for idx, (ot, ct, o, _h, lo, _c) in enumerate(path):
        lv = level_at(raw, tf, n, ot)
        if lv is None:
            continue
        limit = lv * (1 + b * u)
        if idx == 0 and close_ <= limit:
            return RefFill(ot, ct - ot, o, "taker", lv)
        if lo <= limit * (1 - STRESS_THROUGH if stress else 1):
            return RefFill(ot, ct - ot, limit, "maker", lv)
    return None


def ref_exit(
    raw: Raw, tf: str, fill: RefFill, u: float, exit_name: str
) -> tuple[datetime, float, str] | None:
    spec = EXITS[exit_name]
    deadline = fill.t + min(TIMEFRAMES[tf].hold * spec.time, MAX_HOLD)
    p = fill.price
    stop = None
    if spec.stop_below_level is not None:
        stop = fill.level * (1 - spec.stop_below_level * u)
        if stop >= p:
            return fill.t, p, "stop"
        if stop <= 0:
            stop = None
    target = None if spec.take_profit is None else p * (1 + spec.take_profit * u)
    path = ref_path(raw, fill.t, deadline + timedelta(minutes=5))
    ot, ct, _o, h, lo, c = path[0]
    if fill.kind == "maker":
        h = max(p, c)
    path[0] = (ot, ct, p, h, lo, c)
    last_close = None
    for ot, ct, o, h, lo, c in path:
        if ot >= deadline:
            break
        if target is not None and o >= target:
            return ot, o, "take_profit"
        if stop is not None and lo <= stop:
            return (ot, o, "stop") if o <= stop else (ct, stop, "stop")
        if target is not None and h >= target:
            return ct, target, "take_profit"
        last_close = (ct, c)
    assert last_close is not None
    continues = raw.t5[-1] >= deadline  # the series has a candle at or after the deadline
    if continues or not raw.ended or last_close[0] >= deadline:
        if not raw.ended and deadline > raw.last_close:
            return None  # the data ends before the deadline: unknown
        return deadline, last_close[1], "time"
    return last_close[0], last_close[1], "series_end"


def to_dt(us: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=us)


def direct_loop(count_per_cell: int, rng: random.Random) -> None:
    cands = table("candidates").filter(~pl.col("placebo"), pl.col("year") <= 2024)
    sample = []
    for tf in TIMEFRAMES:
        for family in ("limit", "touch"):
            pool = cands.filter(pl.col("tf") == tf, pl.col("family") == family)
            rows = pool.rows(named=True)
            sample += rng.sample(rows, min(count_per_cell, len(rows)))
    raws: dict[str, Raw] = {}
    until = datetime(2025, 1, 1, tzinfo=UTC)
    n_ok = 0
    for r in sample:
        sym, tf, n, family = r["symbol"], r["tf"], r["n"], r["family"]
        if sym not in raws:
            raws[sym] = load_raw(sym)
        raw = raws[sym]
        trig = to_dt(r["trigger_t"])
        problems = []
        # level and day stats
        lv = level_at(raw, tf, n, trig)
        if not close(lv, r["L"]):
            problems.append(f"level {lv} != {r['L']}")
        atr, liq = ref_day_stats(raw, period_start(trig, "1d"))
        if not (close(atr, r["atr_d"]) and close(liq, r["liq"])):
            problems.append("atr_d/liq")
        u = r["u"]
        if not close((atr or 0) * math.sqrt(TIMEFRAMES[tf].hold_days), u):
            problems.append("u")
        # the trigger, and the coin's whole list for this MA and family up to 2024
        ref_list = ref_triggers(raw, tf, n, family, until)
        built = cands.filter(
            pl.col("symbol") == sym,
            pl.col("tf") == tf,
            pl.col("n") == n,
            pl.col("family") == family,
        )["trigger_t"].to_list()
        if trig not in ref_list:
            problems.append("trigger not found by the loop")
        if [to_dt(x) for x in sorted(built)] != ref_list:
            problems.append(f"trigger lists differ ({len(built)} built, {len(ref_list)} loop)")
        # fills and exits
        cid = r["cid"]
        fills = read_part("fills", sym).filter(pl.col("cid") == cid)
        labels = read_part("labels", sym).filter(pl.col("cid") == cid)
        decision = trig + timedelta(minutes=5)
        for f in fills.iter_rows(named=True):
            stress = f["scenario"] == "stress"
            if family == "touch":
                j = next(i for i, t in enumerate(raw.t5) if t >= decision)
                t0 = raw.t5[j]
                lv0 = level_at(raw, tf, n, t0)
                assert lv0 is not None
                rf: RefFill | None = RefFill(t0, timedelta(minutes=5), raw.o5[j], "taker", lv0)
            else:
                rf = ref_fill(raw, tf, n, decision, u, r["close"], f["buffer"], stress)
            if rf is None:
                if f["status"] != "expired":
                    problems.append(f"fill {f['buffer']} {f['scenario']}: loop finds none")
                continue
            got = (to_dt(f["fill_t"]) if f["fill_t"] is not None else None, f["price"], f["kind"])
            if not (got[0] == rf.t and close(got[1], rf.price) and got[2] == rf.kind):
                problems.append(f"fill {f['buffer']} {f['scenario']}: {got} != {rf}")
                continue
            if not close(f["level_at_fill"], rf.level):
                problems.append("level at fill")
            mine = labels.filter(
                pl.col("buffer").eq_missing(f["buffer"]), pl.col("scenario") == f["scenario"]
            )
            for lab in mine.iter_rows(named=True):
                ex = ref_exit(raw, tf, rf, u, lab["exit"])
                if ex is None:
                    if lab["status"] != "censored":
                        problems.append(f"exit {lab['exit']}: loop says censored")
                    continue
                mine_ex = (to_dt(lab["exit_t"]), lab["exit_price"], lab["reason"])
                if not (mine_ex[0] == ex[0] and close(mine_ex[1], ex[1]) and mine_ex[2] == ex[2]):
                    problems.append(f"exit {lab['exit']} {f['buffer']} {f['scenario']} differs")
            if mine.height == 0:
                problems.append("no labels for a fill")
        if fills.height == 0:
            problems.append("no fill rows")
        n_ok += not problems
        check(f"direct loop {cid} ({fills.height} fills)", not problems, "; ".join(problems))
    check(f"direct loop: {n_ok}/{len(sample)} candidates reproduced", n_ok == len(sample))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--quick", action="store_true", help="fewer truncation coins")
    parser.add_argument("--skip", nargs="*", default=[], help="invariants | truncation | loop")
    parser.add_argument("--parts", action="store_true", help="read the part files")
    args = parser.parse_args()
    global USE_PARTS
    USE_PARTS = args.parts
    rng = random.Random(args.seed)
    t0 = time.perf_counter()
    if "invariants" not in args.skip:
        invariants()
    if "truncation" not in args.skip:
        cands = table("candidates").select("symbol")
        syms = sorted(set(cands["symbol"].to_list()))
        picks = ["BTCUSDT", *rng.sample(syms, 2 if args.quick else 5)]
        truncation(picks, 2 if args.quick else 3, rng)
    if "loop" not in args.skip:
        direct_loop(2, rng)
    print(f"{len(FAILURES)} failed ({time.perf_counter() - t0:.0f}s)")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
