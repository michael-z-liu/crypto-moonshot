"""M1: checks of the panel and the labels.

    uv run --group research python check_panel.py [--logic-only]

1. Logic, on synthetic candles (no data needed): hand-computed label paths and trade results
   (target first, stop first, both in one candle, neither, each stop variant, gaps, delisting,
   costs, entry at the first open at or after D with levels relative to P), and look-ahead:
   changing candles after the window changes no label, changing candles at or after D changes no
   panel row. These live here rather than under `tests/` because the research folders are not
   importable packages (each study has its own top-level `common`), see the M1 report.
2. Panel on the real data: invariants, `liq` against `ma_rebound`'s daily bars, and the
   truncation check on real coins (candles cut at D in memory).
3. Discovery labels: invariants, the label look-ahead on real coins, and a spot check of 20
   random discovery entries recomputed by a deliberately slow loop over the raw 5m files,
   written without `build_panel.py` or `build_labels.py`.
4. Sealed labels: shape, schema and keys only. No value of an outcome column is read.

Prints one pass/fail line per check and nothing about outcomes; exits non-zero on a failure.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections.abc import Callable

import numpy as np
import polars as pl
from build_labels import KEY_SCHEMA, LABEL_SCHEMA, Value, label_entry
from build_panel import PANEL_SCHEMA, panel_rows
from common import (
    DAY,
    DISCOVERY_END_US,
    FEE,
    FIRST_D_US,
    FIVE,
    HOUR,
    KLINES_5M,
    LABELS,
    LAST_D_US,
    MA_BARS_1D,
    MIN_LIQUIDITY,
    PANEL,
    SEALED_LABELS,
    SEALED_START_US,
    SLIPPAGE,
    WINDOW_US,
    first_day_pre2020,
    read_5m,
)

FAILURES: list[str] = []
REL = 1e-9
SEED = 20261114


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  ' + detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def same(a: Value, b: Value) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return abs(a - b) <= REL * max(abs(a), abs(b), 1e-12)
    return a == b


def same_dict(a: dict[str, Value], b: dict[str, Value]) -> list[str]:
    return [k for k in a if not same(a[k], b.get(k))]


# --- 1. logic on synthetic candles -------------------------------------------------------------

D0 = 1_700_006_400_000_000  # 2023-11-15 00:00 UTC
P = 100.0


def candles(
    rows: list[tuple[float, float, float, float]], start: int = D0, step: int = FIVE
) -> tuple[np.ndarray, ...]:
    t = start + step * np.arange(len(rows), dtype=np.int64)
    a = np.array(rows, dtype=np.float64).reshape(-1, 4)
    return t, a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy(), a[:, 3].copy()


def flat(n: int, px: float = P) -> list[tuple[float, float, float, float]]:
    return [(px, px + 1, px - 1, px)] * n


def run(arrs: tuple[np.ndarray, ...], d: int = D0, p: float = P) -> dict[str, Value]:
    t, o, h, lo, c = arrs
    return label_entry(t, o, h, lo, c, d, p)


def net(entry: float, exit_px: float, market_exit: bool) -> float:
    sell = exit_px * (1 - (SLIPPAGE if market_exit else 0.0)) * (1 - FEE)
    return sell / (entry * (1 + SLIPPAGE) * (1 + FEE)) - 1


def expect(name: str, got: dict[str, Value], want: dict[str, Value]) -> None:
    bad = [k for k in want if not same(got[k], want[k])]
    check(name, not bad, ", ".join(f"{k}: got {got[k]!r} want {want[k]!r}" for k in bad))


N30 = WINDOW_US // FIVE  # 8640 candles in a complete window


def logic_labels() -> None:
    # Target first: +30% in candle 1, the stop never.
    r = run(candles([*flat(1), (100, 131, 99, 130), *flat(N30 - 2, 130)]))
    expect(
        "target first",
        r,
        {
            "hit10": True,
            "hit20": True,
            "hit_ns": True,
            "reason_10": "target",
            "reason_20": "target",
            "reason_ns": "target",
            "exit_price_10": 130.0,
            "exit_t_10": D0 + FIVE,
            "ret_10": net(100, 130, False),
            "ret_ns": net(100, 130, False),
            "h_target": FIVE / HOUR,
            "h_stop10": None,
            "entry_open": 100.0,
            "entry_t": D0,
            "stopped_early": False,
            "n_candles": N30,
        },
    )
    # Stop first: -11% in candle 1, +35% in candle 2.
    r = run(candles([*flat(1), (100, 100, 89, 95), (95, 135, 94, 134), *flat(N30 - 3, 134)]))
    expect(
        "stop first (10%), target before the 20% stop",
        r,
        {
            "hit10": False,
            "hit20": True,
            "hit_ns": True,
            "reason_10": "stop",
            "exit_price_10": 90.0,
            "exit_t_10": D0 + FIVE,
            "ret_10": net(100, 90, True),
            "reason_20": "target",
            "ret_20": net(100, 130, False),
            "reason_ns": "target",
            "h_stop10": FIVE / HOUR,
            "h_stop20": None,
            "h_target": 2 * FIVE / HOUR,
            "trough30": -0.11,
        },
    )
    # Both in one candle: stop counts first.
    r = run(candles([*flat(1), (100, 131, 89, 120), *flat(N30 - 2, 120)]))
    expect(
        "both in one candle: stop first (10%)",
        r,
        {"hit10": False, "hit20": True, "reason_10": "stop", "ret_10": net(100, 90, True)},
    )
    r = run(candles([*flat(1), (100, 131, 79, 120), *flat(N30 - 2, 120)]))
    expect(
        "both in one candle: stop first (20%), no-stop variant takes the target",
        r,
        {
            "hit10": False,
            "hit20": False,
            "hit_ns": True,
            "reason_20": "stop",
            "exit_price_20": 80.0,
            "ret_20": net(100, 80, True),
            "reason_ns": "target",
            "ret_ns": net(100, 130, False),
        },
    )
    # Neither: time exit at the last close in the window.
    r = run(candles([*flat(N30 - 1), (104, 106, 103, 105)]))
    expect(
        "neither: time exit at the window end",
        r,
        {
            "hit10": False,
            "hit20": False,
            "hit_ns": False,
            "reason_10": "time",
            "reason_20": "time",
            "reason_ns": "time",
            "exit_t_10": D0 + WINDOW_US,
            "exit_price_10": 105.0,
            "ret_10": net(100, 105, True),
            "ret_ns": net(100, 105, True),
            "h_target": None,
            "peak30": 0.06,
            "trough30": -0.01,
        },
    )
    # Each stop variant: -15% then -21% then flat at 90.
    r = run(candles([*flat(1), (100, 100, 85, 88), (88, 88, 79, 90), *flat(N30 - 3, 90)]))
    expect(
        "stop variants: 10% stops, then 20%, no-stop holds to the end",
        r,
        {
            "reason_10": "stop",
            "exit_price_10": 90.0,
            "exit_t_10": D0 + FIVE,
            "reason_20": "stop",
            "exit_price_20": 80.0,
            "exit_t_20": D0 + 2 * FIVE,
            "reason_ns": "time",
            "exit_price_ns": 90.0,
            "ret_20": net(100, 80, True),
            "ret_ns": net(100, 90, True),
            "h_stop20": 2 * FIVE / HOUR,
        },
    )
    # A gap below the stop fills at the open.
    r = run(candles([*flat(1), (85, 86, 84, 85), *flat(N30 - 2, 85)]))
    expect(
        "gap through the stop fills at the open",
        r,
        {"reason_10": "stop", "exit_price_10": 85.0, "ret_10": net(100, 85, True)},
    )
    # Entry at the first open at or after D; levels relative to P; candles before D ignored.
    before = candles([(100, 200, 50, 100)], start=D0 - FIVE)
    after = candles(
        [(110, 129.9, 108, 120), (120, 130, 119, 125), *flat(N30 - 4, 125)], D0 + 10 * 60_000_000
    )
    arrs = tuple(np.concatenate([x, y]) for x, y in zip(before, after, strict=True))
    r = run(arrs)
    expect(
        "entry at the first open at or after D, levels relative to P",
        r,
        {
            "entry_t": D0 + 10 * 60_000_000,
            "entry_open": 110.0,
            "hit10": True,
            "exit_t_10": D0 + 10 * 60_000_000 + FIVE,
            "exit_price_10": 130.0,
            "ret_10": net(110, 130, False),
            "peak30": 0.30,
            "trough30": 0.08,
            "n_candles": N30 - 2,
        },
    )
    # Delisting: candles stop after 10 days.
    n10 = 10 * DAY // FIVE
    r = run(candles([*flat(n10 - 1), (100, 101, 97, 98)]))
    expect(
        "delisting: exit at the last close, flagged",
        r,
        {
            "stopped_early": True,
            "reason_10": "series_end",
            "reason_ns": "series_end",
            "exit_t_10": D0 + n10 * FIVE,
            "exit_price_10": 98.0,
            "ret_ns": net(100, 98, True),
            "hit10": False,
        },
    )
    # A short gap (missing candles) inside the window is not a delisting.
    gap = candles(flat(100))
    late = candles([*flat(10), (100, 131, 99, 130)], start=D0 + 20 * DAY)
    r = run(tuple(np.concatenate([x, y]) for x, y in zip(gap, late, strict=True)))
    expect(
        "missing candles are just absent",
        r,
        {"hit10": True, "exit_t_10": D0 + 20 * DAY + 10 * FIVE, "h_target": 20 * 24 + 50 / 60},
    )
    r = run(candles(flat(5), start=D0 + WINDOW_US))
    expect("no candle in the window: no_entry", r, {"reason_10": "no_entry", "hit10": None})
    # Peaks: a spike on day 8 counts for 30 days only.
    rows = flat(N30)
    rows[8 * DAY // FIVE] = (100, 120, 99, 100)
    rows[2 * DAY // FIVE] = (100, 110, 95, 100)
    r = run(candles(rows))
    expect("peak7 / peak30 windows", r, {"peak7": 0.10, "trough7": -0.05, "peak30": 0.20})


def logic_lookahead() -> None:
    rng = np.random.default_rng(SEED)
    n = 40 * DAY // FIVE
    t = D0 - 5 * DAY + FIVE * np.arange(n, dtype=np.int64)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.concatenate(([100.0], c[:-1]))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    base = label_entry(t, o, h, lo, c, D0, float(c[t < D0][-1]))
    after = t >= D0 + WINDOW_US
    h2, lo2, c2, o2 = h.copy(), lo.copy(), c.copy(), o.copy()
    h2[after] *= 5
    lo2[after] *= 0.1
    c2[after] *= 3
    o2[after] *= 0.2
    changed = label_entry(t, o2, h2, lo2, c2, D0, float(c[t < D0][-1]))
    cut = ~after
    trunc = label_entry(t[cut], o[cut], h[cut], lo[cut], c[cut], D0, float(c[t < D0][-1]))
    bad = same_dict(base, changed) + same_dict(base, trunc)
    check("look-ahead: candles after the window change no label (synthetic)", not bad, str(bad))

    # Panel: 120 days of candles with volume; every row is unchanged when candles at or after
    # its D are dropped or altered.
    n = 120 * DAY // FIVE
    start = D0 - 100 * DAY
    t = start + FIVE * np.arange(n, dtype=np.int64)
    keep = rng.uniform(size=n) > 0.02  # some candles missing
    t = t[keep]
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(t))))
    qv = rng.uniform(0, 8_000, len(t))
    full = panel_rows("SYN", t, close, qv, None)
    bad_rows = 0
    for r in full.iter_rows(named=True):
        d = r["d_us"]
        m = t < d
        cut_rows = panel_rows("SYN", t[m], close[m], qv[m], None).filter(pl.col("d_us") == d)
        alt = np.where(t >= d, 1e9, qv)
        alt_c = np.where(t >= d, 1.0, close)
        alt_rows = panel_rows("SYN", t, alt_c, alt, None).filter(pl.col("d_us") == d)
        want = full.filter(pl.col("d_us") == d)
        if not (cut_rows.equals(want) and alt_rows.equals(want)):
            bad_rows += 1
    check(
        f"look-ahead: candles at or after D change no panel row (synthetic, {full.height} rows)",
        full.height > 30 and bad_rows == 0,
        f"{bad_rows} differ",
    )
    # Hand-computed panel row: liq is the mean of the 30 days before D, P the last close.
    r0 = full.row(0, named=True)
    d = r0["d_us"]
    m = (t >= d - 30 * DAY) & (t < d)
    ok = same(r0["liq"], float(qv[m].sum() / 30)) and r0["P"] == float(close[t < d][-1])
    ok = ok and r0["days_listed"] == (d - t[0] // DAY * DAY) // DAY and r0["liq"] >= MIN_LIQUIDITY
    check("panel row by hand (liq, P, days_listed)", ok, str(r0))
    # Stale price: no candle in the day before D, no entry.
    m = (t < d - DAY - FIVE) | (t >= d)
    rows = panel_rows("SYN", t[m], close[m], qv[m] * 10, None)
    check("panel: no entry without a candle in the day before D", d not in rows["d_us"].to_list())


# --- 2. panel on the real data ---------------------------------------------------------------


def real_panel() -> pl.DataFrame:
    panel = pl.read_parquet(PANEL)
    check("panel schema", dict(panel.schema) == PANEL_SCHEMA, str(panel.schema))
    check(
        "panel keys unique",
        panel.select("symbol", "d_us").n_unique() == panel.height,
    )
    ok = (
        (panel["d_us"] % DAY == 0).all()
        and int(panel.select(pl.col("d_us").min()).item()) >= FIRST_D_US
        and int(panel.select(pl.col("d_us").max()).item()) <= LAST_D_US
        and (panel["liq"] >= MIN_LIQUIDITY).all()
        and (panel["days_listed"] >= 30).all()
        and (panel["P"] > 0).all()
        and panel.null_count().sum_horizontal().item() == 0
    )
    check("panel invariants (midnights, range, floor, age, P > 0, no nulls)", bool(ok))

    # liq against ma_rebound's daily bars (same definition, computed independently there).
    compared, worst, missing = 0, 0.0, 0
    for sym, g in panel.group_by("symbol"):
        path = MA_BARS_1D / f"{sym[0]}.parquet"
        if not path.exists():
            missing += 1
            continue
        bars = pl.read_parquet(path, columns=["t", "liq"]).filter(pl.col("liq").is_not_null())
        j = g.join(bars, left_on="d_us", right_on="t", how="inner")
        compared += j.height
        if j.height:
            rel = ((j["liq"] - j["liq_right"]).abs() / j["liq_right"]).max()
            worst = max(worst, float(rel))  # type: ignore[arg-type]
    check(
        f"liq equals ma_rebound's daily liq ({compared} rows, max rel diff {worst:.1e})",
        compared > 0.95 * panel.height and worst < 1e-9 and missing == 0,
        f"missing={missing}",
    )
    # Panel rows with no candle on day D itself have no ma_rebound bar to compare: that is the
    # only gap allowed above (under 5%).

    rng = random.Random(SEED)
    coins = rng.sample(sorted(panel["symbol"].unique().to_list()), 6)
    bad, n = 0, 0
    for sym in coins:
        c5 = read_5m(sym)
        rows = panel.filter(pl.col("symbol") == sym)
        pre = first_day_pre2020(sym)
        for d in rng.sample(rows["d_us"].to_list(), min(8, rows.height)):
            cut = c5.filter(pl.col("t") < d)
            got = panel_rows(
                sym,
                cut["t"].to_numpy(),
                cut["close"].to_numpy(),
                cut["quote_volume"].to_numpy(),
                pre,
            ).filter(pl.col("d_us") == d)
            n += 1
            if not got.equals(rows.filter(pl.col("d_us") == d)):
                bad += 1
    check(f"look-ahead: panel rows with candles cut at D (real, {n} rows)", n > 0 and bad == 0)
    return panel


# --- 3. discovery labels -----------------------------------------------------------------------


def discovery_labels(panel: pl.DataFrame) -> None:
    labels = pl.read_parquet(LABELS)
    want_schema = {**KEY_SCHEMA, **LABEL_SCHEMA}
    check("labels schema", dict(labels.schema) == want_schema, str(labels.schema))
    check(
        "labels hold no entry from 2024 on",
        int(labels.select(pl.col("d_us").max()).item()) < SEALED_START_US,
    )
    expect_keys = panel.filter(pl.col("d_us") < SEALED_START_US).select("symbol", "d_us")
    got_keys = labels.select("symbol", "d_us")
    check(
        f"labels: one row per discovery and gap panel entry ({labels.height})",
        got_keys.sort(pl.all()).equals(expect_keys.sort(pl.all())),
    )
    has = labels.filter(pl.col("n_candles") > 0)
    inv = has.select(
        ok1=(pl.col("hit10") == (pl.col("reason_10") == "target")).all(),
        ok2=(pl.col("hit20") == (pl.col("reason_20") == "target")).all(),
        ok3=(pl.col("hit_ns") == (pl.col("reason_ns") == "target")).all(),
        ok4=(~pl.col("hit10") | pl.col("hit20")).all(),
        ok5=(~pl.col("hit20") | pl.col("hit_ns")).all(),
        ok6=(pl.col("hit_ns") == pl.col("h_target").is_not_null()).all(),
        ok7=((pl.col("peak30") >= pl.col("peak7")) | pl.col("peak7").is_null()).all(),
        ok8=(pl.col("trough30") <= 0.0 + pl.col("peak30")).all(),
        ok9=(pl.col("entry_t") >= pl.col("d_us")).all(),
        ok10=(pl.col("exit_t_10") <= pl.col("d_us") + WINDOW_US).all(),
    ).row(0)
    check("label invariants (hits vs reasons, nesting, peaks, times)", all(inv), str(inv))
    none = labels.filter(pl.col("n_candles") == 0)
    check(
        "entries without candles have reason no_entry",
        bool((none["reason_10"] == "no_entry").all()),
    )

    rng = random.Random(SEED + 1)
    disc = labels.filter(pl.col("d_us") <= DISCOVERY_END_US, pl.col("n_candles") > 0)
    picks = disc.sample(n=6, seed=SEED)
    by_coin: dict[str, pl.DataFrame] = {}
    bad: list[str] = []
    for r in picks.iter_rows(named=True):
        sym = r["symbol"]
        if sym not in by_coin:
            by_coin[sym] = read_5m(sym)
        c5 = by_coin[sym]
        t = c5["t"].to_numpy()
        o, h, lo, c = (c5[k].to_numpy() for k in ("open", "high", "low", "close"))
        p = float(panel.filter(pl.col("symbol") == sym, pl.col("d_us") == r["d_us"])["P"][0])
        d = r["d_us"]
        full = label_entry(t, o, h, lo, c, d, p)
        m = t < d + WINDOW_US
        after = ~m
        h2, lo2 = h.copy(), lo.copy()
        h2[after] = h2[after] * 10 + rng.random()
        lo2[after] = lo2[after] * 0.01
        cut = label_entry(t[m], o[m], h[m], lo[m], c[m], d, p)
        alt = label_entry(t, o, h2, lo2, c, d, p)
        bad += same_dict(full, cut) + same_dict(full, alt)
    check("look-ahead: real candles after the window change no label (6 entries)", not bad)

    spot_check(disc.sample(n=20, seed=SEED), panel)


def slow_candles(symbol: str, start: int, end: int) -> list[tuple[int, float, float, float, float]]:
    """Raw 5m candles opening in [start, end), read straight from the month files."""
    folder = KLINES_5M / f"symbol={symbol}"
    out: list[tuple[int, float, float, float, float]] = []
    for f in sorted(folder.glob("*.parquet")):
        df = pl.read_parquet(f, columns=["open_time", "open", "high", "low", "close"])
        for row in df.iter_rows():
            ts = int(row[0].timestamp() * 1_000_000)
            if start <= ts < end:
                out.append((ts, row[1], row[2], row[3], row[4]))
    out.sort()
    return out


def slow_label(symbol: str, d: int) -> dict[str, Value]:
    """The labels of one entry, one candle at a time, P included."""
    rows = slow_candles(symbol, d - 2 * DAY, d + WINDOW_US)
    p = None
    for ts, _o, _h, _l, c in rows:
        if ts < d:
            p = c
    assert p is not None
    win = [r for r in rows if r[0] >= d]
    res: dict[str, Value] = {"P": p}
    target = 1.3 * p
    for name, stop in (("10", 0.9 * p), ("20", 0.8 * p), ("ns", None)):
        entry = win[0][1]
        reason, exit_px = "", 0.0
        for ts, o, h, low, _c in win:
            if stop is not None and low <= stop:
                reason, exit_px = "stop", min(stop, o)
                res[f"exit_t_{name}"] = ts
                break
            if h >= target:
                reason, exit_px = "target", target
                res[f"exit_t_{name}"] = ts
                break
        if not reason:
            last = win[-1]
            early = last[0] + FIVE < d + WINDOW_US - DAY
            reason, exit_px = ("series_end" if early else "time"), last[4]
            res[f"exit_t_{name}"] = last[0] + FIVE
        slip = 0.0 if reason == "target" else 0.001
        ret = exit_px * (1 - slip) * 0.999 / (entry * 1.001 * 1.001) - 1
        res[f"reason_{name}"] = reason
        res[f"exit_price_{name}"] = exit_px
        res[f"ret_{name}"] = ret
    res["hit10"] = res["reason_10"] == "target"
    res["hit20"] = res["reason_20"] == "target"
    res["hit_ns"] = res["reason_ns"] == "target"
    peak7 = max(h for ts, _o, h, _l, _c in win if ts < d + 7 * DAY)
    res["peak7"] = peak7 / p - 1
    res["peak30"] = max(r[2] for r in win) / p - 1
    res["trough30"] = min(r[3] for r in win) / p - 1
    res["n_candles"] = len(win)
    return res


def spot_check(sample: pl.DataFrame, panel: pl.DataFrame) -> None:
    bad = []
    for r in sample.iter_rows(named=True):
        want = slow_label(r["symbol"], r["d_us"])
        p = panel.filter(pl.col("symbol") == r["symbol"], pl.col("d_us") == r["d_us"])["P"][0]
        got = {**r, "P": p}
        diff = [k for k in want if not same(got[k], want[k])]
        if diff:
            bad.append(f"{r['symbol']}@{r['d_us']}: {diff}")
    check(
        f"spot check: {sample.height} random discovery entries from raw candles", not bad, str(bad)
    )


# --- 4. sealed labels: shape only --------------------------------------------------------------


def sealed_shape(panel: pl.DataFrame) -> None:
    schema = dict(pl.read_parquet_schema(SEALED_LABELS))
    check("sealed labels schema", schema == {**KEY_SCHEMA, **LABEL_SCHEMA})
    keys = pl.read_parquet(SEALED_LABELS, columns=["symbol", "d_us"])  # keys only
    want = panel.filter(pl.col("d_us") >= SEALED_START_US).select("symbol", "d_us")
    check(
        f"sealed labels: one row per panel entry from 2024 on ({keys.height})",
        keys.null_count().sum_horizontal().item() == 0
        and keys.sort(pl.all()).equals(want.sort(pl.all())),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logic-only", action="store_true", help="synthetic checks only")
    args = parser.parse_args()
    steps: list[Callable[[], object]] = [logic_labels, logic_lookahead]
    for step in steps:
        step()
    if not args.logic_only:
        panel = real_panel()
        discovery_labels(panel)
        sealed_shape(panel)
    print(f"{len(FAILURES)} failed" if FAILURES else "all checks passed")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
