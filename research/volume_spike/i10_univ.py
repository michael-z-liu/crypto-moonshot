"""The universe's forward return path per spike row (docs/plans/cnn-improvements.md, M4).

For every row of `seq2_meta.parquet` the equal-weight mean forward return of the universe over
exactly the window and in exactly the units of the row's path target, so the `xs_univ` variant
can train on target - offset (the coin's excess return over the market) via
`i10_train.I10Cfg.target_offset="wf_i10/univ_fwd.npy"`.

    cd research/volume_spike
    export MOONSHOT_UNIVERSE=all
    uv run --group research python i10_univ.py build    # -> cache_all/wf_i10/univ_fwd.npy (+ _meta)
    uv run --group research python i10_univ.py check    # the checks below, printed

**The row's path target** (m12 `Data.path`): `fwd2[:, 2, PATH_STEPS]`, built by
`build_seq2.fwd_job` on the coin's gap-free 5m grid (`full_grid(load(symbol))`, missing closes
carried forward, missing opens / highs / lows set to that close). With `g` the spike candle's grid
index and `e = g + 1` the entry bar (the candle after the spike), `fwd2[:, 2, k] =
close[e + k] / open[e] - 1`, NaN where `e + k` is past the series' end. `PATH_STEPS` = 2, 5, ..,
95 (the closes of bars 3, 6, .., 96 after the spike: 15m .. 8h). The target is
`clip(fwd2[:, 2, PATH_STEPS] - 0.003, -0.3, 0.3) * 100` (percent, net of the round trip).

**The offset** (`univ_fwd.npy`, float32 (rows, 32), percent): for each row and step k,
`100 * mean_c clip(close_c[e + k] / open_c[e] - 1, -0.3, 0.3)` over the universe coins c, each
on its own grid built exactly as above (same reference price, the open of the entry bar, and the
same bar alignment). The universe is `common.universe()` with `MOONSHOT_UNIVERSE=all` (582 USDT
pairs incl. delisted, pegged coins and tokenised stocks excluded). The row's own coin is part of
it (one of about 300 coins at a time).

- A coin counts at a step when its grid covers both the entry bar and the step bar. A coin listed
  after the entry bar is out of the whole row (it has no reference price); one listed during the
  spike candle itself or earlier is in. The coin's own target is undefined in the same cases.
- A missing candle inside a coin's grid takes the last close (zero return over the gap), exactly
  as the coin's own target. A delisted coin's series ends with `common.load`'s 300 flat candles
  (25h) at its last close, so a coin delisted inside a window counts with a flat price to the
  window's end; past those it drops out at the steps it has no value for. So the coin set can
  differ slightly between the steps of a row.
- Each coin's return is clipped to +-30% before averaging (the bound of the target's own clip),
  so a bad tick in an archived series cannot move the mean; the build prints how often it binds.
- **No cost.** The offset is the gross market return. target - offset = clip(r - 0.003) - market,
  i.e. the coin's excess return net of its own round trip; subtracting the cost twice would only
  shift the target by a constant.
- The values use candles from the entry bar to the step bar only (and, when a coin's entry candle
  is missing, the last close before it, the price known at entry). This is a forward-looking
  **label**, like the target itself: it is only ever subtracted from the training target and is
  never a model input, so the look-ahead is intended. `check` shows it is unchanged when every
  candle before the entry bar or after the window is altered.

Outputs (local, never a shared cache): `cache_all/wf_i10/univ_fwd.npy` and
`cache_all/wf_i10/univ_fwd_meta.npz` (the unique spike grid indices `gidx`, the coin count per
(gidx, step) `count`, and the number of clipped coin returns `clipped`).
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
from build_events import full_grid
from common import CACHE, DATA_END, KLINES_5M, PAD_BARS, load, universe
from i10_train import WF_I10, _new_file

T0 = datetime(2020, 1, 1, tzinfo=UTC)  # build_seq2's global grid origin
STEP = timedelta(minutes=5)
PATH_STEPS = np.arange(3, 97, 3) - 1  # m12.PATH_STEPS (checked in `check`)
FWD = 96
CLIP = 0.3
COST = 0.003
OUT = WF_I10 / "univ_fwd.npy"
OUT_META = WF_I10 / "univ_fwd_meta.npz"
WORKERS = 4
CHUNKS = 16


def gidx_of(t: pl.Series) -> np.ndarray:
    """Global 5m grid index (from T0) of open times."""
    return ((t - T0) / STEP).cast(pl.Int64).to_numpy()


def arrays_of(df: pl.DataFrame) -> tuple[int, np.ndarray, np.ndarray]:
    """Grid offset, open and close of one symbol's candles, filled as `build_seq2.grid`."""
    g = full_grid(df).with_columns(pl.col("close").forward_fill())
    g = g.with_columns(pl.col("open").fill_null(pl.col("close")))
    off = int((g["open_time"][0] - T0) / STEP)
    return off, g["open"].cast(pl.Float64).to_numpy(), g["close"].cast(pl.Float64).to_numpy()


def coin_returns(
    off: int, op: np.ndarray, cl: np.ndarray, gidx: np.ndarray, steps: np.ndarray = PATH_STEPS
) -> np.ndarray:
    """close[e + k] / open[e] - 1 for spike grid indices `gidx` (e = gidx + 1 - off), NaN where
    the grid does not cover both the entry bar and the step bar. (len(gidx), len(steps))."""
    e = gidx + 1 - off
    k = e[:, None] + steps[None, :]
    ok = (e[:, None] >= 0) & (k < len(cl))
    ee = np.clip(e, 0, len(op) - 1)
    kk = np.clip(k, 0, len(cl) - 1)
    return np.where(ok, cl[kk] / op[ee][:, None] - 1, np.nan)


def _chunk(job: tuple[list[str], np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """Sum and count of the clipped coin returns over `symbols`, per unique gidx and step."""
    symbols, ug = job
    s = np.zeros((len(ug), len(PATH_STEPS)))
    n = np.zeros((len(ug), len(PATH_STEPS)), dtype=np.int32)
    clipped = 0
    for sym in symbols:
        off, op, cl = arrays_of(load(sym))
        lo = np.searchsorted(ug, off - 1)  # entry bar at or after the grid start
        hi = np.searchsorted(ug, off + len(cl) - 1)  # entry bar inside the grid
        if hi <= lo:
            continue
        r = coin_returns(off, op, cl, ug[lo:hi])
        ok = np.isfinite(r)
        clipped += int((ok & (np.abs(r) > CLIP)).sum())
        s[lo:hi] += np.where(ok, np.clip(r, -CLIP, CLIP), 0.0)
        n[lo:hi] += ok
    return s, n, clipped


def build() -> None:
    t0 = time.time()
    meta = pl.read_parquet(CACHE / "seq2_meta.parquet", columns=["symbol", "open_time"])
    gidx = gidx_of(meta["open_time"])
    ug, inv = np.unique(gidx, return_inverse=True)
    syms = universe()
    jobs = [(syms[i::CHUNKS], ug) for i in range(CHUNKS)]
    s = np.zeros((len(ug), len(PATH_STEPS)))
    n = np.zeros((len(ug), len(PATH_STEPS)), dtype=np.int32)
    clipped = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for cs, cn, cc in pool.map(_chunk, jobs):
            s += cs
            n += cn
            clipped += cc
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan) * 100
    out = mean[inv].astype(np.float32)
    np.save(_new_file(OUT), out)
    np.savez(_new_file(OUT_META), gidx=ug, count=n.astype(np.int16), clipped=np.array(clipped))
    print(
        f"{len(syms)} coins, {len(meta)} rows, {len(ug)} spike times; {clipped} of"
        f" {int(n.sum())} coin returns clipped at +-{CLIP:.0%}; rows with a NaN step"
        f" {int(np.isnan(out).any(1).sum())}; {time.time() - t0:.0f}s -> {OUT}",
        flush=True,
    )


# ----------------------------------------------------------------------------- checks


def _files(sym: str, start: datetime, end: datetime) -> list[Path]:
    d = KLINES_5M / f"symbol={sym}"
    months = pl.datetime_range(
        start.replace(day=1, hour=0, minute=0), end, "1mo", eager=True, time_zone="UTC"
    )
    return [p for m in months if (p := d / f"{m:%Y-%m}.parquet").exists()]


@cache
def _bounds(sym: str) -> tuple[datetime, datetime]:
    """First and last raw candle open time of a symbol (from its first / last month file)."""
    files = sorted((KLINES_5M / f"symbol={sym}").glob("*.parquet"))
    first = pl.read_parquet(files[0], columns=["open_time"])["open_time"].min()
    last = pl.read_parquet(files[-1], columns=["open_time"])["open_time"].max()
    return first, last  # type: ignore[return-value]


def _us(t: datetime) -> int:
    return (t - T0) // timedelta(microseconds=1)


def hand_value(sym: str, entry: datetime) -> np.ndarray:
    """One coin's step returns for an entry bar opening at `entry`, straight from the raw month
    files and the rules in the docstring (independent of `arrays_of` / `coin_returns`: no grid,
    lookups by time). NaN where the coin does not count."""
    out = np.full(len(PATH_STEPS), np.nan)
    first, last = _bounds(sym)
    if first > entry:
        return out
    wend = entry + STEP * FWD
    files = _files(sym, entry - timedelta(days=31), wend)
    if not files:
        return out
    df = pl.concat([pl.read_parquet(p, columns=["open_time", "open", "close"]) for p in files])
    df = df.filter(pl.col("open_time") <= wend).sort("open_time")
    t = np.array([_us(x) for x in df["open_time"].to_list()], dtype=np.int64)
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    i = int(np.searchsorted(t, _us(entry)))
    if i < len(t) and t[i] == _us(entry):
        ref = float(o[i])  # the entry candle's open
    elif i > 0:
        ref = float(c[i - 1])  # no entry candle: the last close before it
    else:
        return out  # a halt longer than a month: not resolved here
    end_valid = last + STEP * PAD_BARS if last < DATA_END else last
    for j, k in enumerate(PATH_STEPS):
        tk = entry + STEP * int(k)
        if tk > end_valid:
            continue
        u = int(np.searchsorted(t, _us(tk), side="right")) - 1  # last candle at or before tk
        out[j] = float(c[u]) / ref - 1  # the close carried forward (u = i - 1: ref itself)
    return out


def _expect(ok: object, what: str) -> None:
    if not ok:
        raise AssertionError(f"check failed: {what}")


def check_hand(rows: np.ndarray, meta: pl.DataFrame, univ: np.ndarray) -> None:
    syms = universe()
    for r in rows:
        spike = meta["open_time"][int(r)]
        entry = spike + STEP
        vals = np.array([hand_value(s, entry) for s in syms])
        cnt = np.isfinite(vals).sum(0)
        hand = np.nanmean(np.clip(vals, -CLIP, CLIP), 0) * 100
        d = np.abs(hand - univ[r])
        print(
            f"  row {r} ({meta['symbol'][int(r)]} {spike:%Y-%m-%d %H:%M}): coins at 1h / 8h"
            f" {cnt[3]} / {cnt[-1]}; hand 1h {hand[3]:+.4f} vs built {univ[r, 3]:+.4f};"
            f" max abs diff over 32 steps {d.max():.2e}"
        )
        _expect(d.max() < 1e-4, f"hand check failed on row {r}")


def check_own(syms: list[str], meta: pl.DataFrame) -> None:
    fwd = np.load(CACHE / "fwd2.npy", mmap_mode="r")
    gidx = gidx_of(meta["open_time"])
    msym = meta["symbol"].to_numpy()
    tot = diff_rows = only_one = 0
    worst = 0.0
    for s in syms:
        rows = np.flatnonzero(msym == s)
        off, op, cl = arrays_of(load(s))
        mine = coin_returns(off, op, cl, gidx[rows])
        ref = np.asarray(fwd[rows, 2][:, PATH_STEPS], dtype=np.float64)
        both = np.isfinite(mine) & np.isfinite(ref)
        only_one += int((np.isfinite(mine) != np.isfinite(ref)).sum())
        d = np.abs(mine - ref)[both]
        worst = max(worst, float(d.max()) if d.size else 0.0)
        diff_rows += int((np.abs(mine - ref) > 1e-6)[both].sum())
        tot += int(both.sum())
    print(
        f"  {len(syms)} symbols, {tot} (row, step) values: max abs diff vs fwd2 {worst:.2e}"
        f" (float32 storage), values off by > 1e-6: {diff_rows}; finite in one only: {only_one}"
    )
    _expect(worst < 1e-5 and diff_rows == 0, "own-return check failed")


def check_coverage(meta: pl.DataFrame, univ: np.ndarray) -> None:
    with np.load(OUT_META) as f:
        ug, count, clipped = f["gidx"], f["count"], int(f["clipped"])
    inv = np.searchsorted(ug, gidx_of(meta["open_time"]))
    c1h, c8h = count[inv, 3].astype(np.float64), count[inv, -1].astype(np.float64)
    year = meta["open_time"].dt.year().to_numpy()
    print("  coins per row at 1h (min / p1 / median / max) and 8h median, by year:")
    for y in np.unique(year):
        m = year == y
        q = np.percentile(c1h[m], [0, 1, 50, 100])
        print(
            f"    {y}: {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} / {q[3]:.0f};"
            f" 8h {np.median(c8h[m]):.0f}; rows {m.sum()}"
        )
    path = np.clip(
        np.load(CACHE / "fwd2.npy", mmap_mode="r")[:, 2][:, PATH_STEPS] - COST, -CLIP, CLIP
    )
    fin = np.isfinite(path)
    bad = int((fin & ~np.isfinite(univ)).sum())
    print(
        f"  target finite but offset NaN: {bad} (row, step) values; target NaN (data end):"
        f" {int((~fin).any(1).sum())} rows; clipped coin returns: {clipped}"
    )
    _expect(bad == 0, "offset missing where the target exists")
    y = path[:, 3] * 100
    u = univ[:, 3]
    ok = np.isfinite(y) & np.isfinite(u)
    cc = np.corrcoef(y[ok], u[ok])[0, 1]
    print(
        f"  1h: target sd {y[ok].std():.3f} pp, offset sd {u[ok].std():.3f} pp, corr {cc:.3f},"
        f" sd of target - offset {(y[ok] - u[ok]).std():.3f} pp"
    )


def check_lookahead(rows: np.ndarray, meta: pl.DataFrame, rng: np.random.Generator) -> None:
    """Alter every candle before the entry bar and after the window in the coins' raw candles,
    rebuild them with the builder's own code and compare the rows' returns."""
    syms = universe()
    changed = missing_entry = coins = 0
    for r in rows:
        spike = meta["open_time"][int(r)]
        entry, wend = spike + STEP, spike + STEP * (FWD + 1)
        g = gidx_of(meta["open_time"][int(r) : int(r) + 1])
        for s in rng.choice(syms, 40, replace=False):
            df = load(str(s))
            base = coin_returns(*arrays_of(df), g)
            if not np.isfinite(base).any():
                continue
            coins += 1
            f = pl.Series(rng.uniform(0.5, 2.0, len(df)))
            outside = (pl.col("open_time") < entry) | (pl.col("open_time") > wend)
            alt = df.with_columns(
                [pl.when(outside).then(pl.col(c) * f).otherwise(pl.col(c)).alias(c)
                 for c in ("open", "high", "low", "close")]
            )  # fmt: skip
            new = coin_returns(*arrays_of(alt), g)
            has_entry = df.filter(pl.col("open_time") == entry).height > 0
            missing_entry += int(not has_entry)
            same = np.allclose(base, new, equal_nan=True, rtol=0, atol=1e-12)
            if not same:
                changed += 1
                _expect(not has_entry, f"{s} row {r}: changed although its entry candle exists")
    print(
        f"  {len(rows)} rows x 40 coins ({coins} with values): returns changed by altering"
        f" candles outside [entry, window end] for {changed} coins; coins without an entry"
        f" candle (reference = last close before it): {missing_entry}"
    )


def check() -> None:
    import m12_cnn_wf as m12

    _expect(np.array_equal(PATH_STEPS, m12.PATH_STEPS), "PATH_STEPS differ from m12")
    meta = pl.read_parquet(CACHE / "seq2_meta.parquet", columns=["symbol", "open_time"])
    univ = np.load(OUT, mmap_mode="r")
    _expect(univ.shape == (len(meta), len(PATH_STEPS)) and univ.dtype == np.float32, "shape")
    print(f"univ_fwd: shape {univ.shape} {univ.dtype}")
    rng = np.random.default_rng(10)
    hand = rng.choice(len(meta), 6, replace=False)
    print("1. hand computation from the raw month files (all 582 coins per row):")
    check_hand(np.sort(hand), meta, np.asarray(univ))
    print("2. the builder's own-coin returns vs fwd2[:, 2, PATH_STEPS] (all rows of each symbol):")
    syms = sorted(meta["symbol"].unique().to_list())
    check_own([str(s) for s in rng.choice(syms, 40, replace=False)], meta)
    print("3. coverage:")
    check_coverage(meta, np.asarray(univ))
    print("4. look-ahead: candles outside [entry bar, window end] altered:")
    check_lookahead(np.sort(rng.choice(len(meta), 5, replace=False)), meta, rng)
    print("all checks passed")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "check":
        check()
    else:
        raise SystemExit(__doc__)
