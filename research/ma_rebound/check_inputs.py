"""M4 checks of the model inputs and the folds: PASS/FAIL lines.

    uv run --group research python check_inputs.py [--runs base gbm] [--skip-market]

1. **Truncation** (no look-ahead): for sampled candidates (BTC, ETH and random coins, both
   families, every timeframe, every year), the coin's and BTC's candles are cut at the
   candidate's decision time (only candles that closed by then), the bars are rebuilt from them,
   the breadth table is cut there, and the candidate's inputs are rebuilt: the fine and coarse
   windows (as stored, float16), the scalars and the other-MA distances must be identical.
2. **Breadth causality**: the breadth table rebuilt from every coin's candles cut at a time T
   equals the stored table up to T, bit for bit.
3. **Invariants**: the last fine close and the forming bar's close are the candidate's
   `distance`, the level path ends at 0, the other-MA distance of the candidate's own MA is its
   `distance`; at the last 5m candle of a bar the forming aggregates equal the completed bar;
   no NaN or infinity anywhere in the arrays; the float16 precision against float32.
4. **Folds** (per run in `cache/m4/scores/<run>/folds.npz`): every fold's train, validation and
   test rows are recomputed independently from the M1 labels (not through `train.py`) and must
   be the saved sets; train rows are decided and resolved before Y-1-07-01, validation rows are
   decided in [Y-1-07-01, Y-01-01) and resolved before Y-01-01, test rows are decided in Y; the
   three sets are disjoint; the scores cover every candidate of the family decided in 2022-26.

Only PASS/FAIL lines and counts are printed: nothing about an outcome of a 2025-26 fill.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import build_inputs as bi
import numpy as np
import polars as pl
from common import CACHE, TIMEFRAMES, universe
from events import FIVE, bar_index

SEED = 20261113
US = 1_000_000
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" ({detail})" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


# --- 1. truncation ---------------------------------------------------------------------------


def sample_rows(rng: np.random.Generator) -> pl.DataFrame:
    rows = bi.rows()
    coins = ["BTCUSDT", "ETHUSDT"]
    others = sorted(set(rows["symbol"].unique().to_list()) - set(coins))
    coins += list(rng.choice(others, 6, replace=False))
    picks = []
    for sym in coins:
        r = rows.filter(pl.col("symbol") == sym)
        for fam in ("limit", "touch"):
            for tf in TIMEFRAMES:
                g = r.filter(pl.col("family") == fam, pl.col("tf") == tf)
                if g.height:
                    picks.append(g.sample(1, seed=int(rng.integers(1 << 30))))
    out = pl.concat(picks)
    # every timeframe and family covered: add random rows of any (tf, family) still missing
    for fam in ("limit", "touch"):
        for tf in TIMEFRAMES:
            if out.filter(pl.col("family") == fam, pl.col("tf") == tf).height < 3:
                g = rows.filter(pl.col("family") == fam, pl.col("tf") == tf)
                out = pl.concat([out, g.sample(min(3, g.height), seed=int(rng.integers(1 << 30)))])
    return out.unique("row").sort("row")


def _truncated(row: dict[str, object]) -> tuple[int, bool, str]:
    d = int(str(row["decision_t"]))
    coin = bi.load_coin(str(row["symbol"]), d)
    btc = bi.load_coin(bi.BTC, d)
    mk = bi.market().cut(d)
    one = bi.rows().filter(pl.col("row") == row["row"])
    inp = bi.coin_inputs(coin, one, btc, mk)
    arrs = bi.open_arrays("r")
    k = int(str(row["row"]))
    same = {
        "fine": np.array_equal(inp.fine[0].astype(np.float16), arrs["fine"][k]),
        "coarse": np.array_equal(inp.coarse[0].astype(np.float16), arrs["coarse"][k]),
        "scalars": np.array_equal(inp.scalars[0], arrs["scalars"][k], equal_nan=True),
        "other_ma": np.array_equal(inp.other_ma[0], arrs["other_ma"][k], equal_nan=True),
    }
    bad = [key for key, v in same.items() if not v]
    return k, not bad, ",".join(bad)


def truncation(rng: np.random.Generator, workers: int) -> None:
    sample = sample_rows(rng)
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        res = list(pool.map(_truncated, sample.iter_rows(named=True)))
    bad = [(k, why) for k, ok, why in res if not ok]
    years = sorted({int(str(y)[:4]) for y in sample["cid"].str.slice(-12, 4).to_list()})
    cover = sample.group_by("family", "tf").len().sort("family", "tf")
    check(
        "truncation: inputs rebuilt from candles cut at the decision are identical",
        not bad,
        f"{len(res)} candidates, {sample['symbol'].n_unique()} coins, trigger years {years},"
        f" (family, tf) counts {[tuple(r) for r in cover.iter_rows()]};"
        f" {time.perf_counter() - t0:.0f}s" + (f"; differing {bad[:5]}" if bad else ""),
    )


# --- 2. breadth causality ----------------------------------------------------------------------


def _part_cut(args: tuple[str, int]) -> pl.DataFrame:
    return bi.market_part(*args)


def market_causality(cuts: list[int], workers: int) -> None:
    symbols = universe()
    full = bi.market()
    ctx = mp.get_context("spawn")
    for cut in cuts:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
            parts = list(pool.map(_part_cut, [(s, cut) for s in symbols]))
        mk = bi.sum_market(parts).cut(cut)
        ref = full.cut(cut)
        ok = (
            np.array_equal(mk.index, ref.index)
            and np.array_equal(mk.above, ref.above, equal_nan=True)
            and np.array_equal(mk.n_liquid, ref.n_liquid)
        )
        when = np.datetime64(cut, "us")
        check(
            f"breadth rebuilt from candles cut at {when} equals the stored table up to then",
            ok,
            f"{len(ref.index)} grid points, {time.perf_counter() - t0:.0f}s",
        )


# --- 3. invariants ---------------------------------------------------------------------------


def invariants(rng: np.random.Generator) -> None:
    arrs = bi.open_arrays("r")
    rows = bi.rows()
    k = np.sort(rng.choice(rows.height, 20_000, replace=False))
    r = rows[k]
    dist = r["distance"].to_numpy()
    fine = arrs["fine"][k].astype(np.float32)
    coarse = arrs["coarse"][k].astype(np.float32)
    ci = bi.COARSE_CH.index
    fi = bi.FINE_CH.index
    tol = np.maximum(1e-3 * np.abs(dist), 2e-3)
    check(
        "last fine close = distance",
        bool((np.abs(fine[:, fi("close"), -1] - dist) <= tol).all()),
        "float16",
    )
    check(
        "forming bar close = distance",
        bool((np.abs(coarse[:, ci("close"), -1] - dist) <= tol).all()),
    )
    check(
        "level path ends at 0, forming bar and trigger candle valid",
        bool(
            (coarse[:, ci("level"), -1] == 0).all()
            and (coarse[:, ci("mask"), -1] == 1).all()
            and (fine[:, fi("mask"), -1] == 1).all()
        ),
    )
    other = arrs["other_ma"][k]
    own = np.array([bi.study_mas().index((tf, n)) for tf, n in zip(r["tf"], r["n"], strict=True)])
    check(
        "other-MA distance of the own MA = distance",
        bool(np.allclose(other[np.arange(len(k)), own], dist, rtol=1e-9, atol=1e-12)),
    )
    # forming aggregates at the last 5m candle of a bar equal the completed bar
    for sym in ("BTCUSDT", "ETHUSDT", "LINKUSDT"):
        coin = bi.load_coin(sym, bi.END_US)
        t = coin.c5["t"].to_numpy().astype(np.int64)
        oks = []
        for tf in TIMEFRAMES:
            g = bar_index(t, tf)
            last = np.flatnonzero(np.concatenate((g[1:] != g[:-1], [False])))
            last = last[t[last] >= bi.GRID0 + 40 * 86_400 * US]  # bars wholly inside the store
            b = coin.bars[tf]
            pos = np.searchsorted(b["idx"].to_numpy(), g[last])
            run = coin.running[tf]
            ok = all(
                np.allclose(run[key][last], b[col].to_numpy()[pos], rtol=1e-12)
                for key, col in (("o", "open"), ("h", "high"), ("lo", "low"), ("q", "quote_volume"))
            )
            oks.append(ok and np.allclose(coin.c5["close"].to_numpy()[last], b["close"][pos]))
        check(
            f"forming bar at a bar's last 5m candle = the completed bar ({sym}, every tf)", all(oks)
        )


def finite_and_precision(rng: np.random.Generator) -> None:
    arrs = bi.open_arrays("r")
    n = bi.rows().height
    t0 = time.perf_counter()
    bad = 0
    big = 0
    for name in ("fine", "coarse"):
        a = arrs[name]
        for s in range(0, n, 50_000):
            x = np.asarray(a[s : s + 50_000])
            bad += int((~np.isfinite(x)).sum())
            big += int((np.abs(x) >= bi.CLIP16).sum())
    for name in ("scalars", "other_ma"):
        x = np.asarray(arrs[name])
        bad_cols = ~np.isfinite(x)
        if name == "scalars":  # bars_since_touch and slope10 may be null (NaN); nothing else
            nullable = [bi.SCALARS.index("bars_since_touch"), bi.SCALARS.index("slope10")]
            bad_cols[:, nullable] = False
        else:
            bad_cols &= ~np.isnan(x)  # missing levels are NaN by design, never infinite
        bad += int(bad_cols.sum())
    check(
        "no NaN or infinity in the windows; scalars null only where allowed",
        bad == 0,
        f"{bad} bad values; {big} window values at the +-60,000 clip;"
        f" {time.perf_counter() - t0:.0f}s",
    )
    # float16 against float32: rebuild 3 coins in float32 and compare the stored rows
    rows = bi.rows()
    syms = ["BTCUSDT", *rng.choice(sorted(set(rows["symbol"].to_list()) - {"BTCUSDT"}), 2)]
    btc = bi.load_coin(bi.BTC, bi.END_US)
    worst_f, worst_c = 0.0, 0.0
    for sym in syms:
        r = rows.filter(pl.col("symbol") == sym)
        coin = btc if sym == bi.BTC else bi.load_coin(sym, bi.END_US)
        inp = bi.coin_inputs(coin, r, btc, bi.market())
        k = r["row"].to_numpy()
        for x32, name in ((inp.fine, "fine"), (inp.coarse, "coarse")):
            x16 = arrs[name][np.sort(k)][np.argsort(np.argsort(k))].astype(np.float32)
            # what the model sees: z-scores clipped at +-10 sd (train.py's normalisation)
            mu = x32.mean(axis=(0, 2), keepdims=True)
            sd = x32.std(axis=(0, 2), keepdims=True) + 1e-6
            z16 = np.clip((x16 - mu) / sd, -10, 10)
            z32 = np.clip((x32 - mu) / sd, -10, 10)
            err = float(np.abs(z16 - z32).max())
            if name == "fine":
                worst_f = max(worst_f, err)
            else:
                worst_c = max(worst_c, err)
    check(
        "float16 storage error is small against each channel's spread",
        max(worst_f, worst_c) < 0.01,
        f"largest error of the clipped z-score (sd units): fine {worst_f:.2e},"
        f" coarse {worst_c:.2e} ({syms})",
    )


# --- 4. folds ----------------------------------------------------------------------------------


def _year_us(y: int, month: int = 1) -> int:
    return int(np.datetime64(f"{y}-{month:02d}-01", "us").astype(np.int64))


def fold_checks(run: str) -> None:
    folder = bi.M4 / "scores" / run
    meta = json.loads((folder / "run.json").read_text())
    cfg = json.loads(meta["config"])
    family, target, pooled = cfg["family"], cfg["target"], cfg["pooled"]
    rows = bi.rows()
    fam = rows.filter(pl.col("family") == family)
    # independent recomputation from the M1 labels
    key = "touch" if family == "touch" else "b0.05"
    lab = pl.read_parquet(CACHE / "m3" / f"lab-{key}-base-E1.parquet").filter(~pl.col("censored"))
    hold = {tf: int(TIMEFRAMES[tf].hold.total_seconds()) * US for tf in TIMEFRAMES}
    df = fam.select("row", "tf", "decision_t", "cid").join(
        lab.select(pl.col("cand").cast(pl.UInt32).alias("row"), "entry_t", "exit_t"),
        on="row",
        how="left",
    )
    df = df.with_columns(end=pl.col("exit_t"), ok=pl.col("exit_t").is_not_null())
    if target == "path":
        fwd = pl.read_parquet(CACHE / "fwd.parquet", columns=["cid", "buffer", "fwd"])
        fwd = fwd.filter(
            pl.col("buffer").is_null() if family == "touch" else pl.col("buffer") == 0.05
        ).drop_nulls("fwd")
        df = df.with_columns(
            ok=pl.col("ok") & pl.col("cid").is_in(fwd["cid"].implode()),
            end=pl.max_horizontal(
                "exit_t",
                pl.col("entry_t") + pl.col("tf").replace_strict(hold, return_dtype=pl.Int64),
            ),
        )
    elif target == "bracket":
        e3 = pl.read_parquet(CACHE / "m3" / f"lab-{key}-base-E3.parquet").filter(
            ~pl.col("censored")
        )
        df = df.join(
            e3.select(pl.col("cand").cast(pl.UInt32).alias("row"), pl.col("exit_t").alias("e3")),
            on="row",
            how="left",
        ).with_columns(
            ok=pl.col("ok") & pl.col("e3").is_not_null(), end=pl.max_horizontal("exit_t", "e3")
        )
    saved = np.load(folder / "folds.npz")
    groups = (
        {"all": tuple(TIMEFRAMES)}
        if pooled
        else {
            "1h+4h": ("1h", "4h"),
            "1d": ("1d",),
            "1w+1M": ("1w", "1M"),
        }
    )
    d = df["decision_t"].to_numpy()
    end = df["end"].fill_null(np.iinfo(np.int64).max).to_numpy()
    ok = df["ok"].to_numpy()
    row = df["row"].to_numpy().astype(np.int64)
    tf = df["tf"].to_numpy()
    lines = []
    all_ok = True
    for y in meta["years"]:
        cut, yend, nxt = _year_us(y - 1, 7), _year_us(y), _year_us(y + 1)
        for g, tfs in groups.items():
            kk = f"{y}_{g}"
            tr, va, te = (saved[f"{kk}_{p}"] for p in ("train", "valid", "test"))
            ing = np.isin(tf, tfs)
            want_tr = row[ok & ing & (d < cut) & (end < cut)]
            want_va = row[ok & ing & (d >= cut) & (d < yend) & (end < yend)]
            want_te = row[ing & (d >= yend) & (d < nxt)]
            if meta.get("max_train") is None:
                same_tr = np.array_equal(np.sort(tr), np.sort(want_tr))
            else:  # a smoke run's evenly spaced subset
                same_tr = bool(np.isin(tr, want_tr).all())
            pos = {int(r): i for i, r in enumerate(row)}
            it = np.array([pos[int(r)] for r in tr], dtype=np.int64)
            iv = np.array([pos[int(r)] for r in va], dtype=np.int64)
            ie = np.array([pos[int(r)] for r in te], dtype=np.int64)
            bounds = (
                (len(it) == 0 or (d[it].max() < cut and end[it].max() < cut))
                and (len(iv) == 0 or (d[iv].min() >= cut and d[iv].max() < yend))
                and (len(iv) == 0 or end[iv].max() < yend)
                and (len(ie) == 0 or (d[ie].min() >= yend and d[ie].max() < nxt))
            )
            disjoint = not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
            fold_ok = (
                same_tr
                and np.array_equal(np.sort(va), np.sort(want_va))
                and np.array_equal(np.sort(te), np.sort(want_te))
                and bool(bounds)
                and disjoint
                and bool(ok[it].all() if len(it) else True)
            )
            all_ok &= fold_ok
            lines.append(f"{kk} {len(tr)}/{len(va)}/{len(te)}{'' if fold_ok else ' FAIL'}")
    check(
        f"{run}: fold row sets are purged as specified (train/valid/test per fold)",
        all_ok,
        "; ".join(lines),
    )
    for which in sorted(p.stem for p in folder.glob("*.parquet")):
        sc = pl.read_parquet(folder / f"{which}.parquet")
        want = fam.filter(
            (pl.col("decision_t") >= _year_us(min(meta["years"])))
            & (pl.col("decision_t") < _year_us(max(meta["years"]) + 1))
        )
        if not pooled:
            pass
        ok_cov = (
            sc["cid"].n_unique() == sc.height
            and set(sc["cid"].to_list()) == set(want["cid"].to_list())
            and bool(sc["score"].is_between(0, 1).all())
        )
        check(
            f"{run}/{which}: a percentile in [0, 1] for every {family} candidate decided in"
            f" {min(meta['years'])}-{max(meta['years'])}",
            ok_cov,
            f"{sc.height} scored",
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="*", default=["base", "gbm"])
    p.add_argument("--skip-market", action="store_true")
    p.add_argument("--only-folds", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    rng = np.random.default_rng(SEED)
    if not a.only_folds:
        truncation(rng, a.workers)
        if not a.skip_market:
            cuts = [
                _year_us(2021, 3) + 1234 * FIVE,
                _year_us(2023, 9) + 777 * FIVE,
            ]
            market_causality(cuts, a.workers)
        invariants(rng)
        finite_and_precision(rng)
    for run in a.runs:
        if (bi.M4 / "scores" / run / "folds.npz").exists():
            fold_checks(run)
        else:
            check(f"{run}: scores exist", False, "no folds.npz")
    print(f"{'ALL PASS' if not failures else f'{len(failures)} FAILED: {failures}'}")


if __name__ == "__main__":
    main()
