# ruff: noqa: E501  (markdown text and table rows read better unsplit)
"""M5: the one-time reveal of the frozen rule on the sealed confirm period (D 2024-01-01 .. 2026-10-02).

    uv run --group research python m5_confirm.py

Reads the frozen `results/spec.json` (never changes it), `cache/factors.parquet`,
`cache/labels.parquet` (discovery and gap) and `cache/sealed/labels.parquet` (confirm; allowed now
that the spec is pushed), `cache/exclusions.parquet`, `cache/panel.parquet` (for `liq`).

Scoring follows the spec's training rule through `model.score_period` (its assertions kept):
2024 trains on the discovery entries; 2025 and 2026 retrain on every entry whose window ended by
1 January of that year (for this transparent rule: the percentile reference is recomputed).
Entries with D after 2026-10-02 (split `unused`) are dropped before anything is scored.

The pick rule, book, same-day random books, month bootstrap and the within-day factor view are
M4's and M3's own methods (`m4_model.random_books`-equivalent, `m4_model.month_boot`, M3's
within-day decile excess). Before the confirm period is touched, the same code path reproduces
the discovery 2023 walk-forward of M4 (268 book trades, 35.4% hit10) as a pipeline check.

Outputs: `results/m5_confirm.csv` (tidy), `results/m5_confirm.md` (generated from the csv),
`results/m5_trades.csv` (the continuous book's taken trades).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import m4_model
import numpy as np
import polars as pl
from common import (
    CACHE,
    CONFIRM_END_US,
    F64,
    LABELS,
    PANEL,
    RESULTS,
    SEALED_LABELS,
    SEALED_START_US,
)
from m4_model import N_RANDOM, SEED, STRESS, month_boot
from model import (
    STOP_VARIANTS,
    Spec,
    assemble,
    book,
    compound,
    load_spec,
    picks,
    score_period,
    stressed,
    taken_trades,
    train_mask,
    year_start_us,
)

SPEC_JSON = RESULTS / "spec.json"
OUT_CSV = RESULTS / "m5_confirm.csv"
OUT_MD = RESULTS / "m5_confirm.md"
OUT_TRADES = RESULTS / "m5_trades.csv"
EXCLUSIONS_PQ = CACHE / "exclusions.parquet"
FACTORS_PQ = CACHE / "factors.parquet"
M3_CSV = RESULTS / "m3_single.csv"

YEARS = (2024, 2025, 2026)
SCOPES = ("2024", "2025", "2026", "overall")
HITS = ("hit10", "hit20", "hit_ns")
RETS = ("ret_10", "ret_20", "ret_ns")
STOP_LABEL = {"10": "10%", "20": "20%", "ns": "none"}
TAGS = (("base", 0.0), ("stress", STRESS))
FIXED_STAKE = 1_000.0
MIN_DAY_ENTRIES = 30  # M3's within-day view: days with fewer valid entries are out
DAY_US = 86_400 * 1_000_000
FACTORS = ("c_log_liq", "c_dd90")


# --- results store ---------------------------------------------------------------------------


@dataclass
class Rows:
    rows: list[tuple[str, str, str, str, str, float, float | None]]

    def add(
        self,
        table: str,
        scope: str,
        metric: str,
        value: float,
        *,
        stop: str = "",
        stress: str = "",
        n: float | None = None,
    ) -> None:
        self.rows.append((table, scope, stop, stress, metric, float(value), n))

    def frame(self) -> pl.DataFrame:
        cols = ("table", "scope", "stop", "stress", "metric", "value", "n")
        data = [list(c) for c in zip(*self.rows, strict=True)]
        schema = {c: pl.String for c in cols[:5]} | {"value": pl.Float64, "n": pl.Float64}
        return pl.DataFrame(dict(zip(cols, data, strict=True)), schema=schema)


def fmean(x: F64) -> float:
    return float(np.mean(x)) if len(x) else math.nan


def col_f(df: pl.DataFrame, c: str) -> F64:
    return np.asarray(df[c].cast(pl.Float64).to_numpy(), dtype=np.float64)


# --- data ------------------------------------------------------------------------------------


def build_frame() -> pl.DataFrame:
    """Discovery, gap and confirm entries with factors and within-day ranks (model.assemble),
    minus exclusions and entries without an outcome. `unused` entries (D after the confirm end)
    are dropped first."""
    labels = pl.concat([pl.read_parquet(LABELS), pl.read_parquet(SEALED_LABELS)])
    n_unused = labels.filter(pl.col("split") == "unused").height
    labels = labels.filter(pl.col("split") != "unused")
    assert set(labels["split"].unique().to_list()) == {"discovery", "gap", "confirm"}
    assert int(labels["d_us"].max()) <= CONFIRM_END_US  # type: ignore[arg-type]
    excl = pl.read_parquet(EXCLUSIONS_PQ)
    fac = pl.read_parquet(FACTORS_PQ)
    # excluded entries: those of the exclusions table that are labelled entries of the used splits
    in_range = excl.filter(pl.col("d_us") <= CONFIRM_END_US)
    n_excl_labelled = labels.join(excl.select("symbol", "d_us"), on=["symbol", "d_us"], how="semi")
    assert n_excl_labelled.height == in_range.height, (n_excl_labelled.height, in_range.height)
    by_split = dict(n_excl_labelled.group_by("split").len().iter_rows())
    print(f"unused entries dropped: {n_unused}; excluded entries by split: {by_split}")
    df = assemble(fac, labels, excl)
    n_before = labels.height - n_excl_labelled.height
    n_no_outcome = n_before - df.height
    print(
        f"entries {labels.height}, excluded {n_excl_labelled.height}, no outcome {n_no_outcome}, frame {df.height}"
    )
    for c in (*HITS, *RETS):
        df = df.with_columns(pl.col(c).cast(pl.Float64))
    month = (
        pl.from_epoch("d_us", time_unit="us").dt.year() * 12
        + pl.from_epoch("d_us", time_unit="us").dt.month()
        - 1
    )
    df = df.with_columns(month=month)
    panel = pl.read_parquet(PANEL).select("symbol", "d_us", "liq")
    df = df.join(panel, on=["symbol", "d_us"], how="left")
    assert df["liq"].null_count() == 0
    conf = df.filter(pl.col("split") == "confirm")
    assert int(conf["d_us"].min()) >= SEALED_START_US  # type: ignore[arg-type]
    assert int(conf["d_us"].max()) <= CONFIRM_END_US  # type: ignore[arg-type]
    return df


def score_years(
    spec: Spec, df: pl.DataFrame, years: tuple[int, ...], *, discovery_only: bool
) -> pl.DataFrame:
    """Score each test year per the spec's training rule; the scored frame, with `score`, `pct`."""
    d = df["d_us"].to_numpy().astype(np.int64)
    parts = []
    for y in years:
        start = year_start_us(y)
        if y == 2024:
            train = df.filter(pl.col("split") == "discovery")  # the spec's 2024 training set
        else:
            train = df.filter(pl.Series(train_mask(d, start)))
        assert bool(train_mask(train["d_us"].to_numpy().astype(np.int64), start).all())
        test = df.filter((pl.col("year") == y) & (pl.col("d_us") >= start))
        if discovery_only:
            assert set(test["split"].unique().to_list()) == {"discovery"}
        else:
            test = test.filter(pl.col("split") == "confirm")
            assert set(test["split"].unique().to_list()) == {"confirm"}
            assert int(test["d_us"].max()) <= CONFIRM_END_US  # type: ignore[arg-type]
        assert int(train["d_us"].max()) + 30 * DAY_US <= start  # type: ignore[arg-type]
        sc = score_period(spec.method, train, test, spec.rule.gate, start)
        assert sc.frame.height == test.height
        print(
            f"  {y}: train {train.height}, test {test.height}, reference {len(sc.reference)}",
            flush=True,
        )
        parts.append(sc.frame)
    return pl.concat(parts)


# --- random books (M4's method; adds the book's max drawdown) --------------------------------


def max_dd(curve: F64) -> float:
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min())


def random_books(t: pl.DataFrame, n_per_day: dict[int, int], seed: int) -> dict[str, F64]:
    """`m4_model.random_books` (same draws for the same seed) with the max drawdown added."""
    yd = m4_model.year_data(t)
    rng = np.random.default_rng(seed)
    days = np.array(sorted(n_per_day), dtype=np.int64)
    m = np.isin(yd.day, days)
    idx = np.flatnonzero(m)
    day = yd.day[idx]
    need = np.array([n_per_day[int(d)] for d in day], dtype=np.int64)
    starts = np.searchsorted(day, day, side="left")
    out: dict[str, list[float]] = {}

    def put(k: str, v: float) -> None:
        out.setdefault(k, []).append(v)

    for _ in range(N_RANDOM):
        key = rng.random(len(idx))
        order = np.lexsort((key, day))
        pos = np.arange(len(idx)) - starts[order]
        chosen = idx[order[pos < need[order]]]
        put("pick_hit10", float(yd.hit10[chosen].mean()))
        for v in STOP_VARIANTS:
            ex = yd.exit_t[v][chosen]
            r = yd.ret[v][chosen]
            ok = ~np.isnan(r)
            c2, ex2, r2 = chosen[ok], ex[ok], r[ok]
            take = taken_trades(yd.d_us[c2], yd.sym[c2].tolist(), ex2)
            d_t, ex_t, r_t = yd.d_us[c2][take], ex2[take], r2[take]
            put(f"trades_{v}", float(take.sum()))
            put(f"hit10_{v}", float(yd.hit10[c2][take].mean()) if take.any() else math.nan)
            for tag, cost in (("", 0.0), ("_stress", STRESS)):
                rr = stressed(r_t, cost)
                put(f"ret_{v}{tag}", float(rr.mean()) if len(rr) else math.nan)
                _, _, curve = compound(d_t, ex_t, rr)
                put(f"book_{v}{tag}", float(curve[-1] / curve[0] - 1.0))
                put(f"dd_{v}{tag}", max_dd(curve))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def pct_of(x: F64, q: float) -> float:
    return float(np.nanpercentile(x, q))


# --- fixed-stake book ------------------------------------------------------------------------


def fixed_stake(picked: pl.DataFrame, stop: str, extra_cost: float) -> tuple[float, float, int]:
    """$1,000 per slot, at most 10 open, no compounding; the same picks and slot rules as the
    protocol book. Returns (total profit, worst peak-to-trough in dollars (<= 0), trades). The
    dollar curve moves when a trade closes (realised), starting at 0."""
    p = picked.filter(pl.col(f"ret_{stop}").is_not_null())
    d_us = p["d_us"].to_numpy().astype(np.int64)
    sym = p["symbol"].to_list()
    ex = p[f"exit_t_{stop}"].to_numpy().astype(np.int64)
    r = stressed(col_f(p, f"ret_{stop}"), extra_cost)
    take = taken_trades(d_us, sym, ex)
    ex_t, pnl = ex[take], FIXED_STAKE * r[take]
    order = np.argsort(ex_t, kind="stable")
    curve = np.concatenate([[0.0], np.cumsum(pnl[order])])
    worst = float((curve - np.maximum.accumulate(curve)).min())
    return float(pnl.sum()), worst, int(take.sum())


# --- within-day factor view (M3's method) ----------------------------------------------------


def factor_view(t: pl.DataFrame, f: str) -> dict[str, float]:
    """Bottom and top within-day decile excess of hit10 (and ret_10): the decile's mean of
    (outcome minus the day's mean), deciles of the within-day percentile (rank - 0.5) / n among
    the day's valid entries; days with fewer than 30 valid entries are out (M3)."""
    v = col_f(t, f)
    v = np.where(np.isnan(v), np.nan, v)
    valid = ~np.isnan(v)
    tt = t.select("d_us").with_columns(v=pl.Series(v).fill_nan(None))
    r = (
        tt.select(
            (
                (pl.col("v").rank("average").over("d_us") - 0.5) / pl.col("v").count().over("d_us")
            ).alias("p")
        )["p"]
        .to_numpy()
        .astype(np.float64)
    )
    days, day = np.unique(t["d_us"].to_numpy(), return_inverse=True)
    n_valid_day = np.bincount(day[valid], minlength=len(days))
    elig = valid & (n_valid_day[day] >= MIN_DAY_ENTRIES)
    b = np.where(np.isnan(r), -1, np.minimum(np.floor(np.nan_to_num(r) * 10), 9)).astype(np.int64)
    b = np.where(elig, b, -1)
    out: dict[str, float] = {"n_eligible": float(elig.sum())}
    for name, y in (("hit10", col_f(t, "hit10")), ("ret_10", col_f(t, "ret_10"))):
        ok = elig & ~np.isnan(y)
        s = np.bincount(day[ok], weights=y[ok], minlength=len(days))
        c = np.bincount(day[ok], minlength=len(days))
        with np.errstate(invalid="ignore", divide="ignore"):
            dm = np.where(c > 0, s / c, np.nan)
        e = y - dm[day]
        for lab, bk in (("bottom", 0), ("top", 9)):
            sel = ok & (b == bk)
            out[f"{lab}_excess_{name}"] = fmean(e[sel])
            if name == "hit10":
                out[f"{lab}_n"] = float(sel.sum())
                out[f"{lab}_rate_hit10"] = fmean(y[sel])
    return out


# --- one scope -------------------------------------------------------------------------------


def n_per_day_of(pk: pl.DataFrame) -> dict[int, int]:
    return {int(d // DAY_US): int(n) for d, n in pk.group_by("d_us").len().iter_rows()}


def scope_stats(
    rows: Rows, scope: str, test: pl.DataFrame, pk: pl.DataFrame, seed: int
) -> dict[str, Any]:
    """Every number of one scope (a year or the whole period) into `rows`; returns the values the
    verdict and the md need."""
    assert pk.group_by("d_us").len()["len"].max() <= 10  # type: ignore[operator]
    assert pk["d_us"].min() >= test["d_us"].min()  # type: ignore[operator]
    n_test = float(test.height)
    # base rates
    rows.add("base", scope, "entries", n_test)
    rows.add("base", scope, "days", float(test["d_us"].n_unique()))
    for c in (*HITS, *RETS):
        rows.add("base", scope, c, fmean(col_f(test, c)), n=n_test)
    # picks (every pick the rule makes, before the slot rules)
    rows.add("picks", scope, "n_picks", float(pk.height))
    rows.add("picks", scope, "n_days", float(pk["d_us"].n_unique()))
    rows.add("picks", scope, "n_coins", float(pk["symbol"].n_unique()))
    for c in (*HITS, *RETS):
        rows.add("picks", scope, c, fmean(col_f(pk, c)), n=float(pk.height))
    day_mean = test.group_by("d_us").agg(
        pl.col("hit10").mean().alias("dm_hit10"), pl.col("ret_10").mean().alias("dm_ret_10")
    )
    j = pk.join(day_mean, on="d_us")
    rows.add(
        "picks", scope, "excess_hit10_over_day", fmean(col_f(j, "hit10") - col_f(j, "dm_hit10"))
    )
    rows.add(
        "picks", scope, "excess_ret_10_over_day", fmean(col_f(j, "ret_10") - col_f(j, "dm_ret_10"))
    )
    rows.add(
        "picks",
        scope,
        "excess_hit10_over_base",
        fmean(col_f(pk, "hit10")) - fmean(col_f(test, "hit10")),
    )
    # same-day random draws
    rnd = random_books(test, n_per_day_of(pk), seed)
    for q, qn in ((5, "p5"), (50, "p50"), (95, "p95")):
        rows.add("random", scope, f"pick_hit10_{qn}", pct_of(rnd["pick_hit10"], q))
    rows.add("random", scope, "draws", float(len(rnd["pick_hit10"])))
    res: dict[str, Any] = {
        "pick_hit10": fmean(col_f(pk, "hit10")),
        "rand_p95": pct_of(rnd["pick_hit10"], 95),
        "rand_p50": pct_of(rnd["pick_hit10"], 50),
        "books": {},
    }
    for v in STOP_VARIANTS:
        for tag, cost in TAGS:
            b = book(pk, v, cost)
            tr = b.trades
            r = col_f(tr, "ret")
            rk = f"{v}_{tag}"
            suf = "" if tag == "base" else "_stress"
            kw: dict[str, Any] = {"stop": v, "stress": tag}
            rows.add("trades", scope, "trades", float(tr.height), **kw)
            rows.add("trades", scope, "coins", float(tr["symbol"].n_unique()), **kw)
            rows.add("trades", scope, "skipped_picks", float(b.skipped), **kw)
            rows.add("trades", scope, "mean_ret", fmean(r), n=float(len(r)), **kw)
            rows.add(
                "trades",
                scope,
                "median_ret",
                float(np.median(r)) if len(r) else math.nan,
                n=float(len(r)),
                **kw,
            )
            mon = np.asarray(
                tr.select(
                    pl.from_epoch("d_us", time_unit="us").dt.year() * 12
                    + pl.from_epoch("d_us", time_unit="us").dt.month()
                )
                .to_series()
                .to_numpy(),
                dtype=np.int64,
            )
            lo, hi = month_boot(mon, r, np.random.default_rng(SEED))
            rows.add("trades", scope, "mean_ret_ci_lo", lo, **kw)
            rows.add("trades", scope, "mean_ret_ci_hi", hi, **kw)
            hit = tr.select("symbol", "d_us").join(
                pk.select("symbol", "d_us", *HITS), on=["symbol", "d_us"]
            )
            for h in HITS:
                rows.add("trades", scope, h, fmean(col_f(hit, h)), n=float(hit.height), **kw)
            rows.add(
                "trades",
                scope,
                "median_liq",
                float(
                    np.median(
                        col_f(
                            tr.join(pk.select("symbol", "d_us", "liq"), on=["symbol", "d_us"]),
                            "liq",
                        )
                    )
                )
                if tr.height
                else math.nan,
                **kw,
            )
            # random comparison
            rows.add("random", scope, "mean_ret_p50", pct_of(rnd[f"ret_{v}{suf}"], 50), **kw)
            rows.add("random", scope, "mean_ret_p95", pct_of(rnd[f"ret_{v}{suf}"], 95), **kw)
            rows.add("random", scope, "hit10_trades_p50", pct_of(rnd[f"hit10_{v}"], 50), **kw)
            rows.add("random", scope, "hit10_trades_p95", pct_of(rnd[f"hit10_{v}"], 95), **kw)
            rows.add("random", scope, "trades_mean", fmean(rnd[f"trades_{v}"]), **kw)
            # protocol book
            rows.add("book", scope, "book_return", b.total_return, **kw)
            rows.add("book", scope, "max_dd", b.max_drawdown, **kw)
            rows.add("book", scope, "final_equity", b.final, **kw)
            for q, qn in ((5, "p5"), (50, "p50"), (95, "p95")):
                rows.add(
                    "random", scope, f"book_return_{qn}", pct_of(rnd[f"book_{v}{suf}"], q), **kw
                )
                rows.add("random", scope, f"max_dd_{qn}", pct_of(rnd[f"dd_{v}{suf}"], q), **kw)
            # fixed stake
            profit, worst, n_tr = fixed_stake(pk, v, cost)
            rows.add("fixed", scope, "profit", profit, n=float(n_tr), **kw)
            rows.add("fixed", scope, "worst_peak_to_trough", worst, **kw)
            res["books"][rk] = {
                "book_return": b.total_return,
                "rand_p50": pct_of(rnd[f"book_{v}{suf}"], 50),
                "trades": tr,
            }
    return res


# --- main ------------------------------------------------------------------------------------


def reproduce_discovery(spec: Spec, df: pl.DataFrame) -> None:
    """The same code path on the discovery 2023 walk-forward must give M4's numbers."""
    disc = df.filter(pl.col("split") == "discovery")
    sc = score_years(spec, disc, (2023,), discovery_only=True)
    pk = picks(sc, spec.rule)
    b = book(pk, "10", 0.0)
    tr = b.trades.select("symbol", "d_us").join(
        pk.select("symbol", "d_us", "hit10"), on=["symbol", "d_us"]
    )
    hit = fmean(col_f(tr, "hit10"))
    band = spec.extra["expected_band_discovery"]["per_year"]["2023"]
    print(
        f"reproduction 2023: picks {pk.height}, trades {b.trades.height}, hit10 {hit:.4f}, mean ret {fmean(col_f(b.trades, 'ret')):.6f}"
    )
    assert b.trades.height == 268
    assert round(hit * 100, 1) == 35.4
    assert pk.height == int(band["pick_n"])
    assert abs(fmean(col_f(b.trades, "ret")) - band["mean_ret_10_none"]) < 1e-9
    assert abs(b.total_return - band["book_return_10_none"]) < 1e-9
    assert abs(fmean(col_f(sc, "hit10")) - band["base_hit10"]) < 1e-9


def main() -> None:
    spec = load_spec(SPEC_JSON)
    assert spec.rule.name == "k10_q0.95"
    assert spec.extra["name"] == "rule_liq_dd_nogate"
    assert spec.stop == "10"
    assert spec.method.kind == "transparent"
    df = build_frame()
    print("reproducing the discovery 2023 walk-forward ...")
    reproduce_discovery(spec, df)
    print("scoring the confirm years ...")
    scored = score_years(spec, df, YEARS, discovery_only=False)
    assert set(scored["split"].unique().to_list()) == {"confirm"}
    n_conf = df.filter(pl.col("split") == "confirm").height
    assert scored.height == n_conf, (scored.height, n_conf)
    pk_all = picks(scored, spec.rule)
    assert pk_all.group_by("d_us").len()["len"].max() <= spec.rule.k  # type: ignore[operator]
    assert bool((pk_all["pct"] >= spec.rule.q).all())

    # M4's own random_books gives the same draws as the copy here (one year, checked)
    t24 = scored.filter(pl.col("year") == 2024)
    pk24 = picks(t24, spec.rule)
    m4_model._init({2024: m4_model.year_data(t24)})
    a = m4_model.random_books(2024, n_per_day_of(pk24), SEED + 2024)
    bb = random_books(t24, n_per_day_of(pk24), SEED + 2024)
    for k, va in a.items():
        assert np.allclose(va, bb[k], equal_nan=True), k

    rows = Rows([])
    results: dict[str, dict[str, Any]] = {}
    for y in YEARS:
        ty = scored.filter(pl.col("year") == y)
        results[str(y)] = scope_stats(rows, str(y), ty, picks(ty, spec.rule), SEED + y)
    results["overall"] = scope_stats(rows, "overall", scored, pk_all, SEED)

    # factors separately, per year and overall
    for scope in SCOPES:
        t = scored if scope == "overall" else scored.filter(pl.col("year") == int(scope))
        for f in FACTORS:
            for k, val in factor_view(t, f).items():
                rows.add("factor", scope, f"{f}:{k}", val)

    # verdict: the protocol's rule, computed
    years_beat = [
        str(y) for y in YEARS if results[str(y)]["pick_hit10"] > results[str(y)]["rand_p95"]
    ]
    ov = results["overall"]
    c1 = bool(ov["pick_hit10"] > ov["rand_p95"])
    c2 = len(years_beat) >= 2
    bk = ov["books"]["10_base"]
    c3 = bool(bk["book_return"] > 0)
    c4 = bool(bk["book_return"] > bk["rand_p50"])
    confirmed = c1 and c2 and c3 and c4
    for name, ok in (
        ("c1_overall_hit10_over_random_p95", c1),
        ("c2_two_of_three_years", c2),
        ("c3_book_positive", c3),
        ("c4_book_over_random_median", c4),
        ("confirmed", confirmed),
    ):
        rows.add("verdict", "overall", name, float(ok))
    rows.add("verdict", "overall", "years_over_random_p95", float(len(years_beat)))
    for y in YEARS:
        rows.add(
            "verdict",
            str(y),
            "hit10_over_random_p95",
            float(results[str(y)]["pick_hit10"] > results[str(y)]["rand_p95"]),
        )

    rows.frame().write_csv(OUT_CSV, float_precision=6)
    write_trades(spec, pk_all, results["overall"])
    write_md(spec)
    print(
        f"verdict: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'} (c1 {c1}, c2 {c2}, c3 {c3}, c4 {c4})"
    )


def write_trades(spec: Spec, pk: pl.DataFrame, overall: dict[str, Any]) -> None:
    """The continuous book's taken trades: one row per pick taken under at least one stop variant."""
    out = pk.select(
        "symbol", "d_us", "year", "score", "pct", "liq", "pick_rank", "entry_t", "entry_open", *HITS
    )
    out = out.with_columns(D=pl.from_epoch("d_us", time_unit="us").dt.date().cast(pl.String))
    flags: list[str] = []
    for v in STOP_VARIANTS:
        taken = (
            overall["books"][f"{v}_base"]["trades"]
            .select("symbol", "d_us")
            .with_columns(**{f"taken_{v}": pl.lit(True)})
        )
        out = out.join(taken, on=["symbol", "d_us"], how="left").with_columns(
            pl.col(f"taken_{v}").fill_null(False)
        )
        out = out.join(
            pk.select(
                "symbol", "d_us", f"exit_t_{v}", f"exit_price_{v}", f"reason_{v}", f"ret_{v}"
            ),
            on=["symbol", "d_us"],
        )
        out = out.with_columns(
            pl.Series(f"ret_{v}_stress", stressed(col_f(out, f"ret_{v}"), STRESS))
        )
        flags.append(f"taken_{v}")
    out = out.filter(pl.any_horizontal(flags)).sort("d_us", "pick_rank")
    cols = [
        "symbol",
        "D",
        "d_us",
        "year",
        "score",
        "pct",
        "liq",
        "pick_rank",
        "entry_t",
        "entry_open",
        *HITS,
    ]
    for v in STOP_VARIANTS:
        cols += [
            f"taken_{v}",
            f"exit_t_{v}",
            f"exit_price_{v}",
            f"reason_{v}",
            f"ret_{v}",
            f"ret_{v}_stress",
        ]
    out.select(cols).write_csv(OUT_TRADES, float_precision=6)
    print(f"m5_trades.csv: {out.height} rows")


# --- markdown ---------------------------------------------------------------------------------


def pc(x: float, d: int = 1) -> str:
    return "n/a" if math.isnan(x) else f"{x * 100:.{d}f}"


def money(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:,.0f}"


class Look:
    def __init__(self, df: pl.DataFrame) -> None:
        self.d: dict[tuple[str, str, str, str, str], float] = {
            (r["table"], r["scope"], r["stop"], r["stress"], r["metric"]): r["value"]
            for r in df.iter_rows(named=True)
        }

    def __call__(
        self, table: str, scope: str, metric: str, stop: str = "", stress: str = ""
    ) -> float:
        return self.d.get((table, scope, stop, stress, metric), math.nan)


def fmtd(x: float, dec: int) -> str:
    return money(x) if dec < 0 else pc(x, dec)


def verdict_word(b: bool) -> str:
    return "pass" if b else "FAIL"


def table(head: list[str], body: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] * len(head)) + "|",
        *("| " + " | ".join(r) + " |" for r in body),
    ]


def write_md(spec: Spec) -> None:
    df = pl.read_csv(
        OUT_CSV,
        schema_overrides=dict.fromkeys(("table", "scope", "stop", "stress", "metric"), pl.String),
    )
    df = df.with_columns(pl.col("stop").fill_null(""), pl.col("stress").fill_null(""))
    g = Look(df)
    band = spec.extra["expected_band_discovery"]
    L: list[str] = []
    ok = verdict_word
    conf = g("verdict", "overall", "confirmed") == 1.0
    L += [
        "# M5: the reveal on 2024-26",
        "",
        "Generated by `m5_confirm.py` from `results/m5_confirm.csv`; the run is the one-time reveal of the frozen "
        f"spec (`{spec.extra['name']}`, pick rule `{spec.rule.name}`, 10% stop primary). Nothing was tuned after "
        "the outcomes were read. Confirm period: D 2024-01-01 .. 2026-10-02. Rates, returns and drawdowns in "
        "percent unless marked $. Trades are the book's taken trades (10 slots, one position per coin); picks "
        "are every entry the rule selects (up to 10 a day) before the slot rules.",
        "",
        "## Verdict",
        "",
        f"**{'CONFIRMED' if conf else 'NOT CONFIRMED'}**, by the protocol's rule: "
        + spec.extra["confirmed_criteria"],
        "",
    ]
    ov_hit, ov_p95 = g("picks", "overall", "hit10"), g("random", "overall", "pick_hit10_p95")
    nyears = int(g("verdict", "overall", "years_over_random_p95"))
    bret, bmed = (
        g("book", "overall", "book_return", "10", "base"),
        g("random", "overall", "book_return_p50", "10", "base"),
    )
    L += table(
        ["clause", "value", "threshold", "result"],
        [
            [
                "picks' hit10 over the same-day random 95th percentile, overall",
                f"{pc(ov_hit)}%",
                f"{pc(ov_p95)}%",
                ok(g("verdict", "overall", "c1_overall_hit10_over_random_p95") == 1.0),
            ],
            [
                "... and in at least two of the three years",
                f"{nyears} of 3 ({', '.join(f'{y}: {pc(g("picks", y, "hit10"))}% vs {pc(g("random", y, "pick_hit10_p95"))}%' for y in ('2024', '2025', '2026'))})",
                ">= 2",
                ok(g("verdict", "overall", "c2_two_of_three_years") == 1.0),
            ],
            [
                "10%-stop book return positive (continuous, after costs)",
                f"{pc(bret)}%",
                "> 0",
                ok(g("verdict", "overall", "c3_book_positive") == 1.0),
            ],
            [
                "10%-stop book return above the random books' median",
                f"{pc(bret)}%",
                f"{pc(bmed)}%",
                ok(g("verdict", "overall", "c4_book_over_random_median") == 1.0),
            ],
            ["**all four**", "", "", f"**{'confirmed' if conf else 'not confirmed'}**"],
        ],
    )
    L += [
        "",
        "The book clauses use the protocol book with the base costs (0.1% fee and 0.1% slippage per the labels); the 0.3%-a-side stress is reported below.",
        "",
    ]

    # base and picks
    L += [
        "## Base rates and picks",
        "",
        "Base = every usable entry of the scope. Excess over the day = mean of (pick outcome minus the day's mean over all entries); picks are every pick, trades below are the book's.",
        "",
    ]
    head = [
        "scope",
        "entries",
        "base hit10",
        "base hit20",
        "base hit_ns",
        "base ret_10",
        "base ret_20",
        "base ret_ns",
    ]
    L += table(
        head,
        [
            [s, money(g("base", s, "entries")), *[pc(g("base", s, m)) for m in (*HITS, *RETS)]]
            for s in SCOPES
        ],
    )
    L += ["", "Picks:", ""]
    head = [
        "scope",
        "picks",
        "days",
        "coins",
        "hit10",
        "hit20",
        "hit_ns",
        "ret_10",
        "ret_20",
        "ret_ns",
        "hit10 excess over base (pp)",
        "hit10 excess over the day (pp)",
        "ret_10 excess over the day (pp)",
    ]
    L += table(
        head,
        [
            [
                s,
                money(g("picks", s, "n_picks")),
                money(g("picks", s, "n_days")),
                money(g("picks", s, "n_coins")),
                *[pc(g("picks", s, m)) for m in (*HITS, *RETS)],
                pc(g("picks", s, "excess_hit10_over_base")),
                pc(g("picks", s, "excess_hit10_over_day")),
                pc(g("picks", s, "excess_ret_10_over_day")),
            ]
            for s in SCOPES
        ],
    )

    # trades
    L += [
        "",
        "## The book's trades",
        "",
        "Per trade net returns (fees and slippage in the labels; *stress* adds 0.3% a side). CI = month-bootstrap 95% interval of the mean (1,000 draws).",
        "",
    ]
    body = []
    for s in SCOPES:
        for v in STOP_VARIANTS:
            for tag in ("base", "stress"):
                gv = partial(g, "trades", s, stop=v, stress=tag)
                body.append(
                    [
                        s,
                        STOP_LABEL[v],
                        tag,
                        money(gv("trades")),
                        money(gv("coins")),
                        pc(gv("hit10")),
                        pc(gv("hit20")),
                        pc(gv("hit_ns")),
                        pc(gv("mean_ret"), 2),
                        pc(gv("median_ret"), 2),
                        f"{pc(gv('mean_ret_ci_lo'), 2)} .. {pc(gv('mean_ret_ci_hi'), 2)}",
                    ]
                )
    L += table(
        [
            "scope",
            "stop",
            "costs",
            "trades",
            "coins",
            "hit10",
            "hit20",
            "hit_ns",
            "mean ret",
            "median ret",
            "mean ret 95% CI",
        ],
        body,
    )

    # random
    L += [
        "",
        "## Against same-day random (100 draws)",
        "",
        "On each day with n picks, n entries are drawn from the day's usable entries and booked the same way. *Picks hit10* uses every pick; the mean return is over the book's trades.",
        "",
    ]
    body = []
    for s in SCOPES:
        body.append(
            [
                s,
                pc(g("picks", s, "hit10")),
                pc(g("random", s, "pick_hit10_p50")),
                pc(g("random", s, "pick_hit10_p95")),
                ok(g("picks", s, "hit10") > g("random", s, "pick_hit10_p95")),
            ]
        )
    L += table(["scope", "picks hit10", "random median", "random 95th", "picks over 95th"], body)
    L += [
        "",
        "Mean net return per trade (10% / 20% / none stops, base costs) against the random draws:",
        "",
    ]
    body = []
    for s in SCOPES:
        for v in STOP_VARIANTS:
            for tag in ("base", "stress"):
                body.append(
                    [
                        s,
                        STOP_LABEL[v],
                        tag,
                        pc(g("trades", s, "mean_ret", v, tag), 2),
                        pc(g("random", s, "mean_ret_p50", v, tag), 2),
                        pc(g("random", s, "mean_ret_p95", v, tag), 2),
                        pc(g("trades", s, "hit10", v, tag)),
                        pc(g("random", s, "hit10_trades_p50", v, tag)),
                        pc(g("random", s, "hit10_trades_p95", v, tag)),
                    ]
                )
    L += table(
        [
            "scope",
            "stop",
            "costs",
            "picks mean ret",
            "random median",
            "random 95th",
            "trades hit10",
            "random median",
            "random 95th",
        ],
        body,
    )

    # book
    L += [
        "",
        "## The protocol book ($10,000, 10 slots of realised equity)",
        "",
        "Per year a fresh book; *overall* is one continuous book 2024-01 .. 2026-11 (the verdict's book). Random columns: median, and 5th / 95th percentiles of the random books.",
        "",
    ]
    body = []
    for s in SCOPES:
        for v in STOP_VARIANTS:
            for tag in ("base", "stress"):
                gb = partial(g, "book", s, stop=v, stress=tag)
                gr = partial(g, "random", s, stop=v, stress=tag)
                body.append(
                    [
                        s,
                        STOP_LABEL[v],
                        tag,
                        pc(gb("book_return")),
                        f"{pc(gr('book_return_p50'))} ({pc(gr('book_return_p5'))} .. {pc(gr('book_return_p95'))})",
                        pc(gb("max_dd")),
                        f"{pc(gr('max_dd_p50'))} ({pc(gr('max_dd_p5'))} .. {pc(gr('max_dd_p95'))})",
                    ]
                )
    L += table(
        [
            "scope",
            "stop",
            "costs",
            "book return",
            "random return median (p5 .. p95)",
            "max drawdown",
            "random max dd median (p5 .. p95)",
        ],
        body,
    )

    # fixed
    L += [
        "",
        "## Fixed-stake book ($1,000 a slot, at most 10 open, no compounding)",
        "",
        "Same picks and slot rules as the protocol book. Profit in $ is the sum of the trades' $ results; worst peak-to-trough is on the realised dollar curve.",
        "",
    ]
    body = []
    for s in SCOPES:
        for v in STOP_VARIANTS:
            for tag in ("base", "stress"):
                body.append(
                    [
                        s,
                        STOP_LABEL[v],
                        tag,
                        money(g("trades", s, "trades", v, tag)),
                        money(g("fixed", s, "profit", v, tag)),
                        money(g("fixed", s, "worst_peak_to_trough", v, tag)),
                        money(g("trades", s, "median_liq", v, tag)) if tag == "base" else "",
                    ]
                )
    L += table(
        [
            "scope",
            "stop",
            "costs",
            "trades",
            "total profit $",
            "worst peak-to-trough $",
            "median daily volume of traded coins $",
        ],
        body,
    )

    # factors
    L += [
        "",
        "## The two factors separately",
        "",
        "Within-day decile excess of hit10 (pp): the decile's mean of (hit10 minus the day's mean), deciles of the within-day percentile, days with at least 30 valid entries (M3's method). Bottom = lowest value (low liquidity; deep drawdown, i.e. the lowest `c_dd90`).",
        "",
    ]
    m3 = pl.read_csv(M3_CSV).filter(
        (pl.col("view") == "coin_rank_decile")
        & (pl.col("metric") == "hit10_excess")
        & (pl.col("bucket") == "1")
    )
    body = []
    for f in FACTORS:
        for s in SCOPES:
            body.append(
                [
                    f"`{f}`",
                    s,
                    money(g("factor", s, f"{f}:bottom_n")),
                    pc(g("factor", s, f"{f}:bottom_rate_hit10")),
                    pc(g("factor", s, f"{f}:bottom_excess_hit10")),
                    pc(g("factor", s, f"{f}:top_excess_hit10")),
                    pc(g("factor", s, f"{f}:bottom_excess_ret_10"), 2),
                ]
            )
    L += table(
        [
            "factor",
            "scope",
            "bottom-decile entries",
            "bottom hit10",
            "bottom excess hit10 (pp)",
            "top-decile excess hit10 (pp)",
            "bottom excess ret_10 (pp)",
        ],
        body,
    )
    disc_lines = []
    for f in FACTORS:
        vals = {
            r["scope"]: r["value"] for r in m3.filter(pl.col("factor") == f).iter_rows(named=True)
        }
        disc_lines.append(
            f"`{f}` bottom-decile excess in discovery (M3, pp): "
            + ", ".join(f"{k} {pc(float(v))}" for k, v in sorted(vals.items()))
        )
    L += ["", *[f"- {x}" for x in disc_lines]]

    # band
    L += [
        "",
        "## Against the discovery expected band (spec)",
        "",
        "Discovery walk-forward 2021-23 range (min .. max over its three years) against each confirm year. Per-year book rows are fresh $10,000 books.",
        "",
    ]
    rng = band["range_over_years"]
    # (label, spec key, table, metric, stop, stress, decimals; -1 = a count)
    rows_b = [
        ("picks hit10", "pick_hit10", "picks", "hit10", "", "", 1),
        (
            "random 95th of pick hit10",
            "random_pick_hit10_p95",
            "random",
            "pick_hit10_p95",
            "",
            "",
            1,
        ),
        ("trades (10%, base)", "trades_10_none", "trades", "trades", "10", "base", -1),
        (
            "mean net ret per trade (10%, base)",
            "mean_ret_10_none",
            "trades",
            "mean_ret",
            "10",
            "base",
            2,
        ),
        (
            "mean net ret per trade (10%, stress)",
            "mean_ret_10_stress",
            "trades",
            "mean_ret",
            "10",
            "stress",
            2,
        ),
        ("book return (10%, base)", "book_return_10_none", "book", "book_return", "10", "base", 1),
        (
            "book return (10%, stress)",
            "book_return_10_stress",
            "book",
            "book_return",
            "10",
            "stress",
            1,
        ),
        (
            "random book median (10%, base)",
            "random_book_return_p50_10_none",
            "random",
            "book_return_p50",
            "10",
            "base",
            1,
        ),
    ]
    body = []
    for label, key, tbl, metric, stp, strs, dec in rows_b:
        lo, hi = rng[key]
        cells = [label, f"{fmtd(lo, dec)} .. {fmtd(hi, dec)}"]
        for s in ("2024", "2025", "2026"):
            x = g(tbl, s, metric, stp, strs)
            inside = lo <= x <= hi
            cells.append(f"{fmtd(x, dec)}{'' if inside else ' *'}")
        body.append(cells)
    L += table(["metric", "discovery range 2021-23", "2024", "2025", "2026"], body)
    L += ["", "`*` = outside the discovery range."]
    p = band["pooled_book_trades_2021_23"]
    cont = band["continuous_book_2021_23"]
    L += [
        "",
        f"Discovery pooled book trades 2021-23: mean net return per trade (10%, base) {pc(p['mean_ret_10_none'], 2)}% (95% CI {pc(p['mean_ret_ci_10_none'][0], 2)} .. {pc(p['mean_ret_ci_10_none'][1], 2)}); "
        f"confirm {pc(g('trades', 'overall', 'mean_ret', '10', 'base'), 2)}% (CI {pc(g('trades', 'overall', 'mean_ret_ci_lo', '10', 'base'), 2)} .. {pc(g('trades', 'overall', 'mean_ret_ci_hi', '10', 'base'), 2)}).",
        f"Discovery continuous 2021-23 book (10%, base): {pc(cont['book_10_none'])}% against random median {pc(cont['random_book_p50_10_none'])}%; pick hit10 {pc(cont['pick_hit10'])}% against random 95th {pc(cont['random_pick_hit10_p95'])}%.",
    ]

    # also observed
    L += ["", "## Also observed (not part of the verdict)", ""]
    obs: list[str] = []
    for s in SCOPES:
        obs.append(
            f"{s}: picks hit10 {pc(g('picks', s, 'hit10'))}% against a base of {pc(g('base', s, 'hit10'))}% (random median {pc(g('random', s, 'pick_hit10_p50'))}%, 95th {pc(g('random', s, 'pick_hit10_p95'))}%); 10%-stop trades {money(g('trades', s, 'trades', '10', 'base'))}, mean {pc(g('trades', s, 'mean_ret', '10', 'base'), 2)}% base, {pc(g('trades', s, 'mean_ret', '10', 'stress'), 2)}% stress."
        )
    for v in ("20", "ns"):
        obs.append(
            f"{STOP_LABEL[v]}-stop continuous book (base): {pc(g('book', 'overall', 'book_return', v, 'base'))}% against random median {pc(g('random', 'overall', 'book_return_p50', v, 'base'))}%; stress {pc(g('book', 'overall', 'book_return', v, 'stress'))}%."
        )
    obs.append(
        f"10%-stop continuous book under the 0.3% stress: {pc(g('book', 'overall', 'book_return', '10', 'stress'))}% (max drawdown {pc(g('book', 'overall', 'max_dd', '10', 'stress'))}%)."
    )
    obs.append(
        f"Fixed-stake ($1,000 a slot) 10%-stop total profit overall: {money(g('fixed', 'overall', 'profit', '10', 'base'))} base, {money(g('fixed', 'overall', 'profit', '10', 'stress'))} stress (dollars); worst peak-to-trough {money(g('fixed', 'overall', 'worst_peak_to_trough', '10', 'base'))}."
    )
    obs.append(
        f"Median daily volume of the traded coins (10% stop, overall): ${money(g('trades', 'overall', 'median_liq', '10', 'base'))}."
    )
    L += [f"- {o}" for o in obs]
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
