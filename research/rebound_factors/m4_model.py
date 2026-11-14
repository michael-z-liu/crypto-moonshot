"""M4: combined model, pick rule and book on the discovery walk-forward; freezes the spec.

    uv run --group research python m4_model.py [--refit] [--md-only]

Reads `cache/labels.parquet` (rows with `split == "discovery"` only), `cache/factors.parquet`
(rows with D up to the discovery end only) and `cache/exclusions.parquet`. It never opens
`cache/sealed/`; every frame is asserted to end before 2024-01-01.

Walk-forward: for Y in 2021, 2022, 2023, train on the entries whose window ended by Y-01-01
(`model.train_mask`), score year Y. Scores are cached in `cache/m4_scores.parquet` (discovery
rows only); `--refit` recomputes them.

Stages (every variant and rule tried is a row of `results/m4_ledger.csv`; every number is in
`results/m4_results.csv`; `results/m4_model.md` is generated from those two files):
1. Settings: the three fixed `model.SETTINGS` on the full feature set, 3 seeds each. The
   settings with the highest mean (over years) seed-mean AUC are used for every other GBM
   variant, unless a simpler one (shallow < base < deep) is within the seed spread (the range of
   the per-seed mean AUC) of it, then the simplest such.
2. Variants (GBM with the chosen settings, 3 seeds, ensemble = mean probability): all factors +
   ranks; market factors only; coin factors + ranks; the lean set (`lean_set`). Transparent rules
   (no model): BTC 30-day return gate at the training days' 60th percentile, then within-day sort
   on low liquidity, or on low liquidity and deep 90-day drawdown. Benchmarks: every entry and
   same-day random picks.
3. Score views: AUC; top 1/5/10% by score percentile against the training scores; top-K within
   day against the day; "when" against "which" for the top 5%.
4. Pick rules on the ensemble scores: K in {3, 5, 10} x q in {0.90, 0.95, 0.99} x gate off/on
   (BTC 30-day return at or above the training days' 60th percentile). Per year a fresh $10,000
   book for each stop variant, with and without 0.3% extra cost a side; the same-day random book
   (100 draws); a month bootstrap of the mean net return per trade.
5. Choice (rule fixed here, before the ranking was looked at): see `choose`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from common import (
    CACHE,
    DISCOVERY_END_US,
    F64,
    I64,
    LABELS,
    RESULTS,
    SEALED_START_US,
)
from model import (
    SETTINGS,
    STOP_VARIANTS,
    Gate,
    Method,
    PickRule,
    Settings,
    Spec,
    TransparentRule,
    all_features,
    assemble,
    book,
    compound,
    factor_names,
    picks,
    score,
    score_period,
    spec_to_json,
    stressed,
    taken_trades,
    train_mask,
    with_ranks,
    year_start_us,
)
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

FACTORS_PQ = CACHE / "factors.parquet"
EXCLUSIONS_PQ = CACHE / "exclusions.parquet"
SCORES = CACHE / "m4_scores.parquet"
LEDGER = RESULTS / "m4_ledger.csv"
RESULTS_CSV = RESULTS / "m4_results.csv"
MD = RESULTS / "m4_model.md"
SPEC = RESULTS / "spec.json"
M3_CSV = RESULTS / "m3_single.csv"
M3_RED = RESULTS / "m3_redundancy.csv"
FACTORS_MD = RESULTS / "factors.md"
RUN1_LEDGER = CACHE / "m4_run1" / "m4_ledger.csv"  # run 1's ledger, kept and merged
RUN = "2"
RUN1 = "1 (in-sample score reference; superseded)"

TEST_YEARS = (2021, 2022, 2023)
SEEDS = [1, 2, 3]
CONFIRM_SEEDS = [1, 2, 3, 4, 5]
TOPS = (0.01, 0.05, 0.10)
DAY_KS = (3, 5, 10)
RULE_KS = (3, 5, 10)
RULE_QS = (0.90, 0.95, 0.99)
GATE = Gate("m_btc_ret30", 0.6)
STRESS = 0.003
N_RANDOM = 100
N_BOOT = 1000
SEED = 20261114
MIN_TRADES = 100
TIE = 0.001  # 0.1 pp of mean net return per trade
LEAN_MAX = 12
LEAN_RHO = 0.7
HITS = ("hit10", "hit20", "hit_ns")
RETS = ("ret_10", "ret_20", "ret_ns")
OUTS = (*HITS, *RETS)
SIMPLICITY = {
    "rule_liq": 0,
    "rule_liq_dd": 1,
    "rule_liq_dd_nogate": 1,
    "gbm_lean": 2,
    "gbm_market": 3,
    "gbm_ranks": 4,
    "gbm_coin": 5,
}
SETTINGS_ORDER = ("shallow", "base", "deep")
KEEP = [
    "symbol",
    "d_us",
    "year",
    "month",
    "split",
    *HITS,
    *RETS,
    *(f"exit_t_{v}" for v in STOP_VARIANTS),
]


# --- data ------------------------------------------------------------------------------------


def load() -> pl.DataFrame:
    """Discovery entries (minus exclusions, with an outcome) with factors and within-day ranks."""
    labels = pl.read_parquet(LABELS).filter(
        (pl.col("split") == "discovery") & (pl.col("d_us") <= DISCOVERY_END_US)
    )
    fac = pl.read_parquet(FACTORS_PQ).filter(pl.col("d_us") <= DISCOVERY_END_US)
    excl = pl.read_parquet(EXCLUSIONS_PQ).filter(pl.col("d_us") <= DISCOVERY_END_US)
    df = assemble(fac, labels, excl)
    df = df.with_columns(
        month=pl.from_epoch("d_us", time_unit="us").dt.year() * 12
        + pl.from_epoch("d_us", time_unit="us").dt.month()
        - 1
    )
    assert int(df["d_us"].max()) < SEALED_START_US  # type: ignore[arg-type]
    assert int(df["d_us"].max()) <= DISCOVERY_END_US  # type: ignore[arg-type]
    assert set(df["split"].unique().to_list()) == {"discovery"}
    for c in (*HITS, *RETS):
        df = df.with_columns(pl.col(c).cast(pl.Float64))
    print(f"discovery entries used: {df.height}")
    return df


# --- lean set and variants -------------------------------------------------------------------


def lean_set() -> tuple[list[str], list[dict[str, Any]]]:
    """At most 12 factors, chosen from M3 by a fixed rule.

    Candidates: market factors whose best-end quintile hit10 lift has a month-bootstrap 95%
    interval above zero, and coin factors whose best-end within-day decile excess hit10 has an
    interval above zero (`results/m3_single.csv`, all discovery years). In order of that lift,
    a candidate is kept unless its |Spearman| with a kept factor exceeds 0.7
    (`results/m3_redundancy.csv`): one factor per redundancy cluster. Caveat: M3 used all of
    2020-23, so the lean set has seen the walk-forward test years.
    """
    m3 = pl.read_csv(M3_CSV)
    red = pl.read_csv(M3_RED)
    rho = {r["factor"]: r for r in red.iter_rows(named=True)}
    cands: list[dict[str, Any]] = []
    for view in ("mkt_summary", "coin_rank_summary"):
        t = m3.filter((pl.col("view") == view) & (pl.col("scope") == "all"))
        for f in t["factor"].unique().sort().to_list():
            tf = t.filter(pl.col("factor") == f)

            def val(bucket: str, metric: str, tf: pl.DataFrame = tf) -> float:
                v = tf.filter((pl.col("bucket") == bucket) & (pl.col("metric") == metric))
                return float(v["value"][0]) if v.height else float("nan")

            top, bot = val("top", "lift_hit10"), val("bottom", "lift_hit10")
            end = "top" if top >= bot else "bottom"
            lo = val(end, "ci_lo_lift_hit10")
            if lo > 0:
                cands.append({"factor": f, "end": end, "lift": max(top, bot), "ci_lo": lo})
    cands.sort(key=lambda c: -float(c["lift"]))
    kept: list[str] = []
    for c in cands:
        f = str(c["factor"])
        clash = [k for k in kept if abs(float(rho[f][k])) > LEAN_RHO]
        c["kept"] = not clash and len(kept) < LEAN_MAX
        c["clash"] = ",".join(clash)
        if c["kept"]:
            kept.append(f)
    return kept, cands


@dataclass(frozen=True)
class Variant:
    name: str
    method: Method


def gbm_variants(df: pl.DataFrame, settings: Settings, lean: list[str]) -> list[Variant]:
    market, coin = factor_names([c for c in df.columns if not c.startswith("r_")])
    sets = {
        "gbm_all": all_features(market, coin),
        "gbm_market": market,
        "gbm_coin": with_ranks(coin),
        "gbm_lean": with_ranks(lean),
        # added after run 1 (see the md): within-day ranks only, no raw level of any factor
        "gbm_ranks": [f"r_{f}" for f in coin],
    }
    return [Variant(n, Method("gbm", f, settings, SEEDS)) for n, f in sets.items()]


TRANSPARENT = {
    "rule_liq": TransparentRule(GATE, (("c_log_liq", -1),)),
    "rule_liq_dd": TransparentRule(GATE, (("c_log_liq", -1), ("c_dd90", -1))),
    # added after run 1 (see the md): the same sort without the gate
    "rule_liq_dd_nogate": TransparentRule(None, (("c_log_liq", -1), ("c_dd90", -1))),
}


def transparent_variants() -> list[Variant]:
    out = []
    for n, r in TRANSPARENT.items():
        g = [] if r.gate is None else [r.gate.factor]
        feats = [*g, *(f for f, _ in r.coin), *(f"r_{f}" for f, _ in r.coin)]
        out.append(Variant(n, Method("transparent", feats, transparent=r)))
    return out


# --- walk-forward scoring --------------------------------------------------------------------


def walk_forward(df: pl.DataFrame, v: Variant) -> pl.DataFrame:
    """Per test year: seed scores s<seed>/p<seed> (GBM), ensemble `score`/`pct`, `gate_ok`."""
    d = df["d_us"].to_numpy().astype(np.int64)
    parts = []
    for y in TEST_YEARS:
        start = year_start_us(y)
        train = df.filter(pl.Series(train_mask(d, start)))
        test = df.filter(pl.col("year") == y)
        sc = score_period(v.method, train, test, GATE, start)
        out = sc.frame.select("symbol", "d_us", "score", "pct", "gate_ok")
        for j, m in enumerate(sc.models):  # single seeds against their out-of-fold reference
            ref = np.sort(sc.seed_refs[:, j])
            s = score(m, test)
            p = np.searchsorted(ref, s, side="right") / len(ref)
            out = out.with_columns(pl.Series(f"s{m.seed}", s), pl.Series(f"p{m.seed}", p))
        out = out.with_columns(
            variant=pl.lit(v.name),
            year=pl.lit(y),
            gate_thr=pl.lit(sc.gate_threshold),
            n_train=pl.lit(train.height),
            train_max_d=pl.lit(int(train["d_us"].max())),  # type: ignore[arg-type]
        )
        parts.append(out)
        print(f"  {v.name} {y}: train {train.height}, test {test.height}", flush=True)
    return pl.concat(parts, how="diagonal_relaxed")


# --- results store ---------------------------------------------------------------------------


class Rows:
    COLS = ("table", "variant", "rule", "stop", "stress", "year", "seed", "metric", "value", "n")

    def __init__(self) -> None:
        self.rows: list[tuple[Any, ...]] = []

    def add(
        self,
        table: str,
        variant: str,
        metric: str,
        value: float,
        *,
        rule: str = "",
        stop: str = "",
        stress: str = "",
        year: str = "pooled",
        seed: str = "",
        n: float | None = None,
    ) -> None:
        if isinstance(value, float) and math.isnan(value):
            return
        self.rows.append((table, variant, rule, stop, stress, year, seed, metric, float(value), n))

    def frame(self) -> pl.DataFrame:
        cols: list[Any] = (
            [list(c) for c in zip(*self.rows, strict=True)] if self.rows else [[]] * len(self.COLS)
        )
        schema = {c: pl.String for c in self.COLS[:8]} | {"value": pl.Float64, "n": pl.Float64}
        return pl.DataFrame(dict(zip(self.COLS, cols, strict=True)), schema=schema)


# --- score views -----------------------------------------------------------------------------


def auc(y: F64, s: F64) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def seed_cols(sc: pl.DataFrame) -> list[tuple[str, str, str]]:
    """(seed label, score col, pct col): each seed and the ensemble."""
    out = [(c[1:], c, "p" + c[1:]) for c in sc.columns if re.fullmatch(r"s\d+", c)]
    if sc.select(pl.col(out[0][1]).is_null().all()).item() if out else True:
        out = []
    return [*out, ("ens", "score", "pct")]


def score_views(rows: Rows, name: str, t: pl.DataFrame) -> None:
    """AUC, top x% by training-score percentile, within-day top K, when vs which."""
    y = t["hit10"].to_numpy().astype(np.float64)
    years = t["year"].to_numpy()
    day_mean = {c: t.select(pl.col(c).mean().over("d_us")).to_series().to_numpy() for c in OUTS}
    for seed, scol, pcol in seed_cols(t):
        s = t[scol].to_numpy().astype(np.float64)
        p = t[pcol].to_numpy().astype(np.float64)
        for yr in [*TEST_YEARS, 0]:
            m = years == yr if yr else np.ones(len(y), dtype=np.bool_)
            ys = str(yr) if yr else "pooled"
            rows.add("auc", name, "auc", auc(y[m], s[m]), year=ys, seed=seed, n=float(m.sum()))
            for x in TOPS:
                sel = m & (p >= 1.0 - x)
                for c in OUTS:
                    v = t[c].to_numpy().astype(np.float64)[sel]
                    rows.add(
                        "top",
                        name,
                        c,
                        float(np.nanmean(v)) if len(v) else math.nan,
                        rule=f"top{round(x * 100)}",
                        year=ys,
                        seed=seed,
                        n=float(sel.sum()),
                    )
                if x == 0.05 and sel.any():
                    base = float(y[m].mean())
                    when = float(day_mean["hit10"][sel].mean()) - base
                    which = float((y[sel] - day_mean["hit10"][sel]).mean())
                    mon = t["month"].to_numpy()[sel]
                    _, cnt = np.unique(mon, return_counts=True)
                    top3 = float(np.sort(cnt)[::-1][:3].sum() / cnt.sum())
                    for k, val in (
                        ("when_hit10", when),
                        ("which_hit10", which),
                        ("share_top3_months", top3),
                        ("months", float(len(cnt))),
                    ):
                        rows.add("when_which", name, k, val, rule="top5", year=ys, seed=seed)
                    for c in ("ret_10",):
                        rv = t[c].to_numpy().astype(np.float64)
                        rows.add(
                            "when_which",
                            name,
                            "when_ret_10",
                            float(np.nanmean(day_mean[c][sel])) - float(np.nanmean(rv[m])),
                            rule="top5",
                            year=ys,
                            seed=seed,
                        )
                        rows.add(
                            "when_which",
                            name,
                            "which_ret_10",
                            float(np.nanmean(rv[sel] - day_mean[c][sel])),
                            rule="top5",
                            year=ys,
                            seed=seed,
                        )
        # within day: top K by score among the day's entries, excess over the day's mean
        rk = t.select(pl.col(scol).rank("ordinal", descending=True).over("d_us").alias("rk"))[
            "rk"
        ].to_numpy()
        for kk in DAY_KS:
            for yr in [*TEST_YEARS, 0]:
                m = (rk <= kk) & ((years == yr) if yr else True)
                ys = str(yr) if yr else "pooled"
                for c in ("hit10", "ret_10"):
                    v = t[c].to_numpy().astype(np.float64)
                    rows.add(
                        "within_day",
                        name,
                        f"excess_{c}",
                        float(np.nanmean(v[m] - day_mean[c][m])),
                        rule=f"top{kk}/day",
                        year=ys,
                        seed=seed,
                        n=float(m.sum()),
                    )


def base_views(rows: Rows, t: pl.DataFrame) -> None:
    for yr in [*TEST_YEARS, 0]:
        f = t if not yr else t.filter(pl.col("year") == yr)
        ys = str(yr) if yr else "pooled"
        for c in OUTS:
            rows.add("base", "every_entry", c, float(f[c].mean()), year=ys, n=float(f.height))  # type: ignore[arg-type]
        rows.add("base", "every_entry", "days", float(f["d_us"].n_unique()), year=ys)


# --- pick rules, books, random books ---------------------------------------------------------


def rules() -> list[PickRule]:
    return [PickRule(k, q, g) for g in (None, GATE) for q in RULE_QS for k in RULE_KS]


@dataclass
class YearData:
    """One test year's entries as arrays, for the same-day random draws."""

    day: I64  # day number per entry, entries sorted by day
    sym: I64
    hit10: F64
    exit_t: dict[str, I64]
    ret: dict[str, F64]
    d_us: I64


def year_data(t: pl.DataFrame) -> YearData:
    t = t.sort("d_us", "symbol")
    codes = t["symbol"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    d = t["d_us"].to_numpy().astype(np.int64)
    return YearData(
        day=(d // (86_400 * 1_000_000)).astype(np.int64),
        sym=codes,
        hit10=t["hit10"].to_numpy().astype(np.float64),
        exit_t={v: t[f"exit_t_{v}"].to_numpy().astype(np.int64) for v in STOP_VARIANTS},
        ret={v: t[f"ret_{v}"].to_numpy().astype(np.float64) for v in STOP_VARIANTS},
        d_us=d,
    )


_YD: dict[int, YearData] = {}


def _init(yd: dict[int, YearData]) -> None:
    _YD.update(yd)


def random_books(year: int, n_per_day: dict[int, int], seed: int) -> dict[str, F64]:
    """Same-day random picks: on each day with n picks, n entries drawn without replacement
    from that day's eligible entries, in random order; booked like the rule's picks.
    Returns per-draw metrics."""
    yd = _YD[year]
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
        order = np.lexsort((key, day))  # by day, random within day
        pos = np.arange(len(idx)) - starts[order]
        chosen = idx[order[pos < need[order]]]  # day order, random order within day
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
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def month_boot(month: I64, r: F64, rng: np.random.Generator) -> tuple[float, float]:
    """95% interval of the mean of r, resampling calendar months with replacement."""
    if len(r) == 0:
        return math.nan, math.nan
    mu, mi = np.unique(month, return_inverse=True)
    s = np.bincount(mi, weights=r, minlength=len(mu))
    c = np.bincount(mi, minlength=len(mu)).astype(np.float64)
    draws = rng.integers(0, len(mu), size=(N_BOOT, len(mu)))
    w = np.stack([np.bincount(d, minlength=len(mu)) for d in draws]).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (w @ s) / (w @ c)
    lo, hi = np.nanpercentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def eval_rules(
    rows: Rows, ledger: list[dict[str, Any]], name: str, t: pl.DataFrame, pool: ProcessPoolExecutor
) -> None:
    """Every pick rule on one variant's ensemble scores: picks, books, random books."""
    rng = np.random.default_rng(SEED)
    jobs = []
    for rule in rules():
        for y in TEST_YEARS:
            ty = t.filter(pl.col("year") == y)
            pk = picks(ty, rule)
            n_per_day = {
                int(d // (86_400 * 1_000_000)): int(n)
                for d, n in pk.group_by("d_us").len().iter_rows()
            }
            fut = pool.submit(random_books, y, n_per_day, SEED + y) if n_per_day else None
            jobs.append((rule, y, pk, fut))
    by_rule: dict[str, dict[str, Any]] = {}
    for rule, y, pk, fut in jobs:
        rn = rule.name
        ys = str(y)
        rnd = fut.result() if fut is not None else {}
        rec = by_rule.setdefault(rn, {"rule": rule, "trades": {}, "stress10": {}, "p95": {}})
        rows.add("picks", name, "n_picks", float(pk.height), rule=rn, year=ys)
        rows.add("picks", name, "n_days", float(pk["d_us"].n_unique()), rule=rn, year=ys)
        for c in OUTS:
            val = float(pk[c].mean()) if pk.height else math.nan  # type: ignore[arg-type]
            rows.add("picks", name, c, val, rule=rn, year=ys, n=float(pk.height))
        if "pick_hit10" in rnd:
            for qn, q in (("p50", 50), ("p95", 95)):
                rows.add(
                    "random",
                    name,
                    f"pick_hit10_{qn}",
                    float(np.nanpercentile(rnd["pick_hit10"], q)),
                    rule=rn,
                    year=ys,
                )
        for v in STOP_VARIANTS:
            for tag, cost in (("none", 0.0), ("stress", STRESS)):
                b = book(pk, v, cost)
                tr = b.trades
                n_tr = float(tr.height)
                mean_r = float(tr["ret"].mean()) if tr.height else math.nan  # type: ignore[arg-type]
                kw: dict[str, Any] = {"rule": rn, "stop": v, "stress": tag, "year": ys}
                rows.add("book", name, "trades", n_tr, **kw)
                rows.add("book", name, "mean_ret", mean_r, **kw, n=n_tr)
                rows.add("book", name, "book_return", b.total_return, **kw)
                rows.add("book", name, "max_dd", b.max_drawdown, **kw)
                if tag == "none" and tr.height:
                    hit = tr.join(pk.select("symbol", "d_us", *HITS), on=["symbol", "d_us"])
                    for h in HITS:
                        rows.add("book", name, h, float(hit[h].mean()), **kw)  # type: ignore[arg-type]
                key_r = f"ret_{v}" + ("_stress" if tag == "stress" else "")
                key_b = f"book_{v}" + ("_stress" if tag == "stress" else "")
                if key_r in rnd:
                    for qn, q in (("p50", 50), ("p95", 95)):
                        rows.add(
                            "random",
                            name,
                            f"mean_ret_{qn}",
                            float(np.nanpercentile(rnd[key_r], q)),
                            **kw,
                        )
                        rows.add(
                            "random",
                            name,
                            f"book_return_{qn}",
                            float(np.nanpercentile(rnd[key_b], q)),
                            **kw,
                        )
                    rows.add(
                        "random", name, "trades_mean", float(np.mean(rnd[f"trades_{v}"])), **kw
                    )
                if v == "10":
                    if tag == "none":
                        rec["trades"][y] = n_tr
                    else:
                        rec["stress10"][y] = mean_r
                        rec["p95"][y] = (
                            float(np.nanpercentile(rnd[key_r], 95)) if key_r in rnd else math.nan
                        )
    # pooled month bootstrap of the mean net return per trade (book trades, each stop, stress)
    for rn, rec in by_rule.items():
        rule = rec["rule"]
        for v in STOP_VARIANTS:
            for tag, cost in (("none", 0.0), ("stress", STRESS)):
                parts = [
                    book(picks(t.filter(pl.col("year") == y), rule), v, cost).trades
                    for y in TEST_YEARS
                ]
                tr = pl.concat(parts)
                mon = (
                    tr.select(
                        pl.from_epoch("d_us", time_unit="us").dt.year() * 12
                        + pl.from_epoch("d_us", time_unit="us").dt.month()
                    )
                    .to_series()
                    .to_numpy()
                    .astype(np.int64)
                )
                r = tr["ret"].to_numpy().astype(np.float64)
                lo, hi = month_boot(mon, r, rng)
                kw = {"rule": rn, "stop": v, "stress": tag, "year": "pooled"}
                rows.add("book", name, "trades", float(len(r)), **kw)
                rows.add("book", name, "mean_ret", float(r.mean()) if len(r) else math.nan, **kw)
                rows.add("book", name, "mean_ret_ci_lo", lo, **kw)
                rows.add("book", name, "mean_ret_ci_hi", hi, **kw)
        tr_min = min(rec["trades"].get(y, 0.0) for y in TEST_YEARS)
        worst = min(rec["stress10"].get(y, math.nan) for y in TEST_YEARS)
        beats = all(
            rec["stress10"].get(y, math.nan) > rec["p95"].get(y, math.nan) for y in TEST_YEARS
        )
        ledger.append(
            {
                "stage": "pick_rule",
                "variant": name,
                "rule": rn,
                "k": rule.k,
                "q": rule.q,
                "gate": rule.gate is not None,
                "min_trades_year": tr_min,
                **{f"stress_ret10_{y}": rec["stress10"].get(y, math.nan) for y in TEST_YEARS},
                **{f"random_p95_{y}": rec["p95"].get(y, math.nan) for y in TEST_YEARS},
                "worst_year_stress_ret10": worst,
                "beats_random_each_year": beats,
                "eligible": bool(tr_min >= MIN_TRADES and beats),
            }
        )


# --- choice ----------------------------------------------------------------------------------


def simplicity(r: dict[str, Any]) -> tuple[int, int, int]:
    return (SIMPLICITY.get(str(r["variant"]), 9), int(bool(r["gate"])), -int(r["k"]))


def choose(cands: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """The choice rule, fixed before the ranking was looked at.

    1. Eligible: at least 100 book trades (10% stop) in each test year, and the stress mean net
       return per trade (10% stop, book trades) above the same-day random books' 95th percentile
       in each year.
    2. Highest worst-year stress mean net return per trade (10% stop).
    3. Within 0.1 pp of the best: the simplest (transparent before lean before market before
       coin before all; no gate before gate; larger K).
    4. If the pick is a GBM variant, it must beat the best eligible transparent candidate by more
       than its seed spread (range of the worst-year metric over the single-seed books);
       otherwise the transparent candidate is frozen (done by the caller).
    5. If no candidate is eligible: the same ranking over those with at least 100 trades a year.
    """
    elig = [c for c in cands if c["eligible"]]
    note = "eligible"
    if not elig:
        elig = [c for c in cands if c["min_trades_year"] >= MIN_TRADES]
        note = "fallback: none beat the same-day random 95th percentile in every year"
    best = max(float(c["worst_year_stress_ret10"]) for c in elig)
    near = [c for c in elig if float(c["worst_year_stress_ret10"]) >= best - TIE]
    near.sort(key=simplicity)
    return near[0], note


def seed_books(t: pl.DataFrame, rule: PickRule) -> list[float]:
    """Worst-year stress mean net return per trade (10% stop) for each single-seed score."""
    out = []
    for seed in SEEDS:
        worst = math.inf
        for y in TEST_YEARS:
            ty = t.filter(pl.col("year") == y).with_columns(
                score=pl.col(f"s{seed}"), pct=pl.col(f"p{seed}")
            )
            tr = book(picks(ty, rule), "10", STRESS).trades
            worst = min(worst, float(tr["ret"].mean()) if tr.height else math.nan)  # type: ignore[arg-type]
        out.append(worst)
    return out


# --- importance and partial dependence -------------------------------------------------------


def families() -> dict[str, str]:
    fam: dict[str, str] = {}
    for line in FACTORS_MD.read_text().splitlines():
        m = re.match(r"\| \d+ \| `([a-z0-9_]+)` \| ([^|]+) \|", line)
        if m:
            fam[m.group(1)] = m.group(2).strip()
    return fam


def importance(rows: Rows, df: pl.DataFrame, v: Variant, t: pl.DataFrame) -> list[str]:
    """Grouped permutation importance on the walk-forward test years: AUC drop when a factor and
    its within-day rank are shuffled together (seed-1 model of each year, 2 repeats, at most
    40,000 test rows a year), and the same by factor family. Returns the factors by importance."""
    from model import fit

    assert v.method.settings is not None
    fam = families()
    feats = v.method.features
    base = sorted({f.removeprefix("r_") for f in feats})
    groups = {f: [c for c in feats if c in (f, "r_" + f)] for f in base}
    fgroups: dict[str, list[str]] = {}
    for f in base:
        fgroups.setdefault(fam.get(f, "?"), []).extend(groups[f])
    rng = np.random.default_rng(SEED)
    d = df["d_us"].to_numpy().astype(np.int64)
    drops: dict[str, list[float]] = {}
    for y in TEST_YEARS:
        start = year_start_us(y)
        train = df.filter(pl.Series(train_mask(d, start)))
        test = df.filter(pl.col("year") == y)
        if test.height > 40_000:
            test = test.sample(40_000, seed=SEED)
        mdl = fit(train, feats, v.method.settings, SEEDS[0])
        x = test.select(pl.col(feats).cast(pl.Float64)).to_numpy().astype(np.float64)
        yv = test["hit10"].to_numpy().astype(np.float64)
        a0 = auc(yv, mdl.est.predict_proba(x)[:, 1])
        for label, gs in [
            *(("f:" + k, g) for k, g in groups.items()),
            *(("fam:" + k, g) for k, g in fgroups.items()),
        ]:
            cols = [feats.index(c) for c in gs]
            for _ in range(2):
                xp = x.copy()
                perm = rng.permutation(len(xp))
                xp[:, cols] = x[perm][:, cols]
                drops.setdefault(label, []).append(a0 - auc(yv, mdl.est.predict_proba(xp)[:, 1]))
        print(f"  importance {v.name} {y}", flush=True)
    for label, vals in drops.items():
        kind, nm = label.split(":", 1)
        rows.add(
            "importance",
            v.name,
            "auc_drop",
            float(np.mean(vals)),
            rule=("factor:" if kind == "f" else "family:") + nm,
            seed=fam.get(nm, "") if kind == "f" else "",
        )
    ranked = sorted(groups, key=lambda f: -float(np.mean(drops["f:" + f])))
    for f in ranked[:6]:
        col = f
        tt = t.join(
            df.select("symbol", "d_us", col), on=["symbol", "d_us"], how="left", suffix="_x"
        )
        cx = col + "_x" if col in t.columns else col
        tt = tt.filter(pl.col(cx).is_not_null())
        r = tt.select((pl.col(cx).rank("average") - 0.5) / pl.len()).to_series().to_numpy()
        b = np.minimum((r * 10).astype(np.int64), 9)
        tt = tt.with_columns(dec=pl.Series(b))
        g = (
            tt.group_by("dec")
            .agg(
                n=pl.len(),
                lo=pl.col(cx).min(),
                hi=pl.col(cx).max(),
                score=pl.col("score").mean(),
                hit10=pl.col("hit10").mean(),
                ret_10=pl.col("ret_10").mean(),
            )
            .sort("dec")
        )
        for rr in g.iter_rows(named=True):
            for k in ("lo", "hi", "score", "hit10", "ret_10"):
                rows.add(
                    "pdp", v.name, k, float(rr[k]), rule=f, seed=str(rr["dec"]), n=float(rr["n"])
                )
    return ranked


# --- main ------------------------------------------------------------------------------------


def compute_scores(
    df: pl.DataFrame, refit: bool
) -> tuple[pl.DataFrame, str, list[str], list[dict[str, Any]]]:
    lean, cands = lean_set()
    if SCORES.exists() and not refit:
        sc = pl.read_parquet(SCORES)
        meta = json.loads((CACHE / "m4_scores_meta.json").read_text())
        return sc, str(meta["settings"]), lean, cands
    parts = []
    for sname in SETTINGS_ORDER:
        st = SETTINGS[sname]
        v = Variant(
            f"gbm_all[{sname}]",
            Method("gbm", gbm_variants(df, st, lean)[0].method.features, st, SEEDS),
        )
        parts.append(walk_forward(df, v))
    sc = pl.concat(parts, how="diagonal_relaxed")
    chosen = pick_settings(sc, df)
    st = SETTINGS[chosen]
    for v in gbm_variants(df, st, lean)[1:]:
        parts.append(walk_forward(df, v))
    for v in transparent_variants():
        parts.append(walk_forward(df, v))
    sc = pl.concat(parts, how="diagonal_relaxed")
    sc = sc.with_columns(
        variant=pl.when(pl.col("variant") == f"gbm_all[{chosen}]")
        .then(pl.lit("gbm_all"))
        .otherwise(pl.col("variant"))
    )
    assert int(sc["d_us"].max()) < SEALED_START_US  # type: ignore[arg-type]
    sc.write_parquet(SCORES)
    (CACHE / "m4_scores_meta.json").write_text(json.dumps({"settings": chosen}))
    return sc, chosen, lean, cands


def seed_aucs(sc: pl.DataFrame, df: pl.DataFrame, variant: str) -> F64:
    """(seeds,) mean over test years of each seed's AUC."""
    t = sc.filter(pl.col("variant") == variant).join(
        df.select("symbol", "d_us", "hit10"), on=["symbol", "d_us"]
    )
    out = []
    for s in SEEDS:
        a = [
            auc(
                t.filter(pl.col("year") == y)["hit10"].to_numpy().astype(np.float64),
                t.filter(pl.col("year") == y)[f"s{s}"].to_numpy().astype(np.float64),
            )
            for y in TEST_YEARS
        ]
        out.append(float(np.mean(a)))
    return np.asarray(out, dtype=np.float64)


def pick_settings(sc: pl.DataFrame, df: pl.DataFrame) -> str:
    """Highest mean seed AUC; a simpler setting within the seed spread of the best wins."""
    res = {n: seed_aucs(sc, df, f"gbm_all[{n}]") for n in SETTINGS_ORDER}
    best = max(res, key=lambda n: float(res[n].mean()))
    spread = float(res[best].max() - res[best].min())
    for n in SETTINGS_ORDER:  # simplest first
        if float(res[n].mean()) >= float(res[best].mean()) - spread:
            return n
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--md-only", action="store_true")
    a = ap.parse_args()
    if a.md_only:
        import m4_report

        m4_report.write_md()
        return
    df = load()
    sc, chosen_settings, lean, cands = compute_scores(df, a.refit)
    rows = Rows()
    ledger: list[dict[str, Any]] = []
    for c in cands:
        ledger.append(
            {
                "stage": "lean_candidate",
                "variant": "gbm_lean",
                "rule": c["factor"],
                "note": f"end {c['end']}, lift {c['lift']:+.4f}, ci_lo {c['ci_lo']:+.4f}, "
                f"kept {c['kept']}" + (f", clashes with {c['clash']}" if c["clash"] else ""),
            }
        )
    test_df = df.filter(pl.col("year").is_in(TEST_YEARS)).select(KEEP)
    base_views(rows, test_df)
    variants = sorted(sc["variant"].unique().to_list())
    st0 = SETTINGS[chosen_settings]
    n_features = {
        v.name: len(v.method.features)
        for v in [*gbm_variants(df, st0, lean), *transparent_variants()]
    }
    joined: dict[str, pl.DataFrame] = {}
    for vn in variants:
        t = sc.filter(pl.col("variant") == vn).drop("year").join(test_df, on=["symbol", "d_us"])
        t = t.sort("d_us", "symbol")
        joined[vn] = t
        score_views(rows, vn, t)
        sa = [float(x) for x in seed_aucs(sc, df, vn)] if vn.startswith("gbm") else []
        ens_auc = np.mean(
            [
                auc(
                    t.filter(pl.col("year") == y)["hit10"].to_numpy().astype(np.float64),
                    t.filter(pl.col("year") == y)["score"].to_numpy().astype(np.float64),
                )
                for y in TEST_YEARS
            ]
        )
        row = {
            "variant": vn,
            "rule": "",
            "n_features": n_features.get(vn.split("[")[0]),
            "mean_auc_ens": float(ens_auc),
            "mean_auc_seeds": ";".join(f"{x:.4f}" for x in sa),
        }
        if "[" in vn:
            ledger.append({"stage": "settings", **row, "settings": vn[vn.find("[") + 1 : -1]})
        else:
            st_name = chosen_settings if vn.startswith("gbm") else ""
            ledger.append({"stage": "variant", **row, "settings": st_name})
            if vn == "gbm_all":  # the chosen settings' run of the settings comparison
                ledger.append(
                    {
                        "stage": "settings",
                        **row,
                        "variant": f"gbm_all[{chosen_settings}]",
                        "settings": chosen_settings,
                        "note": "chosen: used for every GBM variant",
                    }
                )
        print(f"views {vn}", flush=True)
    yd = {y: year_data(test_df.filter(pl.col("year") == y)) for y in TEST_YEARS}
    cand_variants = [v for v in variants if "[" not in v]
    with ProcessPoolExecutor(
        max_workers=min(8, os.cpu_count() or 1), initializer=_init, initargs=(yd,)
    ) as pool:
        for vn in cand_variants:
            eval_rules(rows, ledger, vn, joined[vn], pool)
            print(f"rules {vn}", flush=True)
    pr = [r for r in ledger if r["stage"] == "pick_rule"]
    pick, note = choose(pr)
    final = pick
    seed_spread = math.nan
    decision = f"{note}; best by the rule: {pick['variant']} {pick['rule']}"
    if str(pick["variant"]).startswith("gbm"):
        sb = seed_books(
            joined[str(pick["variant"])], next(r for r in rules() if r.name == pick["rule"])
        )
        seed_spread = float(max(sb) - min(sb))
        for s, val in zip(SEEDS, sb, strict=True):
            rows.add(
                "seed_book",
                str(pick["variant"]),
                "worst_year_stress_ret10",
                val,
                rule=str(pick["rule"]),
                seed=str(s),
            )
        # the transparent candidates under the same filter as the pick (eligible, or the
        # fallback's at least 100 trades a year)
        tr = [
            r
            for r in pr
            if str(r["variant"]).startswith("rule")
            and (r["eligible"] if note == "eligible" else r["min_trades_year"] >= MIN_TRADES)
        ]
        if tr:
            tbest, _ = choose(tr)
            margin = float(pick["worst_year_stress_ret10"]) - float(
                tbest["worst_year_stress_ret10"]
            )
            decision += (
                f"; margin over transparent {tbest['variant']} {tbest['rule']}: "
                f"{margin:+.4f}, seed spread {seed_spread:.4f}"
            )
            if margin <= seed_spread:
                final = tbest
                decision += " -> transparent frozen (simpler wins)"
            else:
                decision += " -> GBM frozen"
    for r in pr:
        r["chosen"] = r is final
    ledger.append(
        {"stage": "decision", "variant": final["variant"], "rule": final["rule"], "note": decision}
    )
    print(decision, flush=True)
    final_name = str(final["variant"])
    final_rule = next(r for r in rules() if r.name == final["rule"])
    # importance: the chosen model if it is a GBM, else the full GBM for reference
    imp_variant = final_name if final_name.startswith("gbm") else "gbm_all"
    st = SETTINGS[chosen_settings]
    vmap = {v.name: v for v in [*gbm_variants(df, st, lean), *transparent_variants()]}
    ranked = importance(rows, df, vmap[imp_variant], joined[imp_variant])
    # continuous 2021-23 book and its random books, for the chosen spec
    cont = continuous(rows, final_name, final_rule, joined[final_name], test_df)
    write_outputs(rows, ledger)
    write_spec(
        final_name,
        final_rule,
        vmap[final_name],
        chosen_settings,
        lean,
        decision,
        rows,
        cont,
        seed_spread,
        ranked,
        imp_variant,
    )
    import m4_report

    m4_report.write_md()


def continuous(
    rows: Rows, name: str, rule: PickRule, t: pl.DataFrame, test_df: pl.DataFrame
) -> dict[str, float]:
    """One book over 2021-01-01 .. 2023-12-01 (each year scored by its own walk-forward model)
    against 100 same-day random books drawn over the same span."""
    pk = picks(t, rule)
    out: dict[str, float] = {}
    yd_all = year_data(test_df)
    _init({0: yd_all})
    n_per_day = {
        int(d // (86_400 * 1_000_000)): int(n) for d, n in pk.group_by("d_us").len().iter_rows()
    }
    rnd = random_books(0, n_per_day, SEED)
    for v in STOP_VARIANTS:
        for tag, cost in (("none", 0.0), ("stress", STRESS)):
            b = book(pk, v, cost)
            kw: dict[str, Any] = {"rule": rule.name, "stop": v, "stress": tag, "year": "2021-23"}
            rows.add("book_cont", name, "book_return", b.total_return, **kw)
            rows.add("book_cont", name, "max_dd", b.max_drawdown, **kw)
            rows.add("book_cont", name, "trades", float(b.trades.height), **kw)
            rows.add("book_cont", name, "mean_ret", float(b.trades["ret"].mean()), **kw)  # type: ignore[arg-type]
            key = f"book_{v}" + ("_stress" if tag == "stress" else "")
            for qn, q in (("p50", 50), ("p95", 95)):
                rows.add(
                    "book_cont",
                    name,
                    f"random_book_return_{qn}",
                    float(np.nanpercentile(rnd[key], q)),
                    **kw,
                )
            out[f"book_{v}_{tag}"] = b.total_return
            out[f"random_book_p50_{v}_{tag}"] = float(np.nanpercentile(rnd[key], 50))
            out[f"max_dd_{v}_{tag}"] = b.max_drawdown
    out["pick_hit10"] = float(pk["hit10"].mean())  # type: ignore[arg-type]
    out["random_pick_hit10_p95"] = float(np.nanpercentile(rnd["pick_hit10"], 95))
    rows.add("book_cont", name, "pick_hit10", out["pick_hit10"], rule=rule.name, year="2021-23")
    rows.add(
        "book_cont",
        name,
        "random_pick_hit10_p95",
        out["random_pick_hit10_p95"],
        rule=rule.name,
        year="2021-23",
    )
    return out


def write_outputs(rows: Rows, ledger: list[dict[str, Any]]) -> None:
    rows.frame().write_csv(RESULTS_CSV, float_precision=6)
    keys: list[str] = []
    for r in ledger:
        for k in r:
            if k not in keys:
                keys.append(k)
    lf = pl.DataFrame(
        {k: [None if r.get(k) is None else str(r.get(k)) for r in ledger] for k in keys},
        schema=dict.fromkeys(keys, pl.String),
    ).with_columns(run=pl.lit(RUN))
    if RUN1_LEDGER.exists():  # run 1 stays in the ledger: every variant tried is logged
        r1 = pl.read_csv(RUN1_LEDGER, infer_schema_length=0).with_columns(run=pl.lit(RUN1))
        if "chosen" in r1.columns:
            r1 = r1.with_columns(chosen=pl.lit("False"))
        lf = pl.concat([lf, r1], how="diagonal_relaxed")
    lf.select("run", *[c for c in lf.columns if c != "run"]).write_csv(LEDGER)


_INDEX: dict[int, dict[tuple[str, ...], float]] = {}


def get(rows: Rows, table: str, variant: str, metric: str, **kw: str) -> float:
    """Look up one value; unspecified key columns must be empty, except `year` (pooled)."""
    idx = _INDEX.get(id(rows))
    if idx is None or len(idx) == 0:
        idx = {tuple(str(x) for x in r[:8]): float(r[8]) for r in rows.rows}
        _INDEX[id(rows)] = idx
    key = {"rule": "", "stop": "", "stress": "", "year": "pooled", "seed": ""} | kw
    k = (table, variant, key["rule"], key["stop"], key["stress"], key["year"], key["seed"], metric)
    return idx.get(k, math.nan)


CONFIRMED_TEXT = (
    "Confirmed means: the picks' `hit10` rate exceeds the same-day random draws' 95th percentile "
    "overall and in at least two of the three years, and the 10%-stop book's return is positive "
    "and above the random books' median."
)
CONFIRM_TEXT = (
    "Confirm (M5), once. On 2024-26: hit rate of the picks against the base rate and against "
    "random eligible entries on the same days (100 draws); the same per year; trade result and "
    "book after costs for each stop variant, against the same-day random book."
)


def write_spec(
    name: str,
    rule: PickRule,
    v: Variant,
    settings_name: str,
    lean: list[str],
    decision: str,
    rows: Rows,
    cont: dict[str, float],
    seed_spread: float,
    ranked: list[str],
    imp_variant: str,
) -> None:
    m = v.method
    method = Method(
        m.kind, m.features, m.settings, CONFIRM_SEEDS if m.kind == "gbm" else [], m.transparent
    )
    band: dict[str, Any] = {}
    for y in TEST_YEARS:
        ys = str(y)
        kw = {"rule": rule.name, "year": ys}
        band[ys] = {
            "base_hit10": get(rows, "base", "every_entry", "hit10", year=ys),
            "base_ret_10": get(rows, "base", "every_entry", "ret_10", year=ys),
            "pick_n": get(rows, "picks", name, "n_picks", **kw),
            "pick_hit10": get(rows, "picks", name, "hit10", **kw),
            "random_pick_hit10_p95": get(rows, "random", name, "pick_hit10_p95", **kw),
            **{
                f"{k}_{s}_{tag}": get(rows, tbl, name, met, stop=s, stress=tag, **kw)
                for s in STOP_VARIANTS
                for tag in ("none", "stress")
                for k, tbl, met in (
                    ("trades", "book", "trades"),
                    ("mean_ret", "book", "mean_ret"),
                    ("book_return", "book", "book_return"),
                    ("max_dd", "book", "max_dd"),
                    ("random_book_return_p50", "random", "book_return_p50"),
                    ("random_mean_ret_p95", "random", "mean_ret_p95"),
                )
            },
        }
    pooled = {
        f"mean_ret_{s}_{tag}": get(
            rows, "book", name, "mean_ret", rule=rule.name, stop=s, stress=tag, year="pooled"
        )
        for s in STOP_VARIANTS
        for tag in ("none", "stress")
    } | {
        f"mean_ret_ci_{s}_{tag}": [
            get(
                rows,
                "book",
                name,
                "mean_ret_ci_lo",
                rule=rule.name,
                stop=s,
                stress=tag,
                year="pooled",
            ),
            get(
                rows,
                "book",
                name,
                "mean_ret_ci_hi",
                rule=rule.name,
                stop=s,
                stress=tag,
                year="pooled",
            ),
        ]
        for s in STOP_VARIANTS
        for tag in ("none", "stress")
    }

    def rng_of(key: str) -> list[float]:
        vals = [float(band[str(y)][key]) for y in TEST_YEARS]
        return [min(vals), max(vals)]

    yrs_beat = sum(
        band[str(y)]["pick_hit10"] > band[str(y)]["random_pick_hit10_p95"] for y in TEST_YEARS
    )
    spec = Spec(
        method=method,
        rule=rule,
        stop="10",
        extra={
            "name": name,
            "frozen": "M4 of docs/plans/rebound-factors.md, discovery walk-forward 2021-23 only",
            "target": "hit10: +30% within 30 days before -10% (from cache labels)",
            "settings_name": settings_name if m.kind == "gbm" else None,
            "lean_set": lean,
            "data": {
                "frame": "model.assemble(factors, labels, exclusions): within-day ranks on every "
                "panel entry of the day, inner join with the labels, anti-join the exclusions, "
                "drop entries without an outcome",
                "factors": "cache/factors.parquet",
                "exclusions": "cache/exclusions.parquet",
                "labels_discovery": "cache/labels.parquet, split == 'discovery'",
                "labels_confirm": "cache/sealed/labels.parquet (M5 only)",
            },
            "score": "GBM: mean predicted probability over the seeds (model.score_ensemble); "
            "transparent: model.transparent_score with the gate threshold from the training days",
            "training_rule": {
                "mask": "model.train_mask: D + 30 days <= period start",
                "periods": [
                    {
                        "test": "2024-01-01 .. 2024-12-31",
                        "start": "2024-01-01",
                        "train": "discovery entries (split == 'discovery', D 2020-01-31 .. "
                        "2023-12-01), all of which have windows ended before 2024-01-01",
                    },
                    {
                        "test": "2025-01-01 .. 2025-12-31",
                        "start": "2025-01-01",
                        "train": "every entry (discovery, gap and confirm, minus exclusions) "
                        "with D + 30 days <= 2025-01-01",
                    },
                    {
                        "test": "2026-01-01 .. 2026-10-02",
                        "start": "2026-01-01",
                        "train": "every entry with D + 30 days <= 2026-01-01",
                    },
                ],
                "call": "model.score_period(method, train, test, pick_rule.gate, start) per period",
            },
            "score_percentile_reference": "the training scores of the same fit, as "
            "model.score_period computes them (`reference`; `pct` = share of reference scores at "
            "or below): GBM out-of-fold scores (model.oof_scores), transparent rule its plain "
            "score on the training entries. M5 recomputes it at each (re)training.",
            "gate_threshold_rule": "pick-rule gate (if any) and transparent gate: the quantile "
            "`pct` of the gate factor over the distinct training days of the same fit",
            "book": {
                "start_equity": 10000,
                "slots": 10,
                "sizing": "realised equity / 10 at entry, capped at the cash not tied up in "
                "open positions",
                "one_position_per_coin": True,
                "order": "each day the picks in score order; no free slot or coin held: skipped",
                "slot_frees": "at the trade's exit_t (a pick on day D can use a slot whose "
                "exit_t <= D)",
                "trade_return": "ret_10 / ret_20 / ret_ns from the labels (fee 0.1% a side, 0.1% "
                "slippage on market fills)",
                "stress": "model.book(..., extra_cost=0.003): 0.3% extra a side",
                "books": "a fresh $10,000 book per calendar year, and one continuous book over "
                "the whole confirm period (the Confirmed rule's book)",
                "random_books": "100 draws: on each day with n picks, n entries drawn without "
                "replacement from that day's eligible entries (every usable entry of the day), "
                "booked the same way",
            },
            "primary_stop": "10",
            "confirm_protocol": CONFIRM_TEXT,
            "confirmed_criteria": CONFIRMED_TEXT,
            "decision": decision,
            "seed_spread_worst_year_stress_ret10": seed_spread,
            "importance_variant": imp_variant,
            "top_features": ranked[:10],
            "expected_band_discovery": {
                "per_year": band,
                "range_over_years": {
                    k: rng_of(k)
                    for k in (
                        "pick_hit10",
                        "random_pick_hit10_p95",
                        "mean_ret_10_none",
                        "mean_ret_10_stress",
                        "book_return_10_none",
                        "book_return_10_stress",
                        "random_book_return_p50_10_none",
                        "trades_10_none",
                    )
                },
                "pooled_book_trades_2021_23": pooled,
                "continuous_book_2021_23": cont,
                "discovery_would_confirm": {
                    "pick_hit10_over_random_p95_years": int(yrs_beat),
                    "overall_pick_hit10_over_random_p95": cont["pick_hit10"]
                    > cont["random_pick_hit10_p95"],
                    "book_10_positive_and_over_random_median": cont["book_10_none"] > 0
                    and cont["book_10_none"] > cont["random_book_p50_10_none"],
                },
            },
        },
    )
    SPEC.write_text(json.dumps(spec_to_json(spec), indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
