"""M4/M5: the reusable pieces of the combined model, the pick rule and the book.

M4 (`m4_model.py`) runs these on the discovery walk-forward and freezes one choice in
`results/spec.json`; M5 loads that spec with `load_spec` and calls the same functions on the
confirm period, unchanged. Nothing here reads a file except `load_spec`.

Pieces
- Features: the 65 factors of `cache/factors.parquet` plus `r_<coin factor>`, the within-day
  percentile rank of each coin factor among the day's panel entries (`add_ranks`). Ranks use
  factor values only, so they are allowed for any year.
- Training-set rule (`train_mask`): an entry may train a model that scores the period starting at
  `start` only when its 30-day window has ended by then, `D + 30 days <= start`.
- Model (`fit`, `score`): scikit-learn's `HistGradientBoostingClassifier` on `hit10` with one of
  the fixed `SETTINGS`; `max_features < 1` makes the seed matter (a feature subsample per split).
  An ensemble is the mean probability over seeds (`fit_ensemble`, `score_ensemble`).
- Transparent rule (`TransparentRule`): a market gate (a market factor at or above a percentile of
  the training days) and a within-day sort on the mean of one to two coin-factor ranks. Days
  failing the gate score below every day passing it. No fitting beyond the gate threshold.
- Score percentile (`percentile_of`): a score's position in the reference distribution, the
  scores of the training entries, so the pick threshold is usable live. For the GBM these are
  out-of-fold scores (`oof_scores`: 4 contiguous blocks of the training period, purged 30 days
  on each side), because in-sample scores of a boosted model are more extreme than its scores
  on new entries (M4 run 1 used in-sample scores; see `results/m4_model.md`).
- Pick rule (`picks`): each day, up to `k` entries with score percentile at least `q`, in score
  order, optionally only on days whose market gate holds.
- Book (`book`): $10,000, 10 equal slots of realised equity, one position per coin, each day's
  picks in score order, no free slot (or the coin already held): skipped. A slot frees at the
  trade's `exit_t`. Trade return from `ret_10` / `ret_20` / `ret_ns`, with an optional extra
  cost per side for the stress run.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from common import F64, I64, WINDOW_US
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]

type B64 = npt.NDArray[np.bool_]

TARGET_COL = "hit10"
START_EQUITY = 10_000.0
SLOTS = 10
RANK_PREFIX = "r_"
STOP_VARIANTS = ("10", "20", "ns")


# --- features --------------------------------------------------------------------------------


def factor_names(columns: list[str]) -> tuple[list[str], list[str]]:
    """(market, coin) factor names among a frame's columns, in column order."""
    market = [c for c in columns if c.startswith("m_")]
    coin = [c for c in columns if c.startswith("c_")]
    return market, coin


def add_ranks(fac: pl.DataFrame, coin: list[str]) -> pl.DataFrame:
    """Add `r_<f>` for each coin factor: (average rank - 0.5) / n among the day's non-null values.

    `fac` must hold every panel entry of each day it covers (the rank is among the day's panel
    entries), so call it on the factor table before any join with labels or exclusions.
    NaN counts as null.
    """
    fac = fac.with_columns(pl.col(coin).fill_nan(None))
    exprs = [
        ((pl.col(f).rank("average").over("d_us") - 0.5) / pl.col(f).count().over("d_us")).alias(
            RANK_PREFIX + f
        )
        for f in coin
    ]
    return fac.with_columns(exprs)


def all_features(market: list[str], coin: list[str]) -> list[str]:
    return [*market, *coin, *(RANK_PREFIX + f for f in coin)]


def with_ranks(factors: list[str]) -> list[str]:
    """A factor list plus the within-day rank of each coin factor in it."""
    return [*factors, *(RANK_PREFIX + f for f in factors if f.startswith("c_"))]


# --- training-set rule -----------------------------------------------------------------------


def train_mask(d_us: I64, start_us: int) -> B64:
    """True for entries whose 30-day window [D, D + 30 d) has ended by `start_us`."""
    return np.asarray(d_us + WINDOW_US <= start_us, dtype=np.bool_)


# --- gradient-boosted model ------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    name: str
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_features: float


SETTINGS: dict[str, Settings] = {
    s.name: s
    for s in (
        Settings("shallow", 0.05, 200, 7, 2000, 1.0, 0.5),
        Settings("base", 0.05, 300, 15, 500, 1.0, 0.5),
        Settings("deep", 0.05, 500, 31, 200, 1.0, 0.5),
    )
}


@dataclass
class Model:
    features: list[str]
    settings: Settings
    seed: int
    est: Any = field(repr=False)


def matrix(frame: pl.DataFrame, features: list[str]) -> F64:
    """Feature matrix with nulls and NaN as NaN (the model handles them)."""
    x = frame.select(pl.col(features).cast(pl.Float64)).to_numpy()
    return np.asarray(x, dtype=np.float64)


def target(frame: pl.DataFrame) -> I64:
    y = frame[TARGET_COL].cast(pl.Int64).to_numpy()
    return np.asarray(y, dtype=np.int64)


def fit(train: pl.DataFrame, features: list[str], settings: Settings, seed: int) -> Model:
    """Fit the classifier on `hit10` (rows with a null target are dropped)."""
    train = train.filter(pl.col(TARGET_COL).is_not_null())
    est = HistGradientBoostingClassifier(
        learning_rate=settings.learning_rate,
        max_iter=settings.max_iter,
        max_leaf_nodes=settings.max_leaf_nodes,
        min_samples_leaf=settings.min_samples_leaf,
        l2_regularization=settings.l2_regularization,
        max_features=settings.max_features,
        early_stopping=False,
        random_state=seed,
    )
    est.fit(matrix(train, features), target(train))
    return Model(features=list(features), settings=settings, seed=seed, est=est)


def score(model: Model, frame: pl.DataFrame) -> F64:
    """Predicted probability of `hit10` for each row of `frame`."""
    if frame.height == 0:
        return np.zeros(0, dtype=np.float64)
    p = model.est.predict_proba(matrix(frame, model.features))[:, 1]
    return np.asarray(p, dtype=np.float64)


def fit_ensemble(
    train: pl.DataFrame, features: list[str], settings: Settings, seeds: list[int]
) -> list[Model]:
    return [fit(train, features, settings, s) for s in seeds]


def score_ensemble(models: list[Model], frame: pl.DataFrame) -> F64:
    """Mean probability over the ensemble's seeds."""
    return np.asarray(np.mean([score(m, frame) for m in models], axis=0), dtype=np.float64)


# --- transparent rule ------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    """A market gate: `factor` >= its `pct` percentile over the training days."""

    factor: str
    pct: float


@dataclass(frozen=True)
class TransparentRule:
    """Gate, then sort within the day on the mean of the coin ranks (direction -1: low is good)."""

    gate: Gate | None  # None: every day passes
    coin: tuple[tuple[str, int], ...]


def gate_threshold(train: pl.DataFrame, gate: Gate) -> float:
    """The gate's threshold: the `pct` quantile of the factor over the distinct training days."""
    days = train.group_by("d_us").agg(pl.col(gate.factor).first()).drop_nulls()
    v = days[gate.factor].cast(pl.Float64).to_numpy()
    v = v[~np.isnan(v)]
    if len(v) == 0:
        raise ValueError(f"no training days with {gate.factor}")
    return float(np.quantile(v, gate.pct))


def gate_holds(frame: pl.DataFrame, gate: Gate, threshold: float) -> B64:
    v = frame[gate.factor].cast(pl.Float64).fill_nan(None).fill_null(-np.inf).to_numpy()
    return np.asarray(v >= threshold, dtype=np.bool_)


def transparent_score(rule: TransparentRule, frame: pl.DataFrame, threshold: float | None) -> F64:
    """Coin composite in [0, 1] (mean of the oriented ranks, a null rank counts as 0.5), minus 1
    on days failing the gate, so every gated-in entry outranks every gated-out one."""
    parts = []
    for f, direction in rule.coin:
        r = frame[RANK_PREFIX + f].cast(pl.Float64).fill_nan(None).fill_null(0.5).to_numpy()
        parts.append(r if direction > 0 else 1.0 - r)
    comp = np.mean(parts, axis=0)
    if rule.gate is None or threshold is None:
        return np.asarray(comp, dtype=np.float64)
    ok = gate_holds(frame, rule.gate, threshold)
    return np.asarray(np.where(ok, comp, comp - 1.0), dtype=np.float64)


def transparent_features(rule: TransparentRule) -> list[str]:
    g = [] if rule.gate is None else [rule.gate.factor]
    return [*g, *(f for f, _ in rule.coin)]


# --- score percentile ------------------------------------------------------------------------


def percentile_of(reference: F64, scores: F64) -> F64:
    """Share of the reference scores at or below each score (0..1)."""
    ref = np.sort(reference)
    return np.asarray(np.searchsorted(ref, scores, side="right") / len(ref), dtype=np.float64)


# --- one scoring period (M4 walk-forward year, M5 confirm year) ------------------------------


@dataclass(frozen=True)
class PickRule:
    k: int  # at most this many entries a day
    q: float  # score percentile at least this (against the training scores)
    gate: Gate | None  # only on days the gate holds (threshold from the training days)

    @property
    def name(self) -> str:
        g = "" if self.gate is None else f"_gate:{self.gate.factor}>=p{round(self.gate.pct * 100)}"
        return f"k{self.k}_q{self.q:.2f}{g}"


@dataclass(frozen=True)
class Method:
    """What scores the entries: a GBM ensemble or the transparent rule."""

    kind: Literal["gbm", "transparent"]
    features: list[str]
    settings: Settings | None = None
    seeds: list[int] = field(default_factory=list)
    transparent: TransparentRule | None = None


@dataclass
class Scored:
    frame: pl.DataFrame  # test entries with `score`, `pct` and (when a gate is set) `gate_ok`
    reference: F64  # training scores (GBM: out-of-fold ensemble scores)
    gate_threshold: float | None
    models: list[Model]
    seed_refs: F64  # (n_train, n_seeds) per-seed reference scores (GBM: out of fold)


def score_period(
    method: Method,
    train: pl.DataFrame,
    test: pl.DataFrame,
    gate: Gate | None,
    start_us: int,
) -> Scored:
    """Fit on `train` (asserted to obey the training-set rule for a period starting at
    `start_us`), score `test`, and attach score percentiles against the training scores."""
    d = train["d_us"].to_numpy().astype(np.int64)
    assert bool(train_mask(d, start_us).all()), "a training window overlaps the test period"
    assert int(test["d_us"].min()) >= start_us  # type: ignore[arg-type]
    models: list[Model] = []
    if method.kind == "gbm":
        assert method.settings is not None
        assert method.seeds
        models = fit_ensemble(train, method.features, method.settings, method.seeds)
        seed_refs = oof_scores(train, method.features, method.settings, method.seeds)
        ref = np.asarray(seed_refs.mean(axis=1), dtype=np.float64)
        s = score_ensemble(models, test)
    else:
        assert method.transparent is not None
        tg = method.transparent.gate
        thr_t = None if tg is None else gate_threshold(train, tg)
        ref = transparent_score(method.transparent, train, thr_t)
        s = transparent_score(method.transparent, test, thr_t)
        seed_refs = ref[:, None]
    out = test.with_columns(score=pl.Series(s), pct=pl.Series(percentile_of(ref, s)))
    thr: float | None = None
    if gate is not None:
        thr = gate_threshold(train, gate)
        out = out.with_columns(gate_ok=pl.Series(gate_holds(test, gate, thr)))
    return Scored(frame=out, reference=ref, gate_threshold=thr, models=models, seed_refs=seed_refs)


N_FOLDS = 4


def fold_of(d_us: I64, n_folds: int = N_FOLDS) -> I64:
    """Contiguous blocks of days with about equal entry counts (entries need not be sorted)."""
    order = np.argsort(d_us, kind="stable")
    cut = np.zeros(len(d_us), dtype=np.int64)
    cut[order] = np.minimum(np.arange(len(d_us)) * n_folds // max(len(d_us), 1), n_folds - 1)
    # a day must not straddle two folds: give each day the fold of its first entry
    days, first = np.unique(d_us[order], return_index=True)
    day_fold = cut[order][first]
    return np.asarray(day_fold[np.searchsorted(days, d_us)], dtype=np.int64)


def oof_mask(d_us: I64, fold: I64, k: int) -> B64:
    """Training rows for held-out fold k: other folds, minus a 30-day purge on each side of the
    held-out block (their windows overlap the held-out windows)."""
    lo = int(d_us[fold == k].min())
    hi = int(d_us[fold == k].max())
    return np.asarray((d_us < lo - WINDOW_US) | (d_us > hi + WINDOW_US), dtype=np.bool_)


def oof_scores(
    train: pl.DataFrame, features: list[str], settings: Settings, seeds: list[int]
) -> F64:
    """(n_train, n_seeds) out-of-fold scores of the training entries: the reference for the
    score percentile. In-sample scores of a boosted model are more extreme than its scores on
    new entries, so a threshold on them would be unusable live; out-of-fold scores (4 contiguous
    blocks, purged 30 days on each side) behave like scores on new data."""
    train = train.filter(pl.col(TARGET_COL).is_not_null())
    d = train["d_us"].to_numpy().astype(np.int64)
    fold = fold_of(d)
    out = np.full((train.height, len(seeds)), np.nan)
    for k in range(N_FOLDS):
        held = fold == k
        fit_rows = train.filter(pl.Series(oof_mask(d, fold, k)))
        test_rows = train.filter(pl.Series(held))
        for j, seed in enumerate(seeds):
            out[held, j] = score(fit(fit_rows, features, settings, seed), test_rows)
    assert not np.isnan(out).any()
    return np.asarray(out, dtype=np.float64)


# --- pick rule -------------------------------------------------------------------------------


def picks(scored: pl.DataFrame, rule: PickRule) -> pl.DataFrame:
    """Each day, up to `rule.k` entries with `pct >= rule.q` (and `gate_ok` when the rule has a
    gate), sorted by day, then score descending (symbol breaks ties). Adds `pick_rank` (0-based)."""
    f = scored.filter(pl.col("pct") >= rule.q)
    if rule.gate is not None:
        f = f.filter(pl.col("gate_ok"))
    f = f.sort(["d_us", "score", "symbol"], descending=[False, True, False])
    f = f.with_columns(pick_rank=pl.int_range(pl.len()).over("d_us"))
    return f.filter(pl.col("pick_rank") < rule.k)


# --- book ------------------------------------------------------------------------------------


@dataclass
class BookResult:
    trades: pl.DataFrame  # taken trades: symbol, d_us, exit_t, size, ret, pnl
    equity: pl.DataFrame  # t (µs), realised equity after each close; starts at (first D, 10,000)
    skipped: int  # picks not taken (no free slot or coin already held)

    @property
    def final(self) -> float:
        return float(self.equity["equity"][-1])

    @property
    def total_return(self) -> float:
        return self.final / START_EQUITY - 1.0

    @property
    def max_drawdown(self) -> float:
        e = self.equity["equity"].to_numpy().astype(np.float64)
        peak = np.maximum.accumulate(e)
        return float((e / peak - 1.0).min())


def stressed(ret: F64, extra_cost: float) -> F64:
    """A trade return with an extra cost per side: (1 + ret) (1 - c) / (1 + c) - 1."""
    return np.asarray((1.0 + ret) * (1.0 - extra_cost) / (1.0 + extra_cost) - 1.0)


def taken_trades(d_us: I64, symbol: Sequence[Hashable], exit_t: I64, slots: int = SLOTS) -> B64:
    """Which picks the book takes, in the given order (sorted by day, then score).

    A pick is taken when, after freeing every position with `exit_t <= D`, fewer than `slots`
    positions are open and the coin is not held. Which picks are taken does not depend on the
    returns, so one pass serves both cost levels.
    """
    n = len(d_us)
    taken = np.zeros(n, dtype=np.bool_)
    heap: list[tuple[int, int]] = []  # (exit_t, index)
    held: set[Hashable] = set()
    for i in range(n):
        d = int(d_us[i])
        while heap and heap[0][0] <= d:
            _, j = heapq.heappop(heap)
            held.discard(symbol[j])
        if len(heap) >= slots or symbol[i] in held:
            continue
        taken[i] = True
        held.add(symbol[i])
        heapq.heappush(heap, (int(exit_t[i]), i))
    return taken


def compound(
    d_us: I64, exit_t: I64, ret: F64, start: float = START_EQUITY, slots: int = SLOTS
) -> tuple[F64, I64, F64]:
    """Sizes of the taken trades (in order) and the realised equity curve (times, values).

    Each trade gets realised equity / `slots` at its entry, capped at the cash not tied up in
    open positions. Equity changes only when a trade closes.
    """
    n = len(d_us)
    sizes = np.zeros(n, dtype=np.float64)
    heap: list[tuple[int, int]] = []
    eq = start
    tied = 0.0
    times = [int(d_us[0]) if n else 0]
    curve = [start]

    def close_until(t: int) -> None:
        nonlocal eq, tied
        while heap and heap[0][0] <= t:
            te, j = heapq.heappop(heap)
            eq += sizes[j] * ret[j]
            tied -= sizes[j]
            times.append(te)
            curve.append(eq)

    for i in range(n):
        close_until(int(d_us[i]))
        size = min(eq / slots, max(eq - tied, 0.0))
        sizes[i] = size
        tied += size
        heapq.heappush(heap, (int(exit_t[i]), i))
    close_until(np.iinfo(np.int64).max)
    return sizes, np.asarray(times, dtype=np.int64), np.asarray(curve, dtype=np.float64)


def book(
    picked: pl.DataFrame,
    stop: str,
    extra_cost: float = 0.0,
    start: float = START_EQUITY,
    slots: int = SLOTS,
) -> BookResult:
    """Run the book over picks (as `picks` returns them, with the trade-result columns
    `exit_t_<stop>` and `ret_<stop>`), in day order then score order."""
    assert stop in STOP_VARIANTS
    p = picked.filter(pl.col(f"ret_{stop}").is_not_null())
    if p.height:
        d = p["d_us"].to_numpy().astype(np.int64)
        assert bool((np.diff(d) >= 0).all()), "picks must be in day order"
    d_us = p["d_us"].to_numpy().astype(np.int64)
    sym = p["symbol"].to_list()
    ex = p[f"exit_t_{stop}"].to_numpy().astype(np.int64)
    r = stressed(p[f"ret_{stop}"].to_numpy().astype(np.float64), extra_cost)
    take = taken_trades(d_us, sym, ex, slots)
    sizes, times, curve = compound(d_us[take], ex[take], r[take], start, slots)
    trades = (
        p.filter(pl.Series(take))
        .select("symbol", "d_us")
        .with_columns(
            exit_t=pl.Series(ex[take]),
            size=pl.Series(sizes),
            ret=pl.Series(r[take]),
            pnl=pl.Series(sizes * r[take]),
        )
    )
    equity = pl.DataFrame({"t": times, "equity": curve})
    return BookResult(trades=trades, equity=equity, skipped=int((~take).sum()))


# --- frames ----------------------------------------------------------------------------------


def year_start_us(year: int) -> int:
    return int(datetime(year, 1, 1, tzinfo=UTC).timestamp()) * 1_000_000


def assemble(fac: pl.DataFrame, labels: pl.DataFrame, exclusions: pl.DataFrame) -> pl.DataFrame:
    """The model frame: factors with within-day ranks (computed on `fac`, every panel entry of
    each day), inner-joined with `labels` on (symbol, d_us), minus the excluded entries, minus
    entries without an outcome (`no_entry`). Sorted by day, then symbol."""
    _market, coin = factor_names(fac.columns)
    fac = add_ranks(
        fac.with_columns(pl.col([c for c in fac.columns if c[:2] in ("m_", "c_")]).fill_nan(None)),
        coin,
    )
    df = labels.join(fac, on=["symbol", "d_us"], how="inner").join(
        exclusions.select("symbol", "d_us"), on=["symbol", "d_us"], how="anti"
    )
    return df.filter(pl.col(TARGET_COL).is_not_null()).sort("d_us", "symbol")


# --- spec ------------------------------------------------------------------------------------


@dataclass
class Spec:
    method: Method
    rule: PickRule
    stop: str
    extra: dict[str, Any]


def _gate_from(d: dict[str, Any] | None) -> Gate | None:
    return None if d is None else Gate(factor=str(d["factor"]), pct=float(d["pct"]))


def spec_to_json(spec: Spec) -> dict[str, Any]:
    m = spec.method
    return {
        "method": {
            "kind": m.kind,
            "features": m.features,
            "settings": None if m.settings is None else asdict(m.settings),
            "seeds": m.seeds,
            "transparent": None
            if m.transparent is None
            else {
                "gate": None if m.transparent.gate is None else asdict(m.transparent.gate),
                "coin": [[f, d] for f, d in m.transparent.coin],
            },
        },
        "pick_rule": {
            "k": spec.rule.k,
            "q": spec.rule.q,
            "gate": None if spec.rule.gate is None else asdict(spec.rule.gate),
            "name": spec.rule.name,
        },
        "primary_stop": spec.stop,
        **spec.extra,
    }


def spec_from_json(d: dict[str, Any]) -> Spec:
    m = d["method"]
    t = m["transparent"]
    method = Method(
        kind=m["kind"],
        features=list(m["features"]),
        settings=None if m["settings"] is None else Settings(**m["settings"]),
        seeds=[int(s) for s in m["seeds"]],
        transparent=None
        if t is None
        else TransparentRule(
            gate=_gate_from(t["gate"]), coin=tuple((str(f), int(x)) for f, x in t["coin"])
        ),
    )
    r = d["pick_rule"]
    rule = PickRule(k=int(r["k"]), q=float(r["q"]), gate=_gate_from(r["gate"]))
    extra = {k: v for k, v in d.items() if k not in ("method", "pick_rule", "primary_stop")}
    return Spec(method=method, rule=rule, stop=str(d["primary_stop"]), extra=extra)


def load_spec(path: Path) -> Spec:
    return spec_from_json(json.loads(path.read_text()))
