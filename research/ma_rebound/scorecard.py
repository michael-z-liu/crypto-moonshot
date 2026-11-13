"""M3: the scorecard every MA-rebound variant is judged with (docs/plans/ma-rebound.md).

    from scorecard import Rule, scorecard, show
    card = scorecard("every-L0.05-E1", Rule(), stage="M3", source="every candidate")
    print(show(card))

One call books a variant (a `Rule`, optionally with model scores) with `book.py` and returns its
card for 2022-24; it also upserts the variant's row in `results/ledger.csv` and writes every
year, 2025-26 included, to `results/sealed/<name>-<hash>.json` (gitignored, not to be opened
before the M6 reveal).

**Selection.** A candidate is eligible if it is of the rule's family, timeframe / length subset
and (with `majors`) a major on its trigger day. With scores (`cid -> score`, a percentile in
[0, 1]; missing or NaN means not scored, never traded) it is selected if `score >= pct`; without
scores every eligible candidate is selected (`pct` must then be None). Selection happens at the
decision; the book then takes the selected candidates' fills from the labels of the rule's
buffer, scenario and exit.

**Card** (per fill year, and 2022-24 together): trades; fill rate (distinct fills over selected
candidates, by *decision* year, over candidates whose order window ends by the book's cut; orders
refreshed by a later trigger of the same MA fill in the same candle and count once, as in M2);
mean net bp per trade; win rate; book return and max drawdown from the midnight mark-to-market
curve; mean hold in hours; per timeframe, trades and mean bp; `vs BTC`, the mean of each trade's
gross return minus BTCUSDT's over the same window (5m closes), descriptive only.

**Robustness.** *Stress*: the same selection on the stress scenario's fills and labels (its own
book). *Placebo* (scored variants only): `placebos` draws with the scores shuffled among the
eligible scored candidates decided before 2025, within UTC month x timeframe of the decision,
same threshold; books 2022-24 only. *Bootstrap*: 95% CI of the 2022-24 mean bp (the mean of the
three yearly means), resampling fill days with replacement within each year, 2,000 draws, seed 0.
*Without the 5 best days*: drop the trades of the 5 fill days (2022-24) with the largest summed
net return. *Fragile*: some year's mean bp is negative without that year's single best trade.

**Verdict** (2022-24): qualifies if every year has >= 100 trades, mean bp > 0 and book return
> 0, the stress mean bp > 0, and (scored variants) the mean bp beats the placebo's 95th
percentile. "Mean bp" is the mean of the three yearly values. *Every candidate* rows have no
placebo; their verdict says so and is judged on the other criteria. **Rank value**: the mean of
the three yearly book returns; the worst year's return breaks ties.

**Holdout.** The returned card, `show` and the ledger come from books cut at 2025-01-01 (fills
before it, curve points up to it), unless `reveal=True`. The full-range books run only to write
the sealed file. Nobody passes `reveal=True` before the M6 pick is committed and pushed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime

import book as bk
import numpy as np
import numpy.typing as npt
import polars as pl
from common import (
    BASE_PERCENTILE,
    DATA_END,
    EXITS,
    HOLDOUT_START,
    INITIAL_CAPITAL,
    MA_LENGTHS,
    RESULTS,
    SEALED,
    SELECT_YEARS,
    SLOTS,
    TIMEFRAMES,
)

from crypto_moonshot.backtest.metrics import equity_metrics, yearly_metrics

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]
type Bools = npt.NDArray[np.bool_]

LEDGER = RESULTS / "ledger.csv"
LOCK = RESULTS / "ledger.lock"
ALL_YEARS = (2022, 2023, 2024, 2025, 2026)
BOOT_DRAWS = 2000
BOOT_SEED = 0
PLACEBO_SEED = 1000
PLACEBOS = 20
MIN_TRADES = 100
BP = 1e4
TFS = tuple(TIMEFRAMES)
NAN = float("nan")


@dataclass(frozen=True)
class Rule:
    """A trading rule. `buffer` is ignored for touch-market; `pct` None means every candidate
    (scored variants default to 0.80: `scorecard` fills it in when scores are given)."""

    family: str = "limit"  # limit | touch
    buffer: float = 0.05  # 0, 0.05, 0.15 (x u above the level)
    exit: str = "E1"  # E1..E6
    pct: float | None = None
    tfs: tuple[str, ...] | None = None  # MA subset: timeframes
    lengths: tuple[int, ...] | None = None  # MA subset: lengths
    majors: bool = False  # BTC, ETH and the top 20 by `liq` on the trigger day
    slots: int = SLOTS

    def __post_init__(self) -> None:
        if self.family not in ("limit", "touch"):
            raise ValueError(f"family must be limit or touch, got {self.family!r}")
        if self.family == "limit" and self.buffer not in (0.0, 0.05, 0.15):
            raise ValueError(f"no labels for buffer {self.buffer}")
        if self.exit not in EXITS:
            raise ValueError(f"unknown exit {self.exit!r}")
        if self.tfs is not None and not set(self.tfs) <= set(TFS):
            raise ValueError(f"unknown timeframes {self.tfs}")
        lengths = {n for ns in MA_LENGTHS.values() for n in ns}
        if self.lengths is not None and not set(self.lengths) <= lengths:
            raise ValueError(f"unknown lengths {self.lengths}")

    @property
    def label_buffer(self) -> float | None:
        return None if self.family == "touch" else self.buffer

    def describe(self) -> str:
        entry = "touch-market" if self.family == "touch" else f"limit L(1+{self.buffer:g}u)"
        parts = [
            entry,
            f"exit {self.exit}",
            "every candidate" if self.pct is None else f"pct>={self.pct:g}",
        ]
        if self.tfs is not None:
            parts.append("tf " + "/".join(self.tfs))
        if self.lengths is not None:
            parts.append("n " + "/".join(str(n) for n in self.lengths))
        if self.majors:
            parts.append("majors only")
        parts.append(f"{self.slots} slots")
        return ", ".join(parts)


# --- selection -------------------------------------------------------------------------------


def eligible(rule: Rule) -> Bools:
    """Candidates of the rule's family, MA subset and majors filter."""
    c = bk.cands()
    ok = c.family == rule.family
    if rule.tfs is not None:
        ok &= np.isin(c.tf, rule.tfs)
    if rule.lengths is not None:
        ok &= np.isin(c.n, rule.lengths)
    if rule.majors:
        ok &= c.major
    return ok


def score_array(scores: pl.DataFrame) -> F64:
    """A `cid -> score` table as an array over the candidates (NaN where not scored)."""
    c = bk.cands()
    if scores["cid"].n_unique() != scores.height:
        raise ValueError("scores have duplicate cids")
    s = scores.select("cid", pl.col("score").cast(pl.Float64))
    j = pl.DataFrame({"cid": c.cid}).join(s, on="cid", how="left", maintain_order="left")
    unknown = s.join(pl.DataFrame({"cid": c.cid}), on="cid", how="anti").height
    if unknown:
        raise ValueError(f"{unknown} scored cids are not study candidates")
    return j["score"].fill_null(NAN).to_numpy().astype(np.float64)


def selected(rule: Rule, score: F64 | None) -> Bools:
    ok = eligible(rule)
    if score is None:
        return ok
    assert rule.pct is not None
    with np.errstate(invalid="ignore"):
        return ok & np.isfinite(score) & (score >= rule.pct)


def month_shuffle(score: F64, mask: Bools, rng: np.random.Generator) -> F64:
    """`score` with the `mask` candidates' values permuted within UTC month x timeframe of the
    decision."""
    c = bk.cands()
    k = np.flatnonzero(mask)
    month = c.decision_t[k].astype("datetime64[us]").astype("datetime64[M]").astype(np.int64)
    tf = np.searchsorted(np.array(sorted(TFS)), c.tf[k])
    group = month * 8 + tf
    by_group = k[np.argsort(group, kind="stable")]
    shuffled = k[np.lexsort((rng.random(len(k)), group))]
    out = score.copy()
    out[by_group] = score[shuffled]
    return out


# --- one book and its statistics -------------------------------------------------------------


def _year(t: I64) -> I64:
    return t.astype("datetime64[us]").astype("datetime64[Y]").astype(np.int64) + 1970


def year_drawdowns(equity: pl.DataFrame, years: tuple[int, ...]) -> dict[int, float]:
    """Max drawdown within each calendar year of the curve (the equity at the year's start, or
    the initial capital before the first point, counts as a peak)."""
    out: dict[int, float] = {}
    if equity.height == 0:
        return dict.fromkeys(years, 0.0)
    t = equity["time"].dt.epoch("us").to_numpy()
    e = equity["equity"].to_numpy()
    for y in years:
        a, b = bk.us(datetime(y, 1, 1, tzinfo=UTC)), bk.us(datetime(y + 1, 1, 1, tzinfo=UTC))
        i0 = int(np.searchsorted(t, a, side="right")) - 1
        start = INITIAL_CAPITAL if i0 < 0 else float(e[i0])
        seg = np.concatenate(([start], e[(t > a) & (t <= b)]))
        out[y] = float((seg / np.maximum.accumulate(seg) - 1).min())
    return out


@dataclass(frozen=True)
class Book:
    """What the scorecard keeps of one book: per-year stats and the trade list."""

    years: dict[int, dict[str, float]]
    total: dict[str, float]  # all the book's years together
    per_tf: dict[int | str, dict[str, dict[str, float]]]  # year (or "all") -> tf -> trades, bp
    trades: pl.DataFrame  # entry_time, net_return, tf
    skipped: dict[str, int]
    seconds: float


def _stats(net: F64, hold: F64, vs: F64) -> dict[str, float]:
    n = len(net)
    if n == 0:
        return {"trades": 0.0, "bp": NAN, "win": NAN, "hold_h": NAN, "vs_btc_bp": NAN}
    return {
        "trades": float(n),
        "bp": float(net.mean() * BP),
        "win": float((net > 0).mean()),
        "hold_h": float(hold.mean()),
        "vs_btc_bp": float(np.nanmean(vs) * BP) if np.isfinite(vs).any() else NAN,
    }


def fill_rates(
    rule: Rule, sel: Bools, lg: bk.Legs, until: int, years: tuple[int, ...]
) -> dict[int | str, float]:
    """Distinct fills / selected candidates per decision year, over candidates whose order
    window ends by `until` (fills of the same coin, MA and fill candle count once)."""
    c = bk.cands()
    den = sel & (c.window_end <= until) & (c.decision_t >= bk.BOOK_START_US)
    dyear = _year(c.decision_t)
    rows = np.flatnonzero(den[lg.cand] & (lg.entry_t < until))
    cand = lg.cand[rows]
    key = pl.DataFrame(
        {
            "sym": c.sym[cand],
            "tf": c.tf[cand],
            "n": c.n[cand],
            "t": lg.entry_t[rows],
            "y": dyear[cand],
        }
    ).unique(subset=["sym", "tf", "n", "t"])
    fills = dict(key.group_by("y").len().iter_rows())
    out: dict[int | str, float] = {}
    for y in years:
        d = int((den & (dyear == y)).sum())
        out[y] = fills.get(y, 0) / d if d else NAN
    d_all = int((den & np.isin(dyear, years)).sum())
    out["all"] = sum(fills.get(y, 0) for y in years) / d_all if d_all else NAN
    return out


def run_book(
    rule: Rule,
    score: F64 | None,
    scenario: str,
    until: int,
    years: tuple[int, ...],
    *,
    sel: Bools | None = None,
    light: bool = False,
) -> Book:
    """Book the rule's selection on `scenario`, fills in [2022-01-01, until). `light` skips the
    fill rates and the per-timeframe tables (placebo draws)."""
    t0 = time.perf_counter()
    c = bk.cands()
    lg = bk.legs(rule.family, rule.label_buffer, scenario, rule.exit)
    if sel is None:
        sel = selected(rule, score)
    mask = sel[lg.cand] & bk.in_book(lg, until)
    rows = bk.order_rows(lg, mask, score)
    res = bk.book(lg, rows, slots=rule.slots)
    br = res.run
    net = br.net_return
    hold = (br.exit_t - br.entry_t) / 3.6e9
    vs = lg.gross[br.row] - lg.btc[br.row]
    tf = c.tf[lg.cand[br.row]]
    fy = _year(br.entry_t)
    # the curve up to the cut: a position filled before it is not marked past it
    eq = res.equity.filter(pl.col("time") <= bk.from_us(until))
    yearly = yearly_metrics(res.trades, eq, INITIAL_CAPITAL)
    yr = {int(r["year"]): r["return"] for r in yearly.iter_rows(named=True)}
    dd = year_drawdowns(eq, years)
    out: dict[int, dict[str, float]] = {}
    for y in years:
        m = fy == y
        ret = yr.get(y)
        out[y] = _stats(net[m], hold[m], vs[m]) | {"ret": NAN if ret is None else ret, "dd": dd[y]}
    m_all = np.isin(fy, years)
    em = equity_metrics(
        eq, res.trades.filter(pl.col("entry_time").dt.year().is_in(years)), INITIAL_CAPITAL
    )
    total = _stats(net[m_all], hold[m_all], vs[m_all]) | {
        "ret": em["total_return"],
        "dd": em["max_drawdown"],
    }
    per_tf: dict[int | str, dict[str, dict[str, float]]] = {}
    if not light:
        fr = fill_rates(rule, sel, lg, until, years)
        for y in years:
            out[y]["fill_rate"] = fr[y]
        total["fill_rate"] = fr["all"]
        for key, m in [*((y, fy == y) for y in years), ("all", m_all)]:
            per_tf[key] = {
                t: {
                    "trades": float((m & (tf == t)).sum()),
                    "bp": float(net[m & (tf == t)].mean() * BP) if (m & (tf == t)).any() else NAN,
                }
                for t in TFS
            }
    trades = pl.DataFrame({"entry_time": br.entry_t, "net_return": net, "tf": tf}).with_columns(
        pl.from_epoch("entry_time", "us").dt.replace_time_zone("UTC")
    )
    return Book(out, total, per_tf, trades, dict(br.skipped), time.perf_counter() - t0)


# --- day statistics --------------------------------------------------------------------------


def _select_years(trades: pl.DataFrame) -> pl.DataFrame:
    return trades.filter(pl.col("entry_time").dt.year().is_in(SELECT_YEARS))


def yearly_bp(trades: pl.DataFrame) -> dict[int, float]:
    by = (
        _select_years(trades)
        .group_by(year=pl.col("entry_time").dt.year())
        .agg(bp=pl.col("net_return").mean() * BP)
    )
    d = {int(y): float(v) for y, v in by.iter_rows()}
    return {y: d.get(y, NAN) for y in SELECT_YEARS}


def mean_of_years(values: dict[int, float]) -> float:
    return float(np.mean([values.get(y, NAN) for y in SELECT_YEARS]))


def _day_sums(trades: pl.DataFrame) -> pl.DataFrame:
    return (
        _select_years(trades)
        .group_by(day=pl.col("entry_time").dt.date())
        .agg(year=pl.col("entry_time").dt.year().first(), s=pl.col("net_return").sum(), n=pl.len())
        .sort("day")
    )


def bootstrap_ci(trades: pl.DataFrame, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% CI of the mean of the three yearly mean bp, resampling fill days within each year."""
    days = _day_sums(trades)
    rng = np.random.default_rng(seed)
    per_year = []
    for year in SELECT_YEARS:
        d = days.filter(pl.col("year") == year)
        s, n = d["s"].to_numpy(), d["n"].to_numpy().astype(np.float64)
        if len(s) == 0:
            return NAN, NAN
        idx = rng.integers(0, len(s), (BOOT_DRAWS, len(s)))
        per_year.append(s[idx].sum(1) / n[idx].sum(1) * BP)
    lo, hi = np.percentile(np.mean(per_year, 0), [2.5, 97.5])
    return float(lo), float(hi)


def without_best_days(trades: pl.DataFrame, n_days: int = 5) -> float:
    """Mean of the yearly mean bp without the trades of the `n_days` fill days (2022-24) with
    the largest summed net return."""
    best = _day_sums(trades).sort("s", descending=True).head(n_days)["day"]
    rest = trades.filter(~pl.col("entry_time").dt.date().is_in(best.implode()))
    return mean_of_years(yearly_bp(rest))


def ex_best_trade(trades: pl.DataFrame) -> dict[int, float]:
    """Per fill year: mean bp without the year's single best trade."""
    out: dict[int, float] = {}
    for year in SELECT_YEARS:
        r = trades.filter(pl.col("entry_time").dt.year() == year)["net_return"].sort()
        out[year] = float(r[:-1].mean()) * BP if len(r) > 1 else NAN  # type: ignore[arg-type]
    return out


# --- verdict and card ------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    qualifies: bool
    criteria: dict[str, bool | None]  # None: not applicable (no placebo without scores)
    reasons: list[str]
    rank_value: float  # mean of the three yearly book returns
    worst_ret: float  # tie-break


def verdict(
    years: dict[int, dict[str, float]], stress_bp: float, mean_bp: float, placebo_p95: float | None
) -> Verdict:
    y = [years[k] for k in SELECT_YEARS]
    n = [v["trades"] for v in y]
    bp = [v["bp"] for v in y]
    ret = [v["ret"] for v in y]
    crit: dict[str, bool | None] = {
        "c_trades": all(v >= MIN_TRADES for v in n),
        "c_bp": all(v > 0 for v in bp),
        "c_return": all(v > 0 for v in ret),
        "c_stress": stress_bp > 0,
        "c_placebo": None if placebo_p95 is None else mean_bp > placebo_p95,
    }

    def mark(ok: bool | None) -> str:
        return "ok" if ok else "FAIL"

    reasons = [
        f"trades {'/'.join(f'{v:.0f}' for v in n)} {mark(crit['c_trades'])}",
        f"bp {'/'.join(f'{v:+.0f}' for v in bp)} {mark(crit['c_bp'])}",
        f"return {'/'.join(f'{v * 100:+.1f}%' for v in ret)} {mark(crit['c_return'])}",
        f"stress {stress_bp:+.1f} bp {mark(crit['c_stress'])}",
        "placebo n/a (every candidate, no scores)"
        if placebo_p95 is None
        else f"mean {mean_bp:+.1f} vs placebo p95 {placebo_p95:+.1f} bp {mark(crit['c_placebo'])}",
    ]
    ok = all(v for v in crit.values() if v is not None)
    return Verdict(ok, crit, reasons, float(np.mean(ret)), float(np.min(ret)))


@dataclass(frozen=True)
class Card:
    """A variant's card. Years are 2022-24 unless revealed."""

    name: str
    stage: str
    rule: Rule
    source: str
    config_hash: str
    scores_sha: str
    n_selected: int  # selected candidates decided in 2022-24
    years: dict[int, dict[str, float]]
    total: dict[str, float]  # 2022-24 together (pooled trades; book return over the period)
    per_tf: dict[int | str, dict[str, dict[str, float]]]
    stress: dict[int, dict[str, float]]
    placebo: list[float]  # each draw's mean bp 2022-24 (empty without scores)
    ci: tuple[float, float]
    ex5_bp: float
    ex_best: dict[int, float]
    verdict: Verdict
    seconds: float
    book_seconds: float  # mean time of one book
    sealed: str
    revealed: bool = False
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def mean_bp(self) -> float:
        return mean_of_years({y: self.years[y]["bp"] for y in SELECT_YEARS})

    @property
    def stress_bp(self) -> float:
        return mean_of_years({y: self.stress[y]["bp"] for y in SELECT_YEARS})

    @property
    def fragile(self) -> bool:
        return any(v < 0 for v in self.ex_best.values() if not math.isnan(v))

    @property
    def placebo_mean(self) -> float:
        return float(np.mean(self.placebo)) if self.placebo else NAN

    @property
    def placebo_p95(self) -> float:
        return float(np.percentile(self.placebo, 95)) if self.placebo else NAN

    @property
    def rank_value(self) -> float:
        return self.verdict.rank_value


def scores_digest(score: F64 | None) -> str:
    if score is None:
        return "-"
    return hashlib.sha1(np.ascontiguousarray(score.astype(np.float32)).tobytes()).hexdigest()[:12]


def config_hash(rule: Rule, source: str) -> str:
    return hashlib.sha1(f"{rule.describe()}|{source}".encode()).hexdigest()[:12]


def scorecard(
    name: str,
    rule: Rule,
    scores: pl.DataFrame | None = None,
    *,
    stage: str,
    source: str,
    placebos: int = PLACEBOS,
    reveal: bool = False,
    ledger: bool = True,
    seal: bool = True,
) -> Card:
    """Score a variant: see the module docstring. `scores`: `cid`, `score` (or None for every
    candidate). `source` describes where the scores come from (part of the config hash)."""
    t0 = time.perf_counter()
    if scores is not None and rule.pct is None:
        rule = replace(rule, pct=BASE_PERCENTILE)
    if scores is None and rule.pct is not None:
        raise ValueError("a percentile threshold needs scores")
    score = None if scores is None else score_array(scores)
    sel = selected(rule, score)
    cut = bk.HOLDOUT_US
    base = run_book(rule, score, "base", cut, SELECT_YEARS, sel=sel)
    stress = run_book(rule, score, "stress", cut, SELECT_YEARS, sel=sel)
    books = [base, stress]
    placebo: list[float] = []
    if score is not None and placebos:
        c = bk.cands()
        pool = eligible(rule) & np.isfinite(score) & (c.decision_t < cut)
        for j in range(placebos):
            ps = month_shuffle(score, pool, np.random.default_rng(PLACEBO_SEED + j))
            pb = run_book(rule, ps, "base", cut, SELECT_YEARS, light=True)
            books.append(pb)
            placebo.append(mean_of_years({y: pb.years[y]["bp"] for y in SELECT_YEARS}))
    mean_bp = mean_of_years({y: base.years[y]["bp"] for y in SELECT_YEARS})
    stress_bp = mean_of_years({y: stress.years[y]["bp"] for y in SELECT_YEARS})
    p95 = float(np.percentile(placebo, 95)) if placebo else None
    v = verdict(base.years, stress_bp, mean_bp, p95)
    chash = config_hash(rule, source)
    sealed = SEALED / f"{name}-{chash}.json"
    c = bk.cands()
    dsel = sel & (c.decision_t >= bk.BOOK_START_US) & (c.decision_t < cut)
    card = Card(
        name=name,
        stage=stage,
        rule=rule,
        source=source,
        config_hash=chash,
        scores_sha=scores_digest(score),
        n_selected=int(dsel.sum()),
        years=base.years,
        total=base.total,
        per_tf=base.per_tf,
        stress=stress.years,
        placebo=placebo,
        ci=bootstrap_ci(base.trades),
        ex5_bp=without_best_days(base.trades),
        ex_best=ex_best_trade(base.trades),
        verdict=v,
        seconds=0.0,
        book_seconds=float(np.mean([b.seconds for b in books])),
        sealed=str(sealed.relative_to(bk.CACHE.parent)),
        skipped=base.skipped,
    )
    if seal or reveal:
        full = run_book(rule, score, "base", bk.DATA_END_US, ALL_YEARS, sel=sel)
        full_s = run_book(rule, score, "stress", bk.DATA_END_US, ALL_YEARS, sel=sel)
        SEALED.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": name,
            "config_hash": chash,
            "rule": asdict(rule),
            "describe": rule.describe(),
            "source": source,
            "scores_sha": card.scores_sha,
            "data_end": DATA_END.isoformat(),
            "years": full.years,
            "total_2022_26": full.total,
            "per_tf": full.per_tf,
            "stress": full_s.years,
            "skipped": full.skipped,
            "card_2022_24": ledger_row(replace(card, seconds=time.perf_counter() - t0)),
        }
        tmp = sealed.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str))
        tmp.replace(sealed)
        if reveal:
            card = replace(
                card,
                years=full.years,
                total=full.total,
                per_tf=full.per_tf,
                stress=full_s.years,
                revealed=True,
            )
    card = replace(card, seconds=time.perf_counter() - t0)
    if ledger:
        write_ledger(card)
    return card


# --- seeds -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedResult:
    cards: list[Card]
    rank_values: list[float]
    spread: float  # max - min of the per-seed rank values

    @property
    def mean_rank(self) -> float:
        return float(np.mean(self.rank_values))


def score_seeds(
    name: str,
    rule: Rule,
    seeds: list[pl.DataFrame],
    *,
    stage: str,
    source: str,
    placebos: int = PLACEBOS,
    ledger: bool = True,
    seal: bool = True,
) -> SeedResult:
    """Score each seed's table as its own ledger row (`<name>-seed<i>`); returns the per-seed
    rank values and their spread (max - min)."""
    cards = [
        scorecard(
            f"{name}-seed{i}",
            rule,
            s,
            stage=stage,
            source=f"{source}, seed {i}",
            placebos=placebos,
            ledger=ledger,
            seal=seal,
        )
        for i, s in enumerate(seeds)
    ]
    ranks = [c.rank_value for c in cards]
    return SeedResult(cards, ranks, float(max(ranks) - min(ranks)))


# --- ledger and printing ---------------------------------------------------------------------


def _r(v: float, digits: int) -> float:
    return round(v, digits) if v == v else NAN


def ledger_row(card: Card) -> dict[str, object]:
    """2022-24 only (the card's own years are 2022-24 unless it was revealed)."""
    row: dict[str, object] = {
        "stage": card.stage,
        "name": card.name,
        "config_hash": card.config_hash,
        "rule": card.rule.describe(),
        "source": card.source,
        "scores_sha": card.scores_sha,
        "family": card.rule.family,
        "buffer": card.rule.label_buffer,
        "exit": card.rule.exit,
        "pct": card.rule.pct,
        "slots": card.rule.slots,
        "n_selected_2224": card.n_selected,
    }
    for y in SELECT_YEARS:
        s = card.years[y]
        row |= {
            f"trades_{y}": s["trades"],
            f"fill_{y}": _r(s.get("fill_rate", NAN), 4),
            f"bp_{y}": _r(s["bp"], 2),
            f"win_{y}": _r(s["win"], 4),
            f"ret_{y}": _r(s["ret"], 4),
            f"dd_{y}": _r(s["dd"], 4),
            f"stress_bp_{y}": _r(card.stress[y]["bp"], 2),
        }
    t = card.total
    row |= {
        "trades_2224": t["trades"],
        "fill_2224": _r(t.get("fill_rate", NAN), 4),
        "bp_pooled_2224": _r(t["bp"], 2),
        "win_2224": _r(t["win"], 4),
        "ret_2224": _r(t["ret"], 4),
        "dd_2224": _r(t["dd"], 4),
        "hold_h_2224": _r(t["hold_h"], 2),
        "vs_btc_bp_2224": _r(t["vs_btc_bp"], 2),
        "mean_bp": _r(card.mean_bp, 2),
        "stress_bp": _r(card.stress_bp, 2),
        "placebo_mean_bp": _r(card.placebo_mean, 2),
        "placebo_p95_bp": _r(card.placebo_p95, 2),
        "ci_lo_bp": _r(card.ci[0], 2),
        "ci_hi_bp": _r(card.ci[1], 2),
        "ex5_bp": _r(card.ex5_bp, 2),
        **{f"exbest_{y}": _r(card.ex_best[y], 2) for y in SELECT_YEARS},
        "fragile": card.fragile,
        "rank_value": _r(card.rank_value, 4),
        "worst_ret": _r(card.verdict.worst_ret, 4),
        **{k: ("n/a" if v is None else v) for k, v in card.verdict.criteria.items()},
        "qualifies": card.verdict.qualifies,
        "reasons": "; ".join(card.verdict.reasons),
        "per_tf_2224": "; ".join(
            f"{tf} {d['trades']:.0f}/{d['bp']:+.0f}"
            for tf, d in card.per_tf.get("all", {}).items()
            if d["trades"]
        ),
        "seconds": _r(card.seconds, 1),
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sealed": card.sealed,
    }
    return row


def write_ledger(card: Card) -> None:
    """Upsert the card's row (same name and config hash: replaced), under an exclusive lock."""
    if card.revealed:
        raise ValueError("a revealed card does not go to the ledger (2022-24 only)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    new = pl.DataFrame([ledger_row(card)], infer_schema_length=None)
    with LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if LEDGER.exists():
            old = pl.read_csv(LEDGER, infer_schema_length=None)
            same = (pl.col("name") == card.name) & (pl.col("config_hash") == card.config_hash)
            old = old.filter(~same)
            if old.height:
                new = pl.concat([old, new], how="diagonal_relaxed")
        tmp = LEDGER.with_suffix(".tmp")
        new.write_csv(tmp)
        tmp.replace(LEDGER)


def _pct(v: float) -> str:
    return f"{v * 100:+.1f}%" if v == v else "-"


def show(card: Card) -> str:
    """The card as text (2022-24 unless it was revealed)."""
    lines = [
        f"{card.name} [{card.config_hash}] {card.rule.describe()}",
        f"  source: {card.source}; selected candidates 2022-24 {card.n_selected}; "
        f"{card.seconds:.1f}s (book {card.book_seconds:.2f}s)",
        "  year     trades  fill   bp/trade  win   return  maxDD  hold_h  vsBTC | stress bp",
    ]
    for y, s in card.years.items():
        lines.append(
            f"  {y}    {s['trades']:7.0f}  {s.get('fill_rate', NAN):.2f}  {s['bp']:+8.1f}"
            f"  {s['win']:.2f}  {_pct(s['ret']):>7} {_pct(s['dd']):>6}  {s['hold_h']:6.1f}"
            f" {s['vs_btc_bp']:+6.0f} | {card.stress[y]['bp']:+7.1f}"
        )
    t = card.total
    lines.append(
        f"  2022-24 {t['trades']:7.0f}  {t.get('fill_rate', NAN):.2f}  {t['bp']:+8.1f}"
        f"  {t['win']:.2f}  {_pct(t['ret']):>7} {_pct(t['dd']):>6}  {t['hold_h']:6.1f}"
        f" {t['vs_btc_bp']:+6.0f}   (pooled)"
    )
    tfs = card.per_tf.get("all", {})
    lines.append(
        "  per tf 2022-24 (trades/bp): "
        + ", ".join(
            f"{tf} {d['trades']:.0f}/{d['bp']:+.0f}" for tf, d in tfs.items() if d["trades"]
        )
    )
    pl_txt = (
        f"placebo mean {card.placebo_mean:+.1f}, p95 {card.placebo_p95:+.1f}"
        if card.placebo
        else "no placebo"
    )
    lines += [
        f"  mean bp (of yearly) {card.mean_bp:+.1f}; stress {card.stress_bp:+.1f}; {pl_txt};"
        f" day-bootstrap 95% CI [{card.ci[0]:+.1f}, {card.ci[1]:+.1f}];"
        f" without 5 best days {card.ex5_bp:+.1f}",
        "  without each year's best trade "
        + "/".join(f"{card.ex_best[y]:+.0f}" for y in SELECT_YEARS)
        + f" bp: {'FRAGILE' if card.fragile else 'not fragile'}",
        f"  rank value {_pct(card.rank_value)} (worst year {_pct(card.verdict.worst_ret)});"
        f" verdict: {'QUALIFIES' if card.verdict.qualifies else 'fails'} -- "
        + "; ".join(card.verdict.reasons),
    ]
    return "\n".join(lines)


__all__ = ["HOLDOUT_START", "Card", "Rule", "SeedResult", "score_seeds", "scorecard", "show"]
