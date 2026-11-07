"""Scorecard for the CNN robustness study (docs/plans/cnn-robustness.md, S0).

One call scores a variant: a per-row score array aligned with `seq2_meta.parquet` (NaN means not
scored) plus a trading rule. Every book is the production engine (`crypto_moonshot.backtest`) on
5m candles, as one continuous book from 2022-01-01 to the data end (`UNTIL`).

    from scorecard import Rule, scorecard, show, wf_scores
    card = scorecard("base", wf_scores("base"), Rule(), stage="S0", source="m12 base, 2 seeds")
    print(show(card))

Per calendar year (entry year, the engine's yearly table) it gives trades, bp per trade, win
rate, $ P&L, % return and the year's max drawdown, plus:

- **stress**: a 0.6% round trip (fee 0.1% + slippage 0.2% per side) and entry one candle late
  (`entry_delay=1`), per year and averaged over 2022-24;
- **placebo**: `placebos` draws with the scores shuffled within each UTC day among that day's
  scored rows (after the row filter), same threshold and rule, books 2022-24 only;
- **bootstrap**: a 95% CI of the 2022-24 mean bp per trade, resampling entry days with
  replacement within each year (2,000 draws);
- **without the 5 best days**: the 2022-24 mean bp per trade after dropping the trades of the 5
  entry days with the largest summed net return;
- **verdict** under the plan's Selection rule, with reasons, and the ranking key.

"Mean bp per trade 2022-24" is always the average of the three yearly values, as the Selection
rule ranks. Criterion 4 reads: the variant's mean bp 2022-24 is above the 95th percentile of the
placebo draws' mean bp 2022-24.

**Masking.** 2025-26 are computed but the returned `Card`, the printout and the ledger carry
2022-24 only, unless `reveal=True`. Every year, 2025-26 included, goes to
`results/robustness/sealed/<variant>.json`, which is gitignored and must not be opened before S5
(it can be rebuilt from the local score files).

**Ledger.** Each call upserts a row in `results/robustness/ledger.csv`, keyed by variant name and
config hash (the rule, the score source description and the row filter / priority digests). A
rerun with the same config replaces its row; a different config is a new trial (a new row).

Run with `MOONSHOT_UNIVERSE=all` from this folder (it reads `cache_all/seq2_meta.parquet`).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from common import CACHE, DATA_DIR, UNIVERSE

from crypto_moonshot.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from crypto_moonshot.backtest.source import ParquetCandleSource
from crypto_moonshot.broker.base import Costs, ExitPlan
from crypto_moonshot.portfolio import EqualSlots, Sizing, WeightedSlots

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "robustness"
SEALED = OUT / "sealed"
LEDGER = OUT / "ledger.csv"

START = datetime(2022, 1, 1, tzinfo=UTC)
UNTIL = datetime(2026, 11, 2, tzinfo=UTC)  # data end (exclusive), as the golden tests
SELECT_END = datetime(2025, 1, 1, tzinfo=UTC)
YEARS = (2022, 2023, 2024, 2025, 2026)
SELECT_YEARS = (2022, 2023, 2024)
SOURCE_CACHE_BYTES = 768 * 1024**2  # per process; a 1h hold touches few month files
BOOT_DRAWS = 2000

COSTS = Costs(fee=0.001, slippage=0.0005)  # 0.3% round trip
STRESS_COSTS = Costs(fee=0.001, slippage=0.002)  # 0.6% round trip


@dataclass(frozen=True)
class Rule:
    """A trading rule on top of a score array.

    - `q`: enter rows with `score >= q` (scores are fold percentiles for m12 outputs).
    - `slots`: positions at most; `sizing` defaults to `EqualSlots(slots)`.
    - `priority` (an array passed to `scorecard`) defaults to `-score`: best score first.
    - `weights`: `(lo, hi)` makes the stake `equity / slots x w` (capped at the cash), `w` linear
      from `lo` at `score = q` to `hi` at `score = 1` (`WeightedSlots`, S1).
    - `max_per_day`: at most this many entries per UTC day, first come first served (S1).
    """

    q: float = 0.98
    slots: int = 5
    exit: ExitPlan = field(default_factory=lambda: ExitPlan(max_hold=timedelta(hours=1)))
    sizing: Sizing | None = None
    costs: Costs = COSTS
    entry_delay: int = 0
    weights: tuple[float, float] | None = None
    max_per_day: int | None = None

    def book_sizing(self) -> Sizing:
        if self.sizing is not None:
            return self.sizing
        return EqualSlots(self.slots) if self.weights is None else WeightedSlots(self.slots)

    def describe(self) -> str:
        e = self.exit
        parts = [f"pct>={self.q:g}", f"{self.slots} slots", f"sizing {self.book_sizing()!r}"]
        if e.max_hold is not None:
            parts.append(f"hold {e.max_hold}")
        for name in ("take_profit", "stop", "trailing"):
            value = getattr(e, name)
            if value is not None:
                parts.append(f"{name} {value:g}")
        if e.take_profit is not None:
            parts.append(f"tp_fill {e.tp_fill}")
        c = self.costs
        parts.append(f"fee {c.fee:g} slip {c.slippage:g} lat {c.latency:g} per side")
        if self.entry_delay:
            parts.append(f"entry_delay {self.entry_delay}")
        if self.weights is not None:
            parts.append(f"weight {self.weights[0]:g}..{self.weights[1]:g} over pct q..1")
        if self.max_per_day is not None:
            parts.append(f"max {self.max_per_day} entries per day")
        return ", ".join(parts)


# ----------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Meta:
    """The rows of `seq2_meta.parquet`: symbol, signal candle open time, UTC day number."""

    symbol: np.ndarray
    open_time: pl.Series
    t_us: np.ndarray
    day: np.ndarray

    def __len__(self) -> int:
        return len(self.t_us)


_META: Meta | None = None
WF = CACHE / "wf"


def wf_scores(name: str, key: str = "pct") -> np.ndarray:
    """A walk-forward score array from `cache_<universe>/wf/<name>.npz` (row-aligned with
    `seq2_meta.parquet`; NaN outside the test years)."""
    with np.load(WF / f"{name}.npz") as f:
        return np.asarray(f[key], dtype=np.float64)


def meta() -> Meta:
    global _META
    if _META is None:
        m = pl.read_parquet(CACHE / "seq2_meta.parquet", columns=["symbol", "open_time"])
        t = m["open_time"].dt.convert_time_zone("UTC").dt.cast_time_unit("us")
        t_us = t.dt.epoch("us").to_numpy().astype(np.int64)
        _META = Meta(
            symbol=m["symbol"].cast(pl.String).to_numpy(),
            open_time=t,
            t_us=t_us,
            day=t_us // 86_400_000_000,
        )
    return _META


def _us(t: datetime) -> int:
    return int((t - datetime(1970, 1, 1, tzinfo=UTC)) / timedelta(microseconds=1))


def in_range(m: Meta, end: datetime = UNTIL) -> np.ndarray:
    return (m.t_us >= _us(START)) & (m.t_us < _us(end))


def signals(
    score: np.ndarray,
    rule: Rule,
    row_filter: np.ndarray | None = None,
    priority: np.ndarray | None = None,
    end: datetime = UNTIL,
) -> pl.DataFrame:
    """The engine's signal frame: candidate rows `score >= q` in [START, end), with `row`."""
    m = meta()
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(score) & (score >= rule.q) & in_range(m, end)
    if row_filter is not None:
        ok &= row_filter
    k = np.flatnonzero(ok)
    pr = -score[k] if priority is None else priority[k]
    cols: dict[str, object] = {
        "symbol": m.symbol[k],
        "open_time": m.open_time.gather(k),
        "priority": np.asarray(pr, dtype=np.float64),
        "row": k.astype(np.int64),
    }
    if rule.weights is not None:
        lo, hi = rule.weights
        frac = np.clip((score[k] - rule.q) / (1.0 - rule.q), 0.0, 1.0)
        cols["weight"] = lo + (hi - lo) * frac
    return pl.DataFrame(cols)


def day_shuffle(score: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """`score` with the values of the `mask` rows permuted at random within each UTC day."""
    day = meta().day
    k = np.flatnonzero(mask)
    by_day = k[np.argsort(day[k], kind="stable")]
    shuffled = k[np.lexsort((rng.random(len(k)), day[k]))]
    out = score.copy()
    out[by_day] = score[shuffled]
    return out


# ----------------------------------------------------------------------------- engine runs

_SOURCE: ParquetCandleSource | None = None


def source() -> ParquetCandleSource:
    """One candle source per process, so its month cache is reused across runs."""
    global _SOURCE
    if _SOURCE is None:
        _SOURCE = ParquetCandleSource(DATA_DIR, until=UNTIL, cache_bytes=SOURCE_CACHE_BYTES)
    return _SOURCE


def book(sig: pl.DataFrame, rule: Rule, end: datetime | None = None) -> BacktestResult:
    """One continuous engine book from START (to `end`, else the data end)."""
    config = BacktestConfig(
        exit=rule.exit,
        entry_delay=rule.entry_delay,
        costs=rule.costs,
        sizing=rule.book_sizing(),
        max_positions=rule.slots,
        max_entries_per_day=rule.max_per_day,
        start=START,
        end=end,
    )
    return run_backtest(sig, source(), config)


@dataclass(frozen=True)
class Run:
    """What the scorecard keeps of one engine book."""

    years: dict[int, dict[str, float]]
    trades: pl.DataFrame  # entry_time, net_return, pnl
    skipped: dict[str, int]
    ex_best: dict[int, float]  # bp per trade by year without that year's single best trade


def ex_best_trade(trades: pl.DataFrame) -> dict[int, float]:
    """Per entry year: mean bp per trade without the year's single best trade."""
    out: dict[int, float] = {}
    for year in SELECT_YEARS:
        r = trades.filter(pl.col("entry_time").dt.year() == year)["net_return"].sort()
        out[year] = float(r[:-1].mean()) * 1e4 if len(r) > 1 else float("nan")  # type: ignore[arg-type]
    return out


def year_drawdowns(equity: pl.DataFrame, initial: float) -> dict[int, float]:
    """Max drawdown within each calendar year of the mark-to-market curve (peak reset at the
    year's start, the equity there counting as a peak)."""
    if equity.height == 0:
        return {}
    t = equity["time"].dt.epoch("us").to_numpy().astype(np.int64)
    e = equity["equity"].to_numpy().astype(np.float64)
    out = {}
    for year in range(equity["time"][0].year, equity["time"][-1].year + 1):
        a = _us(datetime(year, 1, 1, tzinfo=UTC))
        b = _us(datetime(year + 1, 1, 1, tzinfo=UTC))
        i0 = int(np.searchsorted(t, a, side="right")) - 1
        start = initial if i0 < 0 else float(e[i0])
        seg = np.concatenate(([start], e[(t > a) & (t <= b)]))
        out[year] = float((seg / np.maximum.accumulate(seg) - 1).min())
    return out


def run_book(sig: pl.DataFrame, rule: Rule, end: datetime | None = None) -> Run:
    res = book(sig, rule, end)
    dd = year_drawdowns(res.equity, 10_000.0)
    years: dict[int, dict[str, float]] = {}
    for row in res.yearly.iter_rows(named=True):
        y = int(row["year"])
        years[y] = {
            "trades": float(row["trades"]),
            "bp": _f(row["mean_bp"]),
            "win": _f(row["win_rate"]),
            "pnl": _f(row["pnl"]),
            "ret": _f(row["return"]),
            "dd": dd.get(y, float("nan")),
        }
    trades = res.trades.select("entry_time", "net_return", "pnl")
    return Run(years, trades, dict(res.skipped), ex_best_trade(trades))


def _f(v: object) -> float:
    return float("nan") if v is None else float(v)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- statistics


def mean_bp(years: dict[int, dict[str, float]], which: tuple[int, ...] = SELECT_YEARS) -> float:
    """Average of the yearly bp per trade (a year without trades counts as NaN)."""
    return float(np.mean([years.get(y, {}).get("bp", float("nan")) for y in which]))


def _day_sums(trades: pl.DataFrame) -> pl.DataFrame:
    return (
        trades.filter(pl.col("entry_time").dt.year().is_in(SELECT_YEARS))
        .group_by(day=pl.col("entry_time").dt.date())
        .agg(year=pl.col("entry_time").dt.year().first(), s=pl.col("net_return").sum(), n=pl.len())
        .sort("day")
    )


def bootstrap_ci(trades: pl.DataFrame, seed: int = 0) -> tuple[float, float]:
    """95% CI of the 2022-24 mean bp per trade, resampling entry days within each year."""
    days = _day_sums(trades)
    rng = np.random.default_rng(seed)
    per_year = []
    for year in SELECT_YEARS:
        d = days.filter(pl.col("year") == year)
        s, n = d["s"].to_numpy(), d["n"].to_numpy().astype(np.float64)
        if len(s) == 0:
            return float("nan"), float("nan")
        idx = rng.integers(0, len(s), (BOOT_DRAWS, len(s)))
        per_year.append(s[idx].sum(1) / n[idx].sum(1) * 1e4)
    stat = np.mean(per_year, 0)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(lo), float(hi)


def without_best_days(trades: pl.DataFrame, n_days: int = 5) -> float:
    """2022-24 mean bp per trade without the `n_days` entry days of largest summed return."""
    days = _day_sums(trades)
    best = days.sort("s", descending=True).head(n_days)["day"]
    rest = trades.filter(
        pl.col("entry_time").dt.year().is_in(SELECT_YEARS)
        & ~pl.col("entry_time").dt.date().is_in(best.implode())
    )
    by = rest.group_by(year=pl.col("entry_time").dt.year()).agg(bp=pl.col("net_return").mean())
    bp = {int(y): float(v) * 1e4 for y, v in by.iter_rows()}
    return float(np.mean([bp.get(y, float("nan")) for y in SELECT_YEARS]))


@dataclass(frozen=True)
class Verdict:
    qualifies: bool
    criteria: dict[str, bool]
    reasons: list[str]
    rank_key: tuple[float, float]  # (mean bp 2022-24, worst year bp 2022-24)


def verdict(
    years: dict[int, dict[str, float]],
    cost06: dict[int, dict[str, float]],
    late: dict[int, dict[str, float]],
    placebo_p95: float,
) -> Verdict:
    bp = [years.get(y, {}).get("bp", float("nan")) for y in SELECT_YEARS]
    m, m06, mlate = mean_bp(years), mean_bp(cost06), mean_bp(late)
    c = {
        "c1_every_year": all(v > 0 for v in bp),
        "c2_cost06": m06 > 0,
        "c3_late": mlate > 0,
        "c4_placebo": m > placebo_p95,
    }
    reasons = [
        f"c1 yearly bp {'/'.join(f'{v:+.0f}' for v in bp)}"
        + (" all > 0" if c["c1_every_year"] else " not all > 0"),
        f"c2 0.6% cost mean {m06:+.1f} bp" + (" > 0" if c["c2_cost06"] else " <= 0"),
        f"c3 one candle late mean {mlate:+.1f} bp" + (" > 0" if c["c3_late"] else " <= 0"),
        f"c4 mean {m:+.1f} vs placebo p95 {placebo_p95:+.1f} bp"
        + (" beats" if c["c4_placebo"] else " does not beat"),
    ]
    return Verdict(all(c.values()), c, reasons, (m, float(np.min(bp))))


# ----------------------------------------------------------------------------- the card


@dataclass(frozen=True)
class Card:
    """A variant's scorecard. `years`, `cost06`, `late` hold 2022-24 only unless revealed."""

    variant: str
    stage: str
    rule: str
    source: str
    config_hash: str
    scores_sha: str
    n_signals: int  # candidate rows 2022-24
    years: dict[int, dict[str, float]]
    cost06: dict[int, dict[str, float]]
    late: dict[int, dict[str, float]]
    placebo: list[float]  # each draw's mean bp 2022-24
    ci: tuple[float, float]
    ex5_bp: float
    ex_best: dict[int, float]  # 2022-24 bp by year without that year's best trade
    verdict: Verdict
    seconds: float
    sealed: Path

    @property
    def mean_bp(self) -> float:
        return mean_bp(self.years)

    @property
    def fragile(self) -> bool:
        """Some 2022-24 year turns negative without that year's single best trade."""
        return any(v < 0 for v in self.ex_best.values())

    @property
    def placebo_mean(self) -> float:
        return float(np.mean(self.placebo))

    @property
    def placebo_p95(self) -> float:
        return float(np.percentile(self.placebo, 95))


def _digest(a: np.ndarray | None) -> str:
    if a is None:
        return "-"
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def config_hash(rule: Rule, source: str, row_filter: np.ndarray | None,
                priority: np.ndarray | None, placebos: int) -> str:  # fmt: skip
    text = "|".join(
        [UNIVERSE, rule.describe(), source, _digest(row_filter), _digest(priority), str(placebos)]
    )
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))


def scorecard(
    variant: str,
    score: np.ndarray,
    rule: Rule | None = None,
    *,
    stage: str = "S0",
    source: str = "",
    row_filter: np.ndarray | None = None,
    priority: np.ndarray | None = None,
    placebos: int = 20,
    workers: int = 6,
    reveal: bool = False,
    ledger: bool = True,
) -> Card:
    """Score `variant` (see the module docstring). `score`, `row_filter` and `priority` are
    aligned with the rows of `seq2_meta.parquet`. `source` describes where the scores come
    from (it is part of the config hash)."""
    t0 = time.time()
    rule = rule or Rule()
    m = meta()
    if len(score) != len(m):
        msg = f"score has {len(score)} rows, seq2_meta has {len(m)}"
        raise ValueError(msg)
    score = np.asarray(score, dtype=np.float64)
    # keep the rule's fee and latency; only slippage rises to the 0.6% round trip
    stress = replace(rule, costs=replace(rule.costs, slippage=STRESS_COSTS.slippage))
    late = replace(rule, entry_delay=rule.entry_delay + 1)
    sig = signals(score, rule, row_filter, priority)
    scored = np.isfinite(score) & in_range(m, SELECT_END)
    if row_filter is not None:
        scored &= row_filter
    with _pool(workers) as pool:
        jobs: dict[str, Future[Run]] = {
            "base": pool.submit(run_book, sig, rule),
            "cost06": pool.submit(run_book, sig, stress),
            "late": pool.submit(run_book, sig, late),
        }
        for j in range(placebos):
            shuffled = day_shuffle(score, scored, np.random.default_rng(1000 + j))
            ps = signals(shuffled, rule, row_filter, priority, end=SELECT_END)
            jobs[f"placebo{j}"] = pool.submit(run_book, ps, rule, SELECT_END)
        runs = {k: f.result() for k, f in jobs.items()}
    base, c06, lt = runs["base"], runs["cost06"], runs["late"]
    placebo = [mean_bp(runs[f"placebo{j}"].years) for j in range(placebos)]
    p95 = float(np.percentile(placebo, 95)) if placebo else float("nan")
    v = verdict(base.years, c06.years, lt.years, p95)
    chash = config_hash(rule, source, row_filter, priority, placebos)
    sealed = SEALED / f"{variant}.json"
    SEALED.mkdir(parents=True, exist_ok=True)
    sealed.write_text(
        json.dumps(
            {
                "variant": variant,
                "config_hash": chash,
                "rule": rule.describe(),
                "source": source,
                "years": base.years,
                "cost06": c06.years,
                "late": lt.years,
                "skipped": base.skipped,
            },
            indent=1,
            default=str,
        )
    )
    shown = YEARS if reveal else SELECT_YEARS

    def keep(d: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
        return {y: d[y] for y in shown if y in d}

    card = Card(
        variant=variant,
        stage=stage,
        rule=rule.describe(),
        source=source,
        config_hash=chash,
        scores_sha=_digest(score.astype(np.float32)),
        n_signals=int(sig.filter(pl.col("open_time") < SELECT_END).height),
        years=keep(base.years),
        cost06=keep(c06.years),
        late=keep(lt.years),
        placebo=placebo,
        ci=bootstrap_ci(base.trades),
        ex5_bp=without_best_days(base.trades),
        ex_best=base.ex_best,
        verdict=v,
        seconds=time.time() - t0,
        sealed=sealed,
    )
    if ledger:
        write_ledger(card)
    return card


# ----------------------------------------------------------------------------- output


_NO_TRADES = {"trades": 0.0, "bp": float("nan"), "win": float("nan"), "pnl": 0.0,
              "ret": float("nan"), "dd": float("nan")}  # fmt: skip


def ledger_row(card: Card) -> dict[str, object]:
    # A year without trades (e.g. a stacker that starts in 2023) is a row of NaN, not an error.
    y = {yr: card.years.get(yr, _NO_TRADES) for yr in SELECT_YEARS}
    row: dict[str, object] = {
        "stage": card.stage,
        "variant": card.variant,
        "rule": card.rule,
        "source": card.source,
        "config_hash": card.config_hash,
        "scores_sha": card.scores_sha,
        "universe": UNIVERSE,
        "n_signals_2224": card.n_signals,
    }
    for yr in SELECT_YEARS:
        row |= {f"trades_{yr}": y[yr]["trades"], f"bp_{yr}": round(y[yr]["bp"], 2)}
    for yr in SELECT_YEARS:
        row |= {f"ret_{yr}": round(y[yr]["ret"], 4), f"dd_{yr}": round(y[yr]["dd"], 4)}
    trades = sum(y[yr]["trades"] for yr in SELECT_YEARS)
    row |= {
        "win_2224": round(
            sum(y[yr]["win"] * y[yr]["trades"] for yr in SELECT_YEARS if y[yr]["trades"]) / trades,
            4,
        ),
        "pnl_2224": round(sum(y[yr]["pnl"] for yr in SELECT_YEARS), 2),
        "mean_bp": round(card.mean_bp, 2),
        "worst_bp": round(card.verdict.rank_key[1], 2),
        "cost06_bp": round(mean_bp(card.cost06), 2),
        "late_bp": round(mean_bp(card.late), 2),
        "placebo_mean_bp": round(card.placebo_mean, 2),
        "placebo_p95_bp": round(card.placebo_p95, 2),
        "ci_lo_bp": round(card.ci[0], 2),
        "ci_hi_bp": round(card.ci[1], 2),
        "ex5_bp": round(card.ex5_bp, 2),
        **{f"exbest_{yr}": round(card.ex_best[yr], 2) for yr in SELECT_YEARS},
        "fragile": card.fragile,
        **card.verdict.criteria,
        "qualifies": card.verdict.qualifies,
        "reasons": "; ".join(card.verdict.reasons),
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sealed": str(card.sealed.relative_to(HERE)),
    }
    return row


def write_ledger(card: Card) -> None:
    """Upsert the card's row (same variant and config hash: replaced).

    Several stages run at once, so the read-modify-write holds an exclusive lock on a side file.
    """
    import fcntl

    OUT.mkdir(parents=True, exist_ok=True)
    new = pl.DataFrame([ledger_row(card)])
    with (OUT / "ledger.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if LEDGER.exists():
            old = pl.read_csv(LEDGER, infer_schema_length=None)
            same = (pl.col("variant") == card.variant) & (pl.col("config_hash") == card.config_hash)
            old = old.filter(~same)
            if old.height:
                new = pl.concat([old, new], how="diagonal_relaxed")
        tmp = LEDGER.with_suffix(".tmp")
        new.write_csv(tmp)
        tmp.replace(LEDGER)


def show(card: Card) -> str:
    """A text table of the card (2022-24 unless the card was revealed)."""
    lines = [
        f"{card.variant} [{card.config_hash}] {card.rule}",
        f"  candidates 2022-24 {card.n_signals}, {card.seconds:.0f}s",
        "  year  trades     bp   win     pnl$    ret%    dd%  | bp 0.6%  bp late",
    ]
    for yr, s in card.years.items():
        lines.append(
            f"  {yr}  {s['trades']:6.0f} {s['bp']:+6.1f} {s['win']:5.2f} {s['pnl']:+8.0f}"
            f" {s['ret'] * 100:+7.1f} {s['dd'] * 100:+6.1f}  |"
            f" {card.cost06[yr]['bp']:+7.1f} {card.late[yr]['bp']:+8.1f}"
        )
    lines += [
        f"  mean 2022-24 {card.mean_bp:+.1f} bp (worst year {card.verdict.rank_key[1]:+.1f});"
        f" 0.6% {mean_bp(card.cost06):+.1f}; late {mean_bp(card.late):+.1f}",
        f"  placebo mean {card.placebo_mean:+.1f}, p95 {card.placebo_p95:+.1f};"
        f" day-bootstrap 95% CI [{card.ci[0]:+.1f}, {card.ci[1]:+.1f}];"
        f" without 5 best days {card.ex5_bp:+.1f}",
        "  without each year's best trade "
        + "/".join(f"{card.ex_best[y]:+.0f}" for y in SELECT_YEARS)
        + f" bp: {'FRAGILE' if card.fragile else 'not fragile'}",
        f"  verdict: {'QUALIFIES' if card.verdict.qualifies else 'fails'} -- "
        + "; ".join(card.verdict.reasons),
    ]
    return "\n".join(lines)
