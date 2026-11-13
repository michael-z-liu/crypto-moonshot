"""Issue #10, M2: cascade caps and score heads on base3's scores, no training.

    cd research/volume_spike
    export MOONSHOT_UNIVERSE=all
    uv run --group research python i10_m2.py score [--only daycap5,esc] [--no-seeds]
    uv run --group research python i10_m2.py cascades     # candidate-only cascade statistics
    uv run --group research python i10_m2.py table        # the results table (markdown)

Fourteen fixed rules (docs/plans/cnn-improvements.md, M2), each a ledger row
`M2:base3:<rule>` (20 placebos) plus per-seed rows in `seeds.csv`; every seed uses its own
percentiles and its own cascades. Unless a rule says otherwise it is S5's `a_1h`: pct >= 0.98,
`CashCapped(5)`, highest score first, 1h hold, 0.4% round trip.

Definitions (from candidates only, never from returns):
- candidate: a scored row with pct >= 0.98 in [2022-01-01, 2026-11-02);
- cascade: a maximal run of candidates across all symbols, sorted by open time, with
  consecutive candidates at most 60 minutes apart;
- cascade age of a candidate: minutes since its cascade's first candidate.

Rules (h1 head, 1h hold unless stated):
- `daycap5`, `daycap10`: at most 5 / 10 entries per UTC day (`Rule.max_per_day`);
- `casc5`, `casc10`: at most 5 / 10 entries per cascade (engine `max_entries_per_group`);
- `dll3`, `dll6`: no new entries for the rest of the UTC day once the day's realised P&L has
  fallen 3% / 6% of the day's starting realised equity (`risk.DailyLossLimit`: realised only,
  open positions are not marked to market);
- `age60`, `age180`: only candidates with cascade age <= 60 / 180 minutes (a row filter);
- `esc`: entries 1-5 of a cascade need pct >= 0.98, 6-10 need >= 0.99, later ones >= 0.995
  (engine `group_tiers` ((5, 0.99), (10, 0.995)): counts the entries the book took);
- `h2_1h`, `h2_2h`, `h2ra_1h`, `h2ra_2h`: the 2h-median head and the 2h-median / (q90 - q10)
  head, with a 1h or a 2h hold; `h1_2h`: the h1 head with a 2h hold (reference for the 2h rows).

Placebos (the scorecard shuffles scores within each UTC day over the rows that pass the row
filter): the age rules' row filter removes the failing candidates (it is True for every other
row) and is computed once from the real scores, so a placebo draw has as many candidates as the
real book had surviving ones. Priority is the default (-score, of the shuffled scores). The group
limits' `group` and `score` columns are recomputed from each draw's own shuffled scores
(cascades of the placebo's candidates); caps, day caps and loss limits are applied as in the real
book.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, replace

import i10_score as sc10
import numpy as np
import polars as pl
import s5_confirm
import scorecard as sc

NAME = "base3"
STAGE = "M2"
Q = 0.98
GAP_US = 60 * 60 * 1_000_000
ESC_TIERS = ((5, 0.99), (10, 0.995))


def candidate_mask(score: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.isfinite(score) & (score >= Q) & sc.in_range(sc.meta())


@dataclass(frozen=True)
class Cascades:
    """Per row: cascade id (-1 for non-candidates) and cascade age in minutes (NaN)."""

    group: np.ndarray
    age_min: np.ndarray
    n: int


def cascades(score: np.ndarray) -> Cascades:
    m = sc.meta()
    k = np.flatnonzero(candidate_mask(score))
    k = k[np.argsort(m.t_us[k], kind="stable")]
    t = m.t_us[k]
    new = np.concatenate(([True], np.diff(t) > GAP_US)) if len(k) else np.zeros(0, bool)
    cid = np.cumsum(new) - 1
    first = t[np.flatnonzero(new)][cid] if len(k) else t
    group = np.full(len(m), -1, dtype=np.int64)
    age = np.full(len(m), np.nan)
    group[k] = cid
    age[k] = (t - first) / 60e6
    return Cascades(group, age, int(new.sum()))


def group_cols(score: np.ndarray) -> dict[str, np.ndarray]:
    return {"group": cascades(score).group, "score": score}


def age_filter(limit_min: float) -> Callable[[np.ndarray], np.ndarray]:
    def fn(score: np.ndarray) -> np.ndarray:
        age = cascades(score).age_min
        with np.errstate(invalid="ignore"):
            return ~(age > limit_min)  # True except candidates older than the limit

    return fn


@dataclass(frozen=True)
class M2Rule:
    name: str
    head: str
    rule: sc.Rule
    filter_fn: Callable[[np.ndarray], np.ndarray] | None = None
    cols: Callable[[np.ndarray], dict[str, np.ndarray]] | None = None
    note: str = ""


def rules() -> dict[str, M2Rule]:
    one, two = s5_confirm.rules()["a_1h"][1], s5_confirm.rules()["d_hold2h"][1]
    out = [
        M2Rule("daycap5", "h1", replace(one, max_per_day=5)),
        M2Rule("daycap10", "h1", replace(one, max_per_day=10)),
        M2Rule("casc5", "h1", replace(one, max_per_group=5), cols=group_cols),
        M2Rule("casc10", "h1", replace(one, max_per_group=10), cols=group_cols),
        M2Rule("dll3", "h1", replace(one, daily_loss_limit=0.03)),
        M2Rule("dll6", "h1", replace(one, daily_loss_limit=0.06)),
        M2Rule("age60", "h1", one, filter_fn=age_filter(60)),
        M2Rule("age180", "h1", one, filter_fn=age_filter(180)),
        M2Rule("esc", "h1", replace(one, group_tiers=ESC_TIERS), cols=group_cols),
        M2Rule("h2_1h", "h2", one),
        M2Rule("h2_2h", "h2", two),
        M2Rule("h2ra_1h", "h2ra", one),
        M2Rule("h2ra_2h", "h2ra", two),
        M2Rule("h1_2h", "h1", two),
    ]
    return {r.name: r for r in out}


def variant(r: M2Rule) -> str:
    return f"{STAGE}:{NAME}:{r.name}"


def run(only: set[str] | None, seeds: bool) -> None:
    for r in rules().values():
        if only and r.name not in only:
            continue
        v = variant(r)
        extra = f"M2 rule {r.name}"
        kw = dict(
            stage=STAGE, rule=r.rule, variant=v, extra=extra, filter_fn=r.filter_fn,
            signal_cols=r.cols,
        )  # fmt: skip
        sc10.score_variant(NAME, r.head, workers=4, **kw)  # type: ignore[arg-type]
        if seeds:
            sc10.score_seeds(NAME, r.head, workers=4, **kw)  # type: ignore[arg-type]


def cascade_stats() -> str:
    """Candidate-only statistics of the cascades, 2022-24, for the seed-mean h1 scores and for
    each seed: cascades a year, candidates in the largest one, its span and start."""
    m = sc.meta()
    arrays = {"seed mean": sc10.scores(NAME, "h1")} | {
        f"seed {j}": s for j, s in enumerate(sc10.seed_scores(NAME, "h1"))
    }
    lines = [
        "| Scores | Cascades 2022 / 2023 / 2024 | Candidates 2022 / 2023 / 2024 |"
        " Largest (candidates, symbols, span, first candidate) | Median size | Cascades > 10 |",
        "|---|---|---|---|---|---|",
    ]
    ts = pl.Series(m.t_us).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    for label, score in arrays.items():
        c = cascades(score)
        k = np.flatnonzero(c.group >= 0)
        k = k[ts.gather(k).dt.year().to_numpy() <= 2024]
        df = pl.DataFrame({"g": c.group[k], "t": ts.gather(k), "s": m.symbol[k]}).with_columns(
            year=pl.col("t").dt.year()
        )
        g = df.group_by("g").agg(
            n=pl.len(), syms=pl.col("s").n_unique(), t0=pl.col("t").min(), t1=pl.col("t").max(),
            year=pl.col("year").first(),
        )  # fmt: skip
        per_year = [g.filter(pl.col("year") == y).height for y in sc.SELECT_YEARS]
        cand_year = [df.filter(pl.col("year") == y).height for y in sc.SELECT_YEARS]
        big = g.sort("n", descending=True).row(0, named=True)
        span = (big["t1"] - big["t0"]).total_seconds() / 3600
        lines.append(
            f"| {label} | {' / '.join(map(str, per_year))} | {' / '.join(map(str, cand_year))}"
            f" | {big['n']}, {big['syms']}, {span:.1f} h, {big['t0']:%Y-%m-%d %H:%M} UTC"
            f" | {g['n'].median():.0f} | {g.filter(pl.col('n') > 10).height} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "score":
        only = set(sc10._opt(sys.argv, "--only", "").split(",")) - {""}
        run(only or None, "--no-seeds" not in sys.argv)
    elif cmd == "cascades":
        print(cascade_stats())
    elif cmd == "table":
        print(sc10.table([sc10.BASE_VARIANT, *[variant(r) for r in rules().values()]]))
    else:
        raise SystemExit(__doc__)
