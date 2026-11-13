"""M6a: the finalists and the pick record (docs/plans/ma-rebound.md, "M6 design"), idempotent and
resumable.

    uv run --group research python m6_finalists.py            # score what is ready, write the record
    uv run --group research python m6_finalists.py --report   # only rewrite results/M6_pick.md

Models (`train.py run <model>`, 5 seeds, folds 2022-2026): `base5`, `breadth5`, `otherma5`,
`short5`, `combo5`. Rules, each on every model's seed-mean scores (`m6-<model>-<rule>`) and seed
by seed (`m6-<model>-<rule>-seed<i>`, via `score_seeds`): the base rule, percentile >= 0.9,
1h + 4h MAs only, 20 slots, 1d MAs only. 25 rows plus 125 seed rows, ledger stage `M6`.

**Eligibility** (fixed before M6 ran): only `m6-base5-base` can be the pick, and only if it
qualifies and is not fragile on 2022-24. Every other row is a comparison that can be neither
picked nor confirmed.

Before a model is scored, its tables must cover every limit candidate decided in 2022-2026. A
card is skipped when the ledger already has its row with the same source (the source carries
the training run id), so an interrupted run resumes and a retrained model gets new rows. Card
texts are cached in `cache/m6/cards/`. The holdout stays sealed: everything goes through
`scorecard` / `score_seeds` (never `reveal`), and only 2022-24 numbers are read from the ledger.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import book as bk
import numpy as np
import polars as pl
import train as tr
from common import CACHE, RESULTS
from scorecard import LEDGER, Rule, score_seeds, scorecard, show

STAGE = "M6"
SEEDS = 5
TEXTS = CACHE / "m6" / "cards"
CARDS_FILE = RESULTS / "m6_cards.txt"
REPORT = RESULTS / "M6_pick.md"
BASE_RULE = Rule(family="limit", buffer=0.05, exit="E1", pct=0.80, slots=10)
YEAR_LO, YEAR_HI = 2022, 2026
YEARS = (2022, 2023, 2024)
PICKABLE = "m6-base5-base"


@dataclass(frozen=True)
class Model:
    out: str  # train.py variant and score folder
    what: str
    ref: str | None  # the M4/M5 row of the same model with 3 seeds (base rule)
    ref_out: str | None  # its score folder (run id for the reference row's source)


@dataclass(frozen=True)
class RuleSpec:
    key: str
    what: str
    rule: Rule
    ref: str | None  # the M5 rule row (on the 3-seed base CNN), for the base model only


MODELS: list[Model] = [
    Model("base5", "base CNN", "cnn-base", "base"),
    Model("breadth5", "+ market breadth", "cnn-breadth", "breadth"),
    Model("otherma5", "+ distances to the other MAs", "cnn-otherma", "otherma"),
    Model("short5", "windows 48 x 5m and 64 bars", "cnn-short", "short"),
    Model("combo5", "short windows + breadth + other-MA distances", None, None),
]
RULES: list[RuleSpec] = [
    RuleSpec("base", "base rule (pct >= 0.80, 10 slots)", BASE_RULE, None),
    RuleSpec("pct0.9", "percentile >= 0.9", Rule(pct=0.9), "rule-pct0.9"),
    RuleSpec("tf1h4h", "MAs: 1h + 4h only", Rule(tfs=("1h", "4h"), pct=0.80), "rule-tf1h4h"),
    RuleSpec("slots20", "20 slots", Rule(pct=0.80, slots=20), "rule-slots20"),
    RuleSpec("tf1d", "MAs: 1d only", Rule(tfs=("1d",), pct=0.80), "rule-tf1d"),
]


@dataclass(frozen=True)
class Item:
    key: str  # ledger name (seed rows add -seed<i>)
    model: Model
    spec: RuleSpec


def items() -> list[Item]:
    return [Item(f"m6-{m.out}-{r.key}", m, r) for m in MODELS for r in RULES]


def run_id(out: str) -> str:
    return str(json.loads((tr.SCORES / out / "run.json").read_text())["run_id"])


def table(out: str, which: str) -> pl.DataFrame:
    return pl.read_parquet(tr.SCORES / out / f"{which}.parquet").select("cid", "score")


def ready(out: str) -> bool:
    d = tr.SCORES / out
    if not (d / "run.json").exists():
        return False
    meta = json.loads((d / "run.json").read_text())
    full = meta["years"] == list(tr.YEARS) and meta["max_train"] is None
    return (
        full
        and json.loads(meta["config"])["seeds"] == SEEDS
        and all(
            (d / f"{w}.parquet").exists() for w in ("mean", *[f"seed{i}" for i in range(SEEDS)])
        )
    )


_covered: set[str] = set()


def check_coverage(out: str) -> None:
    """Every limit candidate decided in 2022-2026 has a finite score in every table."""
    if out in _covered:
        return
    c = bk.cands()
    year = pl.from_epoch(pl.Series(c.decision_t), time_unit="us").dt.year()
    want = pl.DataFrame({"cid": c.cid, "family": c.family, "year": year}).filter(
        (pl.col("family") == "limit") & pl.col("year").is_between(YEAR_LO, YEAR_HI)
    )
    for which in ("mean", *[f"seed{i}" for i in range(SEEDS)]):
        t = table(out, which)
        missing = want.join(t, on="cid", how="anti").height
        assert missing == 0, f"{out}/{which}: {missing} of {want.height} candidates have no score"
        assert t["score"].is_finite().all(), f"{out}/{which}: non-finite scores"
    print(
        f"  coverage ok: {out} scores all {want.height} limit candidates of 2022-2026", flush=True
    )
    _covered.add(out)


def ledger() -> pl.DataFrame:
    return pl.read_csv(LEDGER, infer_schema_length=None) if LEDGER.exists() else pl.DataFrame()


def has_row(name: str, source: str) -> bool:
    lg = ledger()
    return (
        lg.height > 0
        and lg.filter((pl.col("name") == name) & (pl.col("source") == source)).height > 0
    )


def sources(it: Item) -> tuple[str, str]:
    """(seed-mean source, per-seed source stem); both carry the training run id."""
    rid = run_id(it.model.out)
    return (
        f"M6 {it.model.out} rule {it.spec.key}, seed mean of {SEEDS}, run {rid}",
        f"M6 {it.model.out} rule {it.spec.key}, run {rid}",
    )


def score_item(it: Item) -> bool:
    """Score one item unless its rows exist. Returns True if work was done."""
    if not ready(it.model.out):
        print(f"{it.key}: scores not ready, skipped", flush=True)
        return False
    mean_src, seed_src = sources(it)
    done = has_row(it.key, mean_src) and all(
        has_row(f"{it.key}-seed{i}", f"{seed_src}, seed {i}") for i in range(SEEDS)
    )
    if done and (TEXTS / f"{it.key}.txt").exists():
        print(f"{it.key}: done, skipped", flush=True)
        return False
    check_coverage(it.model.out)
    t0 = time.perf_counter()
    rule = it.spec.rule
    card = scorecard(it.key, rule, table(it.model.out, "mean"), stage=STAGE, source=mean_src)
    seeds = score_seeds(
        it.key,
        rule,
        [table(it.model.out, f"seed{i}") for i in range(SEEDS)],
        stage=STAGE,
        source=seed_src,
    )
    spread = (
        f"{it.key} per-seed rank values: "
        + ", ".join(f"seed {i} {v * 100:+.2f}%" for i, v in enumerate(seeds.rank_values))
        + f"; mean {seeds.mean_rank * 100:+.2f}%; spread (max - min) {seeds.spread * 100:.2f} pp"
    )
    TEXTS.mkdir(parents=True, exist_ok=True)
    (TEXTS / f"{it.key}.txt").write_text(
        "\n\n".join([show(card), *[show(c) for c in seeds.cards], spread])
    )
    print(
        f"{it.key}: scored in {time.perf_counter() - t0:.0f}s, rank {card.rank_value * 100:+.1f}%",
        flush=True,
    )
    return True


# --- the pick record -------------------------------------------------------------------------


def row_of(name: str, source_prefix: str | None = None) -> dict[str, object] | None:
    lg = ledger()
    if lg.height == 0:
        return None
    r = lg.filter(pl.col("name") == name)
    if source_prefix:
        r = r.filter(pl.col("source").str.contains(source_prefix, literal=True))
    return None if r.height == 0 else r.sort("timestamp").row(-1, named=True)


def f(r: dict[str, object], k: str) -> float:
    return float(r[k])  # type: ignore[arg-type]


def num(v: float) -> str:
    return f"{v:+.1f}" if v == v else "-"


def pc(v: float) -> str:
    return f"{v * 100:+.1f}%" if v == v else "-"


def yn(v: object) -> str:
    return "yes" if bool(v) else "no"


def seed_rows(name: str, stem: str, n: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for i in range(n):
        r = row_of(f"{name}-seed{i}", f"{stem}, seed {i}")
        if r is None:
            return []
        out.append(r)
    return out


def ref_row(it: Item) -> tuple[str, dict[str, object]] | None:
    """The M4/M5 row this M6 row corresponds to (3 seeds), if any."""
    base_id = run_id("base")
    if it.spec.key == "base" and it.model.ref is not None and it.model.ref_out is not None:
        rid = run_id(it.model.ref_out)
        r = row_of(it.model.ref, f"seed mean of 3, run {rid}")
        return None if r is None else (f"{it.model.ref} ({r['stage']}, 3 seeds)", r)
    if it.model.out == "base5" and it.spec.ref is not None:
        r = row_of(it.spec.ref, f"seed mean of 3, run {base_id}")
        return None if r is None else (f"{it.spec.ref} ({r['stage']}, 3-seed base CNN)", r)
    return None


def table_line(name: str, what: str, r: dict[str, object]) -> str:
    return (
        f"| {name} | {what} | "
        + " / ".join(f"{f(r, f'trades_{y}'):.0f}" for y in YEARS)
        + " | "
        + " / ".join(f"{f(r, f'bp_{y}'):+.0f}" for y in YEARS)
        + " | "
        + " / ".join(pc(f(r, f"ret_{y}")) for y in YEARS)
        + f" | {num(f(r, 'stress_bp'))} | {num(f(r, 'placebo_p95_bp'))} | "
        f"{yn(r['fragile'])} | {yn(r['qualifies'])} | {pc(f(r, 'rank_value'))} |"
    )


REVEAL_CONTENT = [
    'From the plan\'s Decisions entry "M6 design" (*Reveal content*), opened once for every ledger row:',
    "",
    "- per row: 2025, 2026 and 2025-26 trades, bp per trade, book return and drawdown;",
    "- the pick's (or, with no pick, the top-ranked finalist's) day-bootstrap CI and its comparison",
    "  with *every candidate*;",
    "- the Spearman correlation between the 2022-24 rank value and the 2025-26 book return over",
    "  non-seed rows;",
    "- a deflated Sharpe ratio over all non-seed ledger rows;",
    "- and, as context, M2's per-timeframe and study-vs-placebo tables on 2025-26.",
    "",
    '**Confirmed** (plan, "M6 pick and reveal") applies to the pick only: on 2025-26 the',
    "day-bootstrap 95% CI of the mean net bp per trade is above 0, the book return is positive, and",
    "it beats *every candidate* under the same rule. With no pick, nothing can be called confirmed.",
]


def report() -> None:
    lines_tab = [
        "| row | model / rule | trades 2022 / 23 / 24 | bp per trade 2022 / 23 / 24 | book return 2022 / 23 / 24 | stress bp | placebo p95 bp | fragile | qualifies | rank value |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    seed_tab = [
        "| model | "
        + " | ".join(f"seed {i}" for i in range(SEEDS))
        + " | seed mean of ranks | spread (max - min) | 5-seed-mean row | mean row inside seed range | seeds qualifying | seeds fragile |",
        "|---|" + "---|" * (SEEDS + 6),
    ]
    rule_seed_tab = [
        "| row | "
        + " | ".join(f"seed {i}" for i in range(SEEDS))
        + " | spread | seeds qualifying | seeds fragile |",
        "|---|" + "---|" * (SEEDS + 3),
    ]
    missing: list[str] = []
    rows: dict[str, dict[str, object]] = {}
    for it in items():
        if not ready(it.model.out):
            missing.append(it.key)
            continue
        ms, ss = sources(it)
        r = row_of(it.key, ms)
        sr = seed_rows(it.key, ss, SEEDS)
        if r is None or not sr:
            missing.append(it.key)
            continue
        rows[it.key] = r
        lines_tab.append(table_line(f"**{it.key}**", f"{it.model.what} / {it.spec.what}", r))
        ref = ref_row(it)
        if ref is not None:
            lines_tab.append(table_line(f"ref: {ref[0]}", "for reference, not an M6 row", ref[1]))
        ranks = [f(x, "rank_value") for x in sr]
        nq = sum(bool(x["qualifies"]) for x in sr)
        nf = sum(bool(x["fragile"]) for x in sr)
        mean_rank = f(r, "rank_value")
        if it.spec.key == "base":
            inside = min(ranks) <= mean_rank <= max(ranks)
            seed_tab.append(
                f"| {it.model.out} | "
                + " | ".join(pc(v) for v in ranks)
                + f" | {pc(float(np.mean(ranks)))} | {(max(ranks) - min(ranks)) * 100:.1f} pp | {pc(mean_rank)} | "
                f"{yn(inside)} | {nq} of {SEEDS} | {nf} of {SEEDS} |"
            )
        rule_seed_tab.append(
            f"| {it.key} | "
            + " | ".join(pc(v) for v in ranks)
            + f" | {(max(ranks) - min(ranks)) * 100:.1f} pp | {nq} of {SEEDS} | {nf} of {SEEDS} |"
        )

    texts = [t.read_text() for t in sorted(TEXTS.glob("*.txt"))] if TEXTS.exists() else []
    order = {it.key: k for k, it in enumerate(items())}
    texts.sort(key=lambda t: order.get(t.split()[0], len(order)))
    CARDS_FILE.write_text("\n\n".join(["# M6 finalists (2022-24)", *texts]) + "\n")

    pick = rows.get(PICKABLE)
    if pick is None:
        pick_lines = [
            f"Not decided yet: `{PICKABLE}` has not been scored ({len(missing)} rows missing).",
        ]
    else:
        q, fr = bool(pick["qualifies"]), bool(pick["fragile"])
        exbest = " / ".join(f"{f(pick, f'exbest_{y}'):+.0f}" for y in YEARS)
        reasons = [
            f"- Verdict criteria (2022-24): {pick['reasons']}.",
            f"- Qualifies: {yn(q)}.",
            f"- Fragile: {yn(fr)} (mean bp per year without that year's best trade: {exbest}).",
        ]
        if q and not fr:
            pick_lines = [
                f"**The pick is {PICKABLE}.** It qualifies and is not fragile on 2022-24, which is",
                "the plan's rule for the only eligible row.",
                "",
                *reasons,
            ]
        else:
            why = [w for w, bad in (("it does not qualify", not q), ("it is fragile", fr)) if bad]
            pick_lines = [
                f"**There is no pick.** The only eligible row, `{PICKABLE}`, can be the pick only if it",
                f"qualifies and is not fragile on 2022-24; {' and '.join(why)}. By the plan,",
                "nothing can be called confirmed at the reveal. The other 24 rows are comparisons and",
                "cannot be picked whatever their numbers.",
                "",
                *reasons,
            ]

    others_q = [k for k, r in rows.items() if k != PICKABLE and bool(r["qualifies"])]
    others_ok = [
        k
        for k, r in rows.items()
        if k != PICKABLE and bool(r["qualifies"]) and not bool(r["fragile"])
    ]

    def top(keys: list[str]) -> str:
        best = max(keys, key=lambda k: (f(rows[k], "rank_value"), f(rows[k], "worst_ret")))
        return f"`{best}` ({pc(f(rows[best], 'rank_value'))})"

    top_lines = (
        [
            'For the reveal\'s "top-ranked finalist" (used only when there is no pick), the 2022-24',
            "rank values give: the protocol's only finalist is `base5`, and `m6-base5-base` is its",
            f"base-rule row; the top-ranked of the five models' base-rule rows is "
            f"{top([k for k in rows if k.endswith('-base')])}; the top-ranked of all 25 rows is "
            f"{top(list(rows))}. Which of these the reveal reports is the master's call, fixed before",
            "the holdout is opened.",
        ]
        if len(rows) == len(items())
        else []
    )
    body = [
        "# M6 pick record (written before the holdout is opened)",
        "",
        "Everything here is 2022-24, the selection period; the book is continuous from 2022-01-01.",
        "2025-01-01 onward is the sealed holdout and no number of it appears here. The design was",
        'fixed in the plan\'s Decisions entry "M6 design" before M6 ran.',
        "",
        "- **Models** (5 seeds each, 0-4, folds 2022-2026, otherwise identical to their registered",
        "  3-seed variant): `base5` (the base CNN), `breadth5`, `otherma5`, `short5`, and `combo5`",
        "  (fine 48, coarse 64, groups coin + btc + breadth + otherma).",
        "- **Rules** on each model's seed-mean scores (row `m6-<model>-<rule>`) and seed by seed",
        "  (`m6-<model>-<rule>-seed<i>` in the ledger): `base` (limit at L(1 + 0.05u), exit E1,",
        "  percentile >= 0.80, 10 slots), `pct0.9`, `tf1h4h` (1h + 4h MAs only), `slots20`, `tf1d`",
        "  (1d MAs only); every non-base rule changes one thing from the base rule.",
        f"- **Eligibility:** only `{PICKABLE}` can be the pick, and only if it qualifies and is not",
        "  fragile on 2022-24. Every other row is a comparison that can be neither picked nor confirmed.",
        "",
        "All cards are in `results/m6_cards.txt`; every row, seed rows included, is in",
        "`results/ledger.csv` (stage M6), with the training run id in its `source`.",
        "",
        "## All 25 rows on 2022-24",
        "",
        "M6 rows in bold. Under each, where one exists, the M4/M5 row it corresponds to (3 seeds):",
        "the same model under the base rule, or for `base5` the same rule on the 3-seed base CNN.",
        "`combo5` and the non-base rules on the other models have no earlier counterpart.",
        "",
        *lines_tab,
        "",
        "Qualifies: at least 100 trades, mean bp > 0 and book return > 0 in each year, stress bp > 0,",
        "and mean bp above the placebo's 95th percentile. Fragile: some year's mean bp turns negative",
        "without that year's best trade. Rank value: the mean of the three yearly book returns.",
        "",
        "## Seeds",
        "",
        "Each model under the base rule, seed by seed. *Mean row inside seed range*: whether the rank",
        "value of the 5-seed-mean score lies between the lowest and the highest per-seed rank value.",
        "",
        *seed_tab,
        "",
        "Every row, seed by seed:",
        "",
        *rule_seed_tab,
        "",
        "## Pick",
        "",
        *pick_lines,
        "",
        f"Comparison rows that qualify: {', '.join(others_q) or 'none'}; that qualify and are not",
        f"fragile: {', '.join(others_ok) or 'none'}. They are recorded for the reveal's rank correlation",
        "and deflated Sharpe ratio only.",
        "",
        *top_lines,
        "",
        f"Rows not yet scored: {', '.join(missing) or 'none'}.",
        "",
        "## What will be opened",
        "",
        *REVEAL_CONTENT,
        "",
    ]
    REPORT.write_text("\n".join(body))
    print(f"wrote {REPORT} and {CARDS_FILE}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--report", action="store_true", help="only rewrite the pick record")
    a = p.parse_args()
    if not a.report:
        for it in items():
            score_item(it)
    report()


if __name__ == "__main__":
    main()
