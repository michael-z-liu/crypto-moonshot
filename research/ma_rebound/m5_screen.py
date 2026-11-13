"""M5: the variant screen (docs/plans/ma-rebound.md, "M5 screen"), idempotent and resumable.

    uv run --group research python m5_screen.py            # score what is ready, write the report
    uv run --group research python m5_screen.py --rules    # only the rules on the base CNN scores
    uv run --group research python m5_screen.py --report   # only rewrite results/M5.md

A. Training variants (`train.py run <variant>`, 3 seeds, folds 2022-2026): each is scored on its
   seed mean (`cnn-<variant>`) and seed by seed (`cnn-<variant>-seed<i>`) under the base rule
   (`touch` under the touch-market family). Before scoring, the score tables must cover every
   candidate of the variant's family decided in 2022-2026.
B. Rules on the base CNN's seed-mean scores, one change from the base rule each
   (`rule-<label>`), plus per-seed rows (`rule-<label>-seed<i>`) on the base CNN's seed tables.

A card is skipped when the ledger already has its row with the same source (the source carries
the training run id), so an interrupted run resumes and a retrained model gets new rows. Card
texts are cached in `cache/m5/cards/`. The holdout stays sealed: everything goes through
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
from common import CACHE, RESULTS, TIMEFRAMES
from scorecard import LEDGER, Rule, score_seeds, scorecard, show

STAGE = "M5"
SEEDS = 3
TEXTS = CACHE / "m5" / "cards"
CARDS_FILE = RESULTS / "m5_cards.txt"
REPORT = RESULTS / "M5.md"
SUMMARY = RESULTS / "m5_summary_text.md"  # prose written after reading the tables
BASE_RULE = Rule(family="limit", buffer=0.05, exit="E1", pct=0.80, slots=10)
TFS = tuple(TIMEFRAMES)
YEAR_LO, YEAR_HI = 2022, 2026


@dataclass(frozen=True)
class Item:
    key: str  # ledger name (seed rows add -seed<i>)
    changed: str
    rule: Rule
    variant: str | None  # trained variant (scores in cache/m4/scores/<variant>), else None
    kind: str  # "train" | "rule"


TRAINED: list[tuple[str, str]] = [
    ("coin_only", "inputs: the coin's own windows only (no BTC)"),
    ("breadth", "inputs: + market breadth"),
    ("otherma", "inputs: + distances to the other MAs"),
    ("short", "windows: 48 x 5m and 64 bars"),
    ("long", "windows: 288 x 5m and 256 bars"),
    ("path", "target: 32-step path quantiles (score = last median)"),
    ("bracket", "target: E3 take-profit probability"),
    ("tfgroup", "one model per timeframe group (1h+4h, 1d, 1w+1M)"),
    ("touch", "touch-market candidates (their own E1)"),
]
RULES: list[tuple[str, str, Rule]] = [
    *[(f"pct{p:g}", f"threshold: percentile >= {p:g}", Rule(pct=p)) for p in (0.5, 0.7, 0.9, 0.95)],
    ("buffer0", "buffer 0", Rule(buffer=0.0, pct=0.80)),
    ("buffer0.15", "buffer 0.15u", Rule(buffer=0.15, pct=0.80)),
    *[(f"exit{e}", f"exit {e}", Rule(exit=e, pct=0.80)) for e in ("E2", "E3", "E4", "E5", "E6")],
    ("tf1h4h", "MAs: 1h + 4h only", Rule(tfs=("1h", "4h"), pct=0.80)),
    ("tf1d", "MAs: 1d only", Rule(tfs=("1d",), pct=0.80)),
    ("tf1w1M", "MAs: 1w + 1M only", Rule(tfs=("1w", "1M"), pct=0.80)),
    ("n200", "MAs: the 200-length ones only", Rule(lengths=(200,), pct=0.80)),
    ("majors", "coins: majors only", Rule(majors=True, pct=0.80)),
    ("slots5", "5 slots", Rule(pct=0.80, slots=5)),
    ("slots20", "20 slots", Rule(pct=0.80, slots=20)),
]


def items() -> list[Item]:
    touch_rule = Rule(family="touch", exit="E1", pct=0.80, slots=10)
    out = [
        Item(f"cnn-{v}", why, touch_rule if v == "touch" else BASE_RULE, v, "train")
        for v, why in TRAINED
    ]
    out += [Item(f"rule-{k}", why, r, "base", "rule") for k, why, r in RULES]
    return out


def run_id(out: str) -> str:
    return str(json.loads((tr.SCORES / out / "run.json").read_text())["run_id"])


def table(out: str, which: str) -> pl.DataFrame:
    return pl.read_parquet(tr.SCORES / out / f"{which}.parquet").select("cid", "score")


def ready(out: str) -> bool:
    d = tr.SCORES / out
    return (d / "run.json").exists() and all(
        (d / f"{w}.parquet").exists() for w in ("mean", "seed0", "seed1", "seed2")
    )


def check_coverage(out: str, family: str) -> None:
    """Every candidate of the family decided in 2022-2026 has a score in every table."""
    c = bk.cands()
    year = pl.from_epoch(pl.Series(c.decision_t), time_unit="us").dt.year()
    want = pl.DataFrame({"cid": c.cid, "family": c.family, "year": year}).filter(
        (pl.col("family") == family) & pl.col("year").is_between(YEAR_LO, YEAR_HI)
    )
    for which in ("mean", *[f"seed{i}" for i in range(SEEDS)]):
        t = table(out, which)
        missing = want.join(t, on="cid", how="anti").height
        assert missing == 0, f"{out}/{which}: {missing} of {want.height} candidates have no score"
        assert t["score"].is_finite().all(), f"{out}/{which}: non-finite scores"
    print(
        f"  coverage ok: {out} scores all {want.height} {family} candidates of 2022-2026",
        flush=True,
    )


def ledger() -> pl.DataFrame:
    return pl.read_csv(LEDGER, infer_schema_length=None) if LEDGER.exists() else pl.DataFrame()


def has_row(name: str, source: str) -> bool:
    lg = ledger()
    return (
        lg.height > 0
        and lg.filter((pl.col("name") == name) & (pl.col("source") == source)).height > 0
    )


def sources(it: Item) -> tuple[str, str, str]:
    """(mean source, per-seed source stem, run id)."""
    assert it.variant is not None
    rid = run_id(it.variant)
    if it.kind == "train":
        return (
            f"M5 {it.variant}, seed mean of {SEEDS}, run {rid}",
            f"M5 {it.variant}, run {rid}",
            rid,
        )
    lab = it.key.removeprefix("rule-")
    return (
        f"M5 rule {lab} on the base CNN seed mean of {SEEDS}, run {rid}",
        f"M5 rule {lab} on the base CNN, run {rid}",
        rid,
    )


def score_item(it: Item) -> bool:
    """Score one item unless its rows exist. Returns True if work was done."""
    assert it.variant is not None
    if not ready(it.variant):
        print(f"{it.key}: scores not ready, skipped", flush=True)
        return False
    mean_src, seed_src, _ = sources(it)
    done = has_row(it.key, mean_src) and all(
        has_row(f"{it.key}-seed{i}", f"{seed_src}, seed {i}") for i in range(SEEDS)
    )
    if done and (TEXTS / f"{it.key}.txt").exists():
        print(f"{it.key}: done, skipped", flush=True)
        return False
    if it.kind == "train":
        check_coverage(it.variant, it.rule.family)
    t0 = time.perf_counter()
    card = scorecard(it.key, it.rule, table(it.variant, "mean"), stage=STAGE, source=mean_src)
    seeds = score_seeds(
        it.key,
        it.rule,
        [table(it.variant, f"seed{i}") for i in range(SEEDS)],
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


# --- report ----------------------------------------------------------------------------------


def row_of(name: str, source_prefix: str | None = None) -> dict[str, object] | None:
    lg = ledger()
    if lg.height == 0:
        return None
    r = lg.filter(pl.col("name") == name)
    if source_prefix:
        r = r.filter(pl.col("source").str.contains(source_prefix, literal=True))
    return None if r.height == 0 else r.sort("timestamp").row(-1, named=True)


def tf_cells(s: object) -> dict[str, tuple[int, float]]:
    out: dict[str, tuple[int, float]] = {}
    for part in str(s).split(";"):
        part = part.strip()
        if not part:
            continue
        tf, tb = part.split(" ")
        n, bp = tb.split("/")
        out[tf] = (int(n), float(bp))
    return out


def num(v: object) -> str:
    x = float(v)  # type: ignore[arg-type]
    return f"{x:+.1f}" if x == x else "-"


def pc(v: float) -> str:
    return f"{v * 100:+.1f}%"


def seed_ranks(name: str, source_stem: str) -> list[float]:
    out: list[float] = []
    for i in range(SEEDS):
        r = row_of(f"{name}-seed{i}", f"{source_stem}, seed {i}")
        if r is None:
            return []
        out.append(float(r["rank_value"]))  # type: ignore[arg-type]
    return out


def report() -> None:
    base_id = run_id("base")
    base = row_of("cnn-base", base_id)
    base_seeds = seed_ranks("cnn-base", f"M4 base CNN, run {base_id}")
    assert base is not None, "M4 base row missing"
    assert base_seeds, "M4 base seed rows missing"
    every = row_of("every-L0.05-E1")
    base_rank = float(base["rank_value"])  # type: ignore[arg-type]
    spread = max(base_seeds) - min(base_seeds)
    bar = base_rank + spread
    lines_tab = [
        "| name | what changed | trades/yr | bp 2022 | bp 2023 | bp 2024 | return 2022 | return 2023 | return 2024 | stress bp | placebo p95 | fragile | qualifies | rank value | vs base | beats baseline |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    seedlines = [
        "| name | seed 0 | seed 1 | seed 2 | mean | spread (max-min) | gain over base in every seed |",
        "|---|---|---|---|---|---|---|",
    ]
    tflines = [
        "| name | " + " | ".join(f"{t} trades / bp" for t in TFS) + " |",
        "|---|" + "---|" * len(TFS),
    ]
    quals: list[str] = []
    beats: list[str] = []
    near: list[str] = []
    between: list[str] = []
    below: list[str] = []
    missing: list[str] = []

    def add(
        name: str, changed: str, r: dict[str, object], stem: str | None, trained_or_rule: bool
    ) -> None:
        rank = float(r["rank_value"])  # type: ignore[arg-type]
        q = bool(r["qualifies"])
        fr = bool(r["fragile"])
        beat = q and not fr and rank > bar
        tr_year = np.mean([float(r[f"trades_{y}"]) for y in (2022, 2023, 2024)])  # type: ignore[arg-type]
        lines_tab.append(
            f"| {name} | {changed} | {tr_year:.0f} | "
            + " | ".join(f"{float(r[f'bp_{y}']):+.0f}" for y in (2022, 2023, 2024))  # type: ignore[arg-type]
            + " | "
            + " | ".join(pc(float(r[f"ret_{y}"])) for y in (2022, 2023, 2024))  # type: ignore[arg-type]
            + f" | {num(r['stress_bp'])} | {num(r['placebo_p95_bp'])} | "  # type: ignore[arg-type]
            f"{'yes' if fr else 'no'} | {'yes' if q else 'no'} | {pc(rank)} | {(rank - base_rank) * 100:+.1f} pp | "
            f"{'YES' if beat else 'no'} |"
        )
        cells = tf_cells(r["per_tf_2224"])
        tflines.append(
            f"| {name} | "
            + " | ".join(f"{cells[t][0]} / {cells[t][1]:+.0f}" if t in cells else "-" for t in TFS)
            + " |"
        )
        if not trained_or_rule:
            return
        if q:
            quals.append(name)
        if beat:
            beats.append(name)
        elif rank > bar:
            near.append(
                f"{name} (rank {pc(rank)}; qualifies {'yes' if q else 'no'}, fragile {'yes' if fr else 'no'})"
            )
        elif rank > base_rank:
            between.append(f"{name} ({(rank - base_rank) * 100:+.1f} pp)")
        else:
            below.append(f"{name} ({(rank - base_rank) * 100:+.1f} pp)")
        sr = seed_ranks(name, stem) if stem else []
        if sr:
            gain = "yes" if all(a > b for a, b in zip(sr, base_seeds, strict=True)) else "no"
            seedlines.append(
                f"| {name} | "
                + " | ".join(pc(v) for v in sr)
                + f" | {pc(float(np.mean(sr)))} | {(max(sr) - min(sr)) * 100:.1f} pp | {gain} |"
            )

    assert every is not None
    add("every candidate", "no model, base rule", every, None, False)
    add("cnn-base", "base CNN (M4), seed mean", base, None, False)
    seedlines.append(
        "| cnn-base | "
        + " | ".join(pc(v) for v in base_seeds)
        + f" | {pc(float(np.mean(base_seeds)))} | {spread * 100:.1f} pp | - |"
    )
    for it in items():
        assert it.variant is not None
        if not ready(it.variant):
            missing.append(it.key)
            continue
        ms, ss, _ = sources(it)
        r = row_of(it.key, ms)
        if r is None:
            missing.append(it.key)
            continue
        add(it.key, it.changed, r, ss, True)

    texts = [t.read_text() for t in sorted(TEXTS.glob("*.txt"))] if TEXTS.exists() else []
    order = {it.key: k for k, it in enumerate(items())}
    texts.sort(key=lambda t: order.get(t.split()[0].removesuffix("-seed0"), 0))
    m4 = (RESULTS / "m4_cards.txt").read_text() if (RESULTS / "m4_cards.txt").exists() else ""
    CARDS_FILE.write_text(
        "\n\n".join(["# Reference (M4)", m4.strip(), "# M5 variants", *texts]) + "\n"
    )

    summary = SUMMARY.read_text().strip() if SUMMARY.exists() else "(summary not written yet)"
    body = [
        "# M5: variant screen",
        "",
        "Everything here is 2022-24 (the book is continuous from 2022-01-01; 2025-26 stays sealed).",
        "Each variant changes one thing from the base: limit entry at buffer 0.05u, exit E1, score",
        "percentile >= 0.80, 10 slots. Trained variants use seeds 0-2 over folds 2022-2026; their row is",
        "the seed mean, and the per-seed rows are in the ledger. Rules (`rule-*`) are applied to the base",
        "CNN's seed-mean scores (no training). All cards are in `results/m5_cards.txt`; every row is in",
        "`results/ledger.csv` (stage M5).",
        "",
        "The pre-registered bar to beat the baseline: qualifies, not fragile, and rank value above the",
        f"base CNN's {pc(base_rank)} plus its seed spread {spread * 100:.1f} pp, i.e. above {pc(bar)}.",
        "",
        "## All variants",
        "",
        *lines_tab,
        "",
        "## Per-seed rank values",
        "",
        "Rank value = mean of the three yearly book returns. A rule's seed rows score the base CNN's three",
        "seed tables under the rule, so 'gain in every seed' compares each seed with the same seed's base value.",
        "",
        *seedlines,
        "",
        "## Where trades and bp come from (2022-24, by timeframe of the MA)",
        "",
        *tflines,
        "",
        "## Mechanical verdicts",
        "",
        f"- Qualify: {', '.join(quals) or 'none'}.",
        f"- Beat the baseline (qualify, not fragile, rank above {pc(bar)}): {', '.join(beats) or 'none'}.",
        f"- Rank above the bar but not beating the baseline: {'; '.join(near) or 'none'}.",
        f"- Rank above the base CNN's but below the bar: {', '.join(between) or 'none'}.",
        f"- Rank at or below the base CNN's: {', '.join(below) or 'none'}.",
        f"- Not yet scored: {', '.join(missing) or 'none'}.",
        "",
        "## Summary",
        "",
        summary,
        "",
    ]
    REPORT.write_text("\n".join(body))
    print(f"wrote {REPORT} and {CARDS_FILE}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--rules", action="store_true", help="only part B")
    p.add_argument("--train", action="store_true", help="only part A")
    p.add_argument("--report", action="store_true", help="only rewrite the report")
    a = p.parse_args()
    if not a.report:
        for it in items():
            if (a.rules and it.kind != "rule") or (a.train and it.kind != "train"):
                continue
            score_item(it)
    report()


if __name__ == "__main__":
    main()
