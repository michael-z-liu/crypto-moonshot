"""Issue #10 scoring (docs/plans/cnn-improvements.md, Evaluation protocol).

    cd research/volume_spike
    export MOONSHOT_UNIVERSE=all
    uv run --group research python i10_score.py score base3 [--heads h1,h2] [--stage M1]
    uv run --group research python i10_score.py seeds base3 [--heads h1] [--stage M1]
    uv run --group research python i10_score.py band [M1:base3:h1]
    uv run --group research python i10_score.py table M1:base3:h1 ...

The book is the robustness scorecard (`scorecard.py`) with S5's `a_1h` rule
(`s5_confirm.rules()["a_1h"]`: pct >= 0.98, `CashCapped(5)`, highest score first, 1h hold, fee
0.1% + slippage 0.05% + latency 0.05% per side). Everything goes to `results/issue10/` (via
`scorecard.configure`), never to the robustness ledger:

- `score`: the seed-mean head of `cache_all/wf_i10/<name>.npz` as a ledger trial
  (`ledger.csv`, 20 placebos), variant `<stage>:<name>:<head>`, plus a text card in `cards/`;
  then the same head seed by seed (see `seeds`).
- `seeds`: each seed's own percentiles (`pct_seed*`), no placebos, not a ledger trial; the
  2022-24 rows (trades and bp per year, mean, 0.6% cost, one candle late, fragile) are upserted
  into `seeds.csv` by (variant, seed).
- `band`: the noise band, max - min of the per-seed 2022-24 mean bp (default base3's h1).
- `table`: a markdown table (2022-24) of ledger variants and their seeds.

2025-26 is computed by the scorecard but written only to `results/issue10/sealed/` (gitignored);
nothing here reads it.
"""

from __future__ import annotations

import fcntl
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import i10_train
import numpy as np
import polars as pl
import s5_confirm
import scorecard as sc

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "issue10"
sc.configure(OUT)
SEEDS_CSV = OUT / "seeds.csv"
CARDS = OUT / "cards"
WORKERS = 8
RULE_NAME = "a_1h"
PCT_KEY = {"h1": "pct", "h2": "pct_h2", "h2ra": "pct_h2ra"}
SEED_KEY = {"h1": "pct_seed", "h2": "pct_seed_h2", "h2ra": "pct_seed_h2ra"}
BASE_VARIANT = "M1:base3:h1"
ScoreFn = Callable[[np.ndarray], np.ndarray]
SignalCols = Callable[[np.ndarray], dict[str, np.ndarray]]


def rule_of(name: str = RULE_NAME) -> sc.Rule:
    return s5_confirm.rules()[name][1]


def npz(name: str) -> Path:
    return i10_train.WF_I10 / f"{name}.npz"


def scores(name: str, head: str = "h1") -> np.ndarray:
    with np.load(npz(name)) as f:
        return np.asarray(f[PCT_KEY[head]], dtype=np.float64)


def seed_scores(name: str, head: str = "h1") -> np.ndarray:
    with np.load(npz(name)) as f:
        return np.asarray(f[SEED_KEY[head]], dtype=np.float64)


def source_of(name: str, head: str, seed: int | None = None) -> str:
    """The score source text (part of the scorecard's config hash)."""
    key = PCT_KEY[head] if seed is None else f"{SEED_KEY[head]}[{seed}]"
    with np.load(npz(name)) as f:
        cfg = str(f["cfg"]) if "cfg" in f else i10_train.I10[name].describe()
    return f"i10_train {name} {cfg} (cache_all/wf_i10/{name}.npz {key})"


def variant_of(stage: str, name: str, head: str, rule_name: str = RULE_NAME) -> str:
    v = f"{stage}:{name}:{head}"
    return v if rule_name == RULE_NAME else f"{v}:{rule_name}"


def card_file(variant: str) -> Path:
    return CARDS / (variant.replace(":", "_") + ".txt")


def score_variant(
    name: str,
    head: str = "h1",
    *,
    stage: str = "M1",
    rule_name: str = RULE_NAME,
    rule: sc.Rule | None = None,
    variant: str | None = None,
    extra: str = "",
    row_filter: np.ndarray | None = None,
    priority: np.ndarray | None = None,
    placebos: int = 20,
    filter_fn: ScoreFn | None = None,
    signal_cols: SignalCols | None = None,
    workers: int = WORKERS,
    head_scores: np.ndarray | None = None,
) -> sc.Card:
    """A ledger trial: the seed-mean `head` of `name` with the rule, 20 placebos, and a card.

    `filter_fn(score)` computes the row filter from the scores being scored (used by M2's age
    rules; it overrides `row_filter`); `signal_cols` is passed to the scorecard (group limits).
    """
    variant = variant or variant_of(stage, name, head, rule_name)
    source = source_of(name, head) + (f"; {extra}" if extra else "")
    score = scores(name, head) if head_scores is None else head_scores
    if filter_fn is not None:
        row_filter = filter_fn(score)
    card = sc.scorecard(
        variant,
        score,
        rule or rule_of(rule_name),
        stage=stage,
        source=source,
        row_filter=row_filter,
        priority=priority,
        placebos=placebos,
        workers=workers,
        signal_cols=signal_cols,
    )
    text = sc.show(card) + f"\n  source: {source}"
    print(text, flush=True)
    CARDS.mkdir(parents=True, exist_ok=True)
    card_file(variant).write_text(text + "\n")
    return card


def seed_row(card: sc.Card, variant: str, seed: int, stage: str) -> dict[str, object]:
    row: dict[str, object] = {"stage": stage, "variant": variant, "seed": seed}
    for y in sc.SELECT_YEARS:
        s = card.years.get(y, {})
        row |= {f"trades_{y}": s.get("trades", 0.0), f"bp_{y}": round(s.get("bp", np.nan), 2)}
    row |= {
        "mean_bp": round(card.mean_bp, 2),
        "worst_bp": round(card.verdict.rank_key[1], 2),
        "cost06_bp": round(sc.mean_bp(card.cost06), 2),
        "late_bp": round(sc.mean_bp(card.late), 2),
        **{f"exbest_{y}": round(card.ex_best[y], 2) for y in sc.SELECT_YEARS},
        "fragile": card.fragile,
        "c1_every_year": card.verdict.criteria["c1_every_year"],
        "c2_cost06": card.verdict.criteria["c2_cost06"],
        "c3_late": card.verdict.criteria["c3_late"],
        "rule": card.rule,
        "source": card.source,
        "config_hash": card.config_hash,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return row


def upsert_seeds(rows: list[dict[str, object]]) -> None:
    """Upsert per-seed rows. Stages run at once, so the read-modify-write holds a file lock."""
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ledger.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        new = pl.DataFrame(rows)
        if SEEDS_CSV.exists():
            old = pl.read_csv(SEEDS_CSV, infer_schema_length=None)
            keys = new.select("variant", "seed")
            old = old.join(keys, on=["variant", "seed"], how="anti")
            if old.height:
                new = pl.concat([old, new], how="diagonal_relaxed")
        tmp = SEEDS_CSV.with_suffix(".tmp")
        new.sort("variant", "seed").write_csv(tmp)
        tmp.replace(SEEDS_CSV)


def score_seeds(
    name: str,
    head: str = "h1",
    *,
    stage: str = "M1",
    rule_name: str = RULE_NAME,
    rule: sc.Rule | None = None,
    variant: str | None = None,
    extra: str = "",
    row_filter: np.ndarray | None = None,
    priority: np.ndarray | None = None,
    filter_fn: ScoreFn | None = None,
    signal_cols: SignalCols | None = None,
    workers: int = 3,
) -> list[dict[str, object]]:
    """Each seed's own percentiles with the rule (no placebos, not a ledger trial)."""
    variant = variant or variant_of(stage, name, head, rule_name)
    rows = []
    for j, s in enumerate(seed_scores(name, head)):
        source = source_of(name, head, j) + (f"; {extra}" if extra else "")
        if filter_fn is not None:
            row_filter = filter_fn(s)
        card = sc.scorecard(
            f"{variant}:seed{j}",
            s,
            rule or rule_of(rule_name),
            stage=stage,
            source=source,
            row_filter=row_filter,
            priority=priority,
            placebos=0,
            workers=workers,
            ledger=False,
            signal_cols=signal_cols,
        )
        rows.append(seed_row(card, variant, j, stage))
        print(
            f"{variant} seed {j}: "
            + " / ".join(f"{card.years.get(y, {}).get('bp', np.nan):+.1f}" for y in sc.SELECT_YEARS)
            + f" bp, mean {card.mean_bp:+.1f}, fragile {card.fragile}",
            flush=True,
        )
    upsert_seeds(rows)
    return rows


def noise_band(variant: str = BASE_VARIANT) -> tuple[float, list[float]]:
    """Max - min of the variant's per-seed 2022-24 mean bp, and the per-seed values."""
    t = pl.read_csv(SEEDS_CSV).filter(pl.col("variant") == variant).sort("seed")
    if t.height == 0:
        raise SystemExit(f"no seed rows for {variant} in {SEEDS_CSV}")
    means = [float(v) for v in t["mean_bp"]]
    return max(means) - min(means), means


def table(variants: list[str]) -> str:
    """Markdown (2022-24): each ledger variant's seed-mean row, then its seeds."""
    led = s5_confirm.latest(pl.read_csv(sc.LEDGER, infer_schema_length=None))
    seeds = pl.read_csv(SEEDS_CSV) if SEEDS_CSV.exists() else None
    out = [
        "| Run | Trades 22 / 23 / 24 | 2022 | 2023 | 2024 | Mean | Worst | 0.6% | Late"
        " | Placebo p95 | Fragile (w/o best 22 / 23 / 24) | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def f(v: object) -> str:
        return "n/a" if v is None else f"{float(v):+.1f}"  # type: ignore[arg-type]

    def line(label: str, r: dict[str, object], p95: str, verdict: str) -> str:
        trades = " / ".join(f"{float(r[f'trades_{y}']):.0f}" for y in sc.SELECT_YEARS)  # type: ignore[arg-type]
        ex = " / ".join(f(r[f"exbest_{y}"]) for y in sc.SELECT_YEARS)
        bp = " | ".join(f(r[f"bp_{y}"]) for y in sc.SELECT_YEARS)
        return (
            f"| {label} | {trades} | {bp} | **{f(r['mean_bp'])}** | {f(r['worst_bp'])}"
            f" | {f(r['cost06_bp'])} | {f(r['late_bp'])} | {p95}"
            f" | {'**yes**' if r['fragile'] else 'no'} ({ex}) | {verdict} |"
        )

    for v in variants:
        rows = led.filter(pl.col("variant") == v)
        if rows.height:
            r = rows.row(0, named=True)
            verdict = "qualifies" if r["qualifies"] else "fails"
            out.append(line(f"{v} (seed mean)", r, f(r["placebo_p95_bp"]), verdict))
        if seeds is not None:
            for r in seeds.filter(pl.col("variant") == v).sort("seed").iter_rows(named=True):
                out.append(line(f"{v} seed {r['seed']}", r, "-", "(not a trial)"))
    return "\n".join(out)


def _opt(argv: list[str], flag: str, default: str) -> str:
    return argv[argv.index(flag) + 1] if flag in argv else default


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    args = sys.argv[2:]
    if cmd in ("score", "seeds"):
        name = args[0]
        heads = _opt(args, "--heads", "h1").split(",")
        stage = _opt(args, "--stage", "M1")
        for head in heads:
            if cmd == "score":
                score_variant(name, head, stage=stage)
            score_seeds(name, head, stage=stage)
    elif cmd == "band":
        v = args[0] if args else BASE_VARIANT
        band, means = noise_band(v)
        print(f"{v}: per-seed mean bp {[round(m, 2) for m in means]}, noise band {band:.2f} bp")
    elif cmd == "table":
        print(table(args))
    else:
        raise SystemExit(__doc__)
