"""M4: score the base CNN (seed mean and seed by seed) and the GBM with the M3 scorecard.

    uv run --group research python m4_score.py

Base rule: limit at L(1 + 0.05u), exit E1, percentile >= 0.80, 10 slots. Rows written to the
ledger: `cnn-base` (seed mean), `cnn-base-seed<i>` (via `score_seeds`) and `gbm-base` (seed 0).
Each `source` carries the training run id, so a retrained model gets a new row instead of
silently replacing the old one. The *every candidate* base row is recomputed for the comparison
without touching the ledger.

Diagnostics (written to `results/m4/`): the validation Spearman correlation by epoch for folds
2022-2025, and the mean target by score decile and the score-target Spearman correlation on
test rows decided and filled before 2025 (folds 2022-2024). Nothing of a 2025-26 fill is read
here: the label table is filtered on fill time first, and the scorecard seals 2025-26 itself.
"""

from __future__ import annotations

import json
import time

import build_inputs as bi
import polars as pl
import train as tr
from common import CACHE, HOLDOUT_START, RESULTS
from scorecard import Rule, score_seeds, scorecard, show

STAGE = "M4"
OUT = RESULTS / "m4"
HOLD_US = int(HOLDOUT_START.timestamp()) * 1_000_000
SEEDS = 3


def run_id(out: str) -> str:
    return str(json.loads((tr.SCORES / out / "run.json").read_text())["run_id"])


def scores(out: str, which: str) -> pl.DataFrame:
    return pl.read_parquet(tr.SCORES / out / f"{which}.parquet").select("cid", "score")


def targets_before_holdout() -> pl.DataFrame:
    """cid, decision year and the base target of limit rows decided and filled before 2025."""
    lab = pl.read_parquet(CACHE / "m3" / "lab-b0.05-base-E1.parquet").filter(
        pl.col("entry_t") < HOLD_US
    )
    assert not lab["censored"].any(), "no censored label before 2025"
    rows = bi.rows().select("row", "cid", "u", "decision_t")
    df = lab.select(pl.col("cand").cast(pl.UInt32).alias("row"), "entry_price", "exit_price").join(
        rows, on="row"
    )
    df = df.filter(pl.col("decision_t") < HOLD_US).with_columns(
        y=((pl.col("exit_price") / pl.col("entry_price") - 1) / pl.col("u")).clip(-5, 5),
        year=pl.from_epoch("decision_t", time_unit="us").dt.year(),
    )
    return df.select("cid", "year", "y")


def deciles(name: str, sc: pl.DataFrame, tg: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    d = sc.join(tg, on="cid").filter(pl.col("year").is_between(2022, 2024))
    d = d.with_columns(decile=(pl.col("score") * 10).floor().clip(0, 9).cast(pl.Int32))
    tab = (
        d.group_by("decile")
        .agg(rows=pl.len(), mean_y=pl.col("y").mean(), win=(pl.col("y") > 0).mean())
        .sort("decile")
        .with_columns(model=pl.lit(name))
    )
    rho = []
    for y in (2022, 2023, 2024):
        g = d.filter(pl.col("year") == y)
        rho.append(
            f"{y} {tr.spearman(g['score'].to_numpy(), g['y'].to_numpy()):+.3f} ({g.height} rows)"
        )
    return tab, rho


def curves(out: str) -> pl.DataFrame:
    log = pl.read_csv(tr.SCORES / out / "log.csv")
    assert log["year"].max() < tr.SEALED_YEAR, "no sealed fold in the committed log"
    return log


def main() -> None:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rule = Rule(family="limit", buffer=0.05, exit="E1")
    cnn_id, gbm_id = run_id("base"), run_id("gbm")
    texts: list[str] = []

    every = scorecard(
        "every-L0.05-E1", rule, stage="M3", source="every candidate (no model)", ledger=False,
        seal=False,
    )  # fmt: skip
    texts.append(show(every))
    mean = scorecard(
        "cnn-base",
        rule,
        scores("base", "mean"),
        stage=STAGE,
        source=f"M4 base CNN, seed mean of 3, run {cnn_id}",
    )
    texts.append(show(mean))
    seeds = score_seeds(
        "cnn-base",
        rule,
        [scores("base", f"seed{i}") for i in range(SEEDS)],
        stage=STAGE,
        source=f"M4 base CNN, run {cnn_id}",
    )
    texts += [show(c) for c in seeds.cards]
    gbm = scorecard(
        "gbm-base",
        rule,
        scores("gbm", "seed0"),
        stage=STAGE,
        source=f"M4 GBM benchmark, seed 0, run {gbm_id}",
    )
    texts.append(show(gbm))
    spread = (
        "base CNN per-seed rank values: "
        + ", ".join(f"seed {i} {v * 100:+.2f}%" for i, v in enumerate(seeds.rank_values))
        + f"; mean {seeds.mean_rank * 100:+.2f}%; spread (max - min) {seeds.spread * 100:.2f} pp"
    )
    texts.append(spread)
    print("\n\n".join(texts), flush=True)
    (RESULTS / "m4_cards.txt").write_text("\n\n".join(texts) + "\n")

    tg = targets_before_holdout()
    tabs, rhos = [], []
    for name, sc in [
        ("cnn seed mean", scores("base", "mean")),
        *[(f"cnn seed {i}", scores("base", f"seed{i}")) for i in range(SEEDS)],
        ("gbm", scores("gbm", "seed0")),
    ]:
        tab, rho = deciles(name, sc, tg)
        tabs.append(tab)
        rhos.append(f"{name}: " + "; ".join(rho))
    dec = pl.concat(tabs)
    dec.write_csv(OUT / "deciles.csv")
    (OUT / "spearman_test.txt").write_text("\n".join(rhos) + "\n")
    with pl.Config(tbl_rows=60, float_precision=3):
        print(dec.pivot("model", index="decile", values="mean_y"))
    print("\n".join(rhos))
    cv = curves("base")
    cv.write_csv(OUT / "curves_cnn.csv")
    piv = (
        cv.with_columns(key=pl.format("{} s{}", "year", "seed"))
        .pivot("epoch", index="key", values="valid_spearman")
        .sort("key")
    )
    with pl.Config(tbl_rows=60, tbl_cols=20, float_precision=3):
        print(piv)
    pl.read_csv(tr.SCORES / "gbm" / "log.csv").write_csv(OUT / "curves_gbm.csv")
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
