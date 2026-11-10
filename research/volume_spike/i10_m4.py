"""Issue #10 M4 summaries (docs/plans/cnn-improvements.md, M4): epochs and training time.

    cd research/volume_spike
    export MOONSHOT_UNIVERSE=all
    uv run --group research python i10_m4.py epochs base3 xs_univ volscale ...

For each run, from `cache_all/wf_i10/<name>_log.csv`, folds 2022-24 only (the fold-2026 rows
validate on 2025 H2, a sealed period, and are filtered out before anything is collected):
epochs run / kept per (fold, seed) (kept counted from 1, as in M1.md) and the fit minutes
(sum of the epochs' seconds, training plus validation), and from `<name>_run.txt` the run's
total wall time (all five folds).
"""

from __future__ import annotations

import re
import sys

import polars as pl
from i10_train import WF_I10

FOLDS = (2022, 2023, 2024)


def epochs(name: str) -> pl.DataFrame:
    """Per (fold, seed) in 2022-24: epochs run, epoch kept, fit minutes."""
    log = (
        pl.scan_csv(WF_I10 / f"{name}_log.csv")
        .filter(pl.col("fold").cast(pl.Int64).is_in(FOLDS))
        .collect()
    )
    rows = []
    for (fold, seed), g in log.group_by("fold", "seed", maintain_order=True):
        g = g.sort("epoch")
        fixed = g["select"][0] == "fixed"
        kept = len(g) if fixed else int(g["epoch"][int(g["metric"].arg_max() or 0)]) + 1
        rows.append(
            {
                "fold": fold,
                "seed": seed,
                "run": len(g),
                "kept": kept,
                "fit_min": g["secs"].sum() / 60,
            }
        )
    return pl.DataFrame(rows).sort("fold", "seed")


def total_minutes(name: str) -> float | None:
    text = (WF_I10 / f"{name}_run.txt").read_text()
    m = re.search(r"^total ([\d.]+) min", text, re.M)
    return float(m.group(1)) if m else None


def show(name: str) -> None:
    e = epochs(name)
    cells = [
        f"{r['fold']}: "
        + " ".join(
            f"{x['run']}/{x['kept']}"
            for x in e.filter(pl.col("fold") == r["fold"]).iter_rows(named=True)
        )
        + f" ({e.filter(pl.col('fold') == r['fold'])['fit_min'].sum():.1f} min)"
        for r in e.unique("fold", maintain_order=True).iter_rows(named=True)
    ]
    print(
        f"{name}: epochs run/kept per seed, 2022-24 -- {'; '.join(cells)};"
        f" mean run {e['run'].mean():.2f}, kept {e['kept'].mean():.2f};"
        f" total wall time (5 folds) {total_minutes(name)} min"
    )


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "epochs":
        for n in sys.argv[2:]:
            show(n)
    else:
        raise SystemExit(__doc__)
