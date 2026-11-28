"""P1 (perp-only coins): score the perp-only spikes with the day5 walk-forward fold of their year
(2022 .. 2025: `.cache/forward_model/day5_<Y>`, from h1_fold2022.py / p1_folds.py; 2026: the saved
2026 fold `day5`). pct = the fold's percentile, exactly as `paper.Model` / forward_signals.py.

Writes OWN/p1/scores.parquet (row, raw, pct, fold).
Run: uv run --group research python p1_score.py
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from p1_lib import P1, FoldModel, fold_name


def main() -> None:
    meta = pl.read_parquet(P1 / "spikes.parquet")
    seq = np.load(P1 / "seq.npy", mmap_mode="r")
    yr = meta["year"].to_numpy()
    qv = meta["qv_base"].to_numpy()
    raw = np.full(meta.height, np.nan, np.float32)
    pct = np.full(meta.height, np.nan, np.float32)
    for y in sorted(set(yr.tolist())):
        rows = np.flatnonzero(yr == y)
        fm = FoldModel(fold_name(y))
        for a in range(0, len(rows), 100_000):
            r = rows[a : a + 100_000]
            raw[r], pct[r] = fm.score(np.asarray(seq[r]), qv[r])
        print(y, fold_name(y), len(rows), "scored; pct >= 0.995:", int((pct[rows] >= 0.995).sum()),
              file=sys.stderr)  # fmt: skip
    pl.DataFrame(
        {
            "row": np.arange(meta.height),
            "raw": raw,
            "pct": pct,
            "fold": [fold_name(int(y)) for y in yr],
        }
    ).write_parquet(P1 / "scores.parquet")


if __name__ == "__main__":
    main()
