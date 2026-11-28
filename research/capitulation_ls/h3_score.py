"""H3 (other quotes, 2020-26): score the converted-pair spikes (h3_build.py) with the day5
walk-forward fold of their year, as p1_score.py does (`p1_lib.FoldModel`, the `paper.Model`
scoring: seed-mean 1h median forecast, pct against the fold's own reference scores):
2022 .. 2025 -> `day5_<Y>`, 2026 -> `day5`. 2020 and 2021 have no walk-forward fold in the study:
they get the 2022 fold applied backwards (`day5_2022`, trained on 2020-01 .. 2021-06 USDT rows
and early-stopped on 2021 H2, so those years are in its training period, on other coins).

For the "new flush days" comparison in 2020-21 the study's own USDT spikes of those years
(cache_all seq2) are scored the same way with `day5_2022`.

Writes OWN/h3/{scores.parquet, usdt_2020_21.parquet}.
Run: uv run --group research python h3_score.py [pairs] [usdt]
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from common_ls import CACHE_ALL
from h3_fetch import H3
from p1_lib import FoldModel


def fold_of(year: int) -> str:
    return "day5" if year == 2026 else f"day5_{max(year, 2022)}"


def score_rows(
    seq: np.ndarray, qv: np.ndarray, rows: np.ndarray, fm: FoldModel
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.full(len(rows), np.nan, np.float32)
    pct = np.full(len(rows), np.nan, np.float32)
    for a in range(0, len(rows), 50_000):
        r = rows[a : a + 50_000]
        x = np.asarray(seq[r])
        ok = np.isfinite(x).all(axis=(1, 2))
        rr, pp = fm.score(x, qv[r])
        raw[a : a + 50_000] = np.where(ok, rr, np.nan)
        pct[a : a + 50_000] = np.where(ok, pp, np.nan)
    return raw, pct


def main() -> None:
    what = sys.argv[1:] or ["pairs", "usdt"]
    if "pairs" in what:
        score_pairs()
    if "usdt" in what:
        score_usdt()


def score_pairs() -> None:
    meta = pl.read_parquet(H3 / "meta.parquet")
    seq = np.load(H3 / "seq.npy", mmap_mode="r")
    yr = meta["open_time"].dt.year().to_numpy()
    qv = meta["qv_base"].to_numpy()
    raw = np.full(meta.height, np.nan, np.float32)
    pct = np.full(meta.height, np.nan, np.float32)
    for y in sorted(set(yr.tolist())):
        rows = np.flatnonzero(yr == y)
        fm = FoldModel(fold_of(y))
        raw[rows], pct[rows] = score_rows(seq, qv, rows, fm)
        print(y, fold_of(y), len(rows), "scored; pct >= 0.995:", int((pct[rows] >= 0.995).sum()),
              file=sys.stderr)  # fmt: skip
    pl.DataFrame(
        {"row": np.arange(meta.height), "raw": raw, "pct": pct,
         "fold": [fold_of(int(y)) for y in yr]}
    ).write_parquet(H3 / "scores.parquet")  # fmt: skip


def score_usdt() -> None:
    um = pl.read_parquet(CACHE_ALL / "seq2_meta.parquet", columns=["open_time", "qv_base"])
    useq = np.load(CACHE_ALL / "seq2.npy", mmap_mode="r")
    rows = np.flatnonzero((um["open_time"].dt.year() < 2022).to_numpy())
    fm = FoldModel("day5_2022")
    r2, p2 = score_rows(useq, um["qv_base"].to_numpy(), rows, fm)
    print("USDT 2020-21 rows", len(rows), "pct >= 0.995:", int((p2 >= 0.995).sum()),
          file=sys.stderr)  # fmt: skip
    pl.DataFrame({"row": rows, "raw": r2, "pct": p2}).write_parquet(H3 / "usdt_2020_21.parquet")


if __name__ == "__main__":
    main()
