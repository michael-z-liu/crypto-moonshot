"""H1 (holdout): score the 2017-08 .. 2020-01 liquid spikes (h1_build.py) with the saved day5
models, exactly as forward_signals.py does: `paper.Model` (seed-mean 1h median forecast `raw`;
`pct` = searchsorted(fold reference scores, raw) / n, the reference being the fold's own sorted
training-row scores saved with the model; frozen channel means / sds and the liq z-score of
norm.npz).

Models: `day5` (the 2026 fold: trained before 2025-07-01, early-stopped on 2025 H2; reference =
its 2020-01 .. 2025-06 training rows) and, if saved by h1_fold2022.py, `day5_2022` (trained
before 2021-07-01, early-stopped on 2021 H2; reference = 2020-01 .. 2021-06 rows).

Rows whose inputs are not finite (15 spikes right after the 2018-02-09 outage: BTC / ETH
volume baselines of 0) get pct NaN instead of the 1.0 a NaN score would sort to.

Parity checks (written to holdout/score_checks.txt):
1. the day5 model through this path vs the walk-forward `day5` pct on 20,000 shared 2026 rows;
2. holdout 2020-01 spikes vs the shared seq2 rows of the same (symbol, open time): spike
   features, channels 0-10 (coin, BTC, ETH; the market channels 11-12 differ by universe), raw.

Writes holdout/scores.parquet (row, raw/pct per model).
Run: uv run --group research python h1_score.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "volume_spike"))

import m12_cnn_wf as m12  # noqa: E402
import numpy as np  # noqa: E402
import paper  # noqa: E402
import polars as pl  # noqa: E402

HOLDOUT = Path("/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/holdout")
CACHE_ALL = HERE.parent / "volume_spike" / "cache_all"
MODELS = ("day5", "day5_2022")


def load_model(name: str) -> paper.Model:
    mdir = ROOT / ".cache" / "forward_model" / name
    paper.MODEL = mdir
    seeds = json.loads((mdir / "config.json").read_text())["seeds"]
    m12.VARIANTS[paper.VARIANT] = replace(m12.VARIANTS[paper.VARIANT], seeds=seeds)
    return paper.Model()


def main() -> None:
    meta = pl.read_parquet(HOLDOUT / "seq2_meta.parquet")
    seq = np.load(HOLDOUT / "seq2.npy", mmap_mode="r")
    finite = np.isfinite(np.asarray(seq, dtype=np.float32)).all(axis=(1, 2))
    out = pl.DataFrame({"row": np.arange(meta.height), "finite": finite})
    checks: list[str] = [f"holdout rows {meta.height}, non-finite inputs {int((~finite).sum())}"]
    smeta = pl.read_parquet(CACHE_ALL / "seq2_meta.parquet").with_row_index("srow")
    sseq = np.load(CACHE_ALL / "seq2.npy", mmap_mode="r")
    for name in MODELS:
        if not (ROOT / ".cache" / "forward_model" / name / "norm.npz").exists():
            checks.append(f"{name}: not saved, skipped")
            continue
        model = load_model(name)
        raw, pct = model.score(seq, meta["qv_base"].to_numpy())
        pct = np.where(finite, pct, np.nan)
        raw = np.where(finite, raw, np.nan)
        out = out.with_columns(
            pl.Series(f"raw_{name}", raw.astype(np.float64)),
            pl.Series(f"pct_{name}", pct.astype(np.float64)),
        )
        # 1. reproduction on the shared rows of the model's own test year
        wf = {"day5": "day5", "day5_2022": "day5"}[name]
        year = 2026 if name == "day5" else 2022
        from common_ls import scores

        ref = scores(wf)
        yrows = np.flatnonzero((smeta["open_time"].dt.year() == year).to_numpy())
        rows = np.sort(np.random.default_rng(0).choice(yrows, 20_000, replace=False))
        _, p2 = model.score(sseq, smeta["qv_base"].to_numpy(), rows)
        same = float(np.mean(p2 == ref[rows].astype(np.float32)))
        checks.append(
            f"{name}: pct vs walk-forward day5 on 20,000 shared {year} rows: equal {same:.2%},"
            f" max abs diff {np.max(np.abs(p2 - ref[rows])):.2e}"
        )
        # 2. parity of the holdout 2020-01 rows against the shared seq2
        j = meta.with_row_index("hrow").filter(pl.col("open_time").dt.year() == 2020)
        j = j.join(smeta.select("srow", "symbol", "open_time", s_qv="qv_base", s_vm="vol_mult"),
                   on=["symbol", "open_time"], how="inner")  # fmt: skip
        hr, sr = j["hrow"].to_numpy(), j["srow"].to_numpy()
        rs, ps = model.score(sseq, smeta["qv_base"].to_numpy(), sr)
        dq = np.max(np.abs(j["qv_base"].to_numpy() / j["s_qv"].to_numpy() - 1))
        dv = np.max(np.abs(j["vol_mult"].to_numpy() / j["s_vm"].to_numpy() - 1))
        a = np.asarray(seq[hr], dtype=np.float32)
        b = np.asarray(sseq[sr], dtype=np.float32)
        dch = np.abs(a - b).max(axis=(0, 2))
        n2020 = int((meta["open_time"].dt.year() == 2020).sum())
        checks.append(
            f"{name}: 2020-01 parity: {len(j)} of {n2020} holdout spikes found in the shared"
            f" seq2; qv_base max rel diff {dq:.2e}, vol_mult {dv:.2e}; channel max abs diff "
            + ", ".join(f"{c}:{d:.3f}" for c, d in enumerate(dch))
            + f"; raw corr {np.corrcoef(raw[hr], rs)[0, 1]:.4f}, median |pct diff|"
            f" {np.median(np.abs(pct[hr] - ps)):.4f}"
        )
    out.write_parquet(HOLDOUT / "scores.parquet")
    (HOLDOUT / "score_checks.txt").write_text("\n".join(checks) + "\n")
    print("\n".join(checks))


if __name__ == "__main__":
    main()
