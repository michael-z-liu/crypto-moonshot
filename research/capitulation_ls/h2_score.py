"""H2 (holdout, BTC pairs): score the converted BTC-pair spikes (h2_build.py) with the saved day5
models exactly as h1_score.py does (`h1_score.load_model`, `paper.Model.score`: seed-mean 1h
median forecast `raw`, pct against the fold's own reference scores). Models: `day5` (2026 fold,
primary) and `day5_2022` (check). Rows with non-finite inputs get pct NaN, as in h1.

Sensitivity: the market channels 11-12 are rebuilt from h1's market series (USDT pairs only)
instead of the combined universe, and the day5 model scores those inputs too (`pct_day5_mktusdt`).

Writes holdout/btc/scores.parquet.
Run: uv run --group research python h2_score.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from h1_score import MODELS, ROOT, load_model

HOLDOUT = Path("/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/holdout")
OUT = HOLDOUT / "btc"
T0_H = datetime(2017, 7, 1, tzinfo=UTC)
SEQ = 288


def usdt_market_channels(meta: pl.DataFrame) -> np.ndarray:
    """Channels 11-12 from h1's USDT-only market series, as build_seq2 computes them."""
    market = pl.read_parquet(HOLDOUT / "market.parquet")
    gidx = (
        meta.select((pl.col("open_time") - pl.lit(T0_H)) / pl.lit(timedelta(minutes=5)))
        .to_series()
        .cast(pl.Int64)
        .to_numpy()
    )
    k = gidx[:, None] - SEQ + 1 + np.arange(SEQ)[None, :]
    cum = np.cumsum(market["alt_r"].to_numpy())
    share = market["spike_share"].to_numpy()
    out = np.empty((meta.height, 2, SEQ), dtype=np.float16)
    out[:, 0] = (cum[k] - cum[gidx][:, None]).astype(np.float16)
    out[:, 1] = share[k].astype(np.float16)
    return out


def main() -> None:
    meta = pl.read_parquet(OUT / "seq2_meta.parquet")
    seq = np.load(OUT / "seq2.npy")
    finite = np.isfinite(seq).all(axis=(1, 2))
    qv = meta["qv_base"].to_numpy()
    out = pl.DataFrame({"row": np.arange(meta.height), "finite": finite})
    lines = [f"BTC-pair rows {meta.height}, non-finite inputs {int((~finite).sum())}"]
    for name in MODELS:
        if not (ROOT / ".cache" / "forward_model" / name / "norm.npz").exists():
            lines.append(f"{name}: not saved, skipped")
            continue
        model = load_model(name)
        raw, pct = model.score(seq, qv)
        out = out.with_columns(
            pl.Series(f"raw_{name}", np.where(finite, raw, np.nan).astype(np.float64)),
            pl.Series(f"pct_{name}", np.where(finite, pct, np.nan).astype(np.float64)),
        )
        lines.append(f"{name}: share pct >= 0.995 {float(np.mean(pct[finite] >= 0.995)):.4f}")
        if name == "day5":
            alt = seq.copy()
            alt[:, 11:13] = usdt_market_channels(meta)
            raw2, pct2 = model.score(alt, qv)
            out = out.with_columns(
                pl.Series("pct_day5_mktusdt", np.where(finite, pct2, np.nan).astype(np.float64))
            )
            ok = finite & np.isfinite(raw2)
            lines.append(
                f"day5 with USDT-only market channels: raw corr"
                f" {np.corrcoef(raw[ok], raw2[ok])[0, 1]:.4f}, share pct >= 0.995"
                f" {float(np.mean(pct2[ok] >= 0.995)):.4f}, agreement on >= 0.995"
                f" {float(np.mean((pct[ok] >= 0.995) == (pct2[ok] >= 0.995))):.4f}"
            )
    out.write_parquet(OUT / "scores.parquet")
    (OUT / "score_checks.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), file=sys.stderr)


if __name__ == "__main__":
    main()
