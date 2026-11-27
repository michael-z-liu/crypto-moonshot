"""Forward signals (unseen data after 2026-11-02): every liquid spike in the synced candles,
scored with the path_all 2026 fold saved by the paper test (P1 of the robustness study). Writes
OWN/forward_signals.parquet with the spike features and pct; computes no outcomes.

Reads candles from the shared data folder (no writes there) and the model from
.cache/paper/model (a copy of the paper test's). Run: uv run --group research python forward_signals.py
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ["PAPER_DIR"] = str(ROOT / ".cache" / "paper")
os.environ["PAPER_DATA_DIR"] = os.environ.get("MOONSHOT_DATA_DIR", "/Users/mike/code/crypto-moonshot/data")
sys.path.insert(0, str(HERE.parent / "volume_spike"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import paper  # noqa: E402

OUT = Path("/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/forward_signals.parquet")


def main() -> None:
    start = datetime(2026, 11, 2, tzinfo=UTC)
    end = paper.data_end()
    sp = paper.spikes(paper.coins(), start, end)
    print(len(sp), "spikes from", start, "to", end, file=sys.stderr)
    seq = paper.build_inputs(sp, paper.INPUTS / "forward", paper.market_series(end))
    raw, pct = paper.Model().score(seq, sp["qv_base"].to_numpy())
    # BTC 4h return at each spike (as seq2_meta's btc_ret_4h): BTC close channel 7, 48 bars back.
    btc4 = -np.asarray(seq[:, 7, -49], dtype=np.float64)
    sp = sp.with_columns(raw=pl.Series(raw.astype(np.float64)), pct=pl.Series(pct.astype(np.float64)), btc_ret_4h=pl.Series(np.expm1(btc4)))
    sp.write_parquet(OUT)
    print("written", OUT, sp.height, "rows; data end", end, file=sys.stderr)


if __name__ == "__main__":
    main()
