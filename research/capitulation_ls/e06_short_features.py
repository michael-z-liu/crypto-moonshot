"""E06: which pumps revert most? Conditional short returns by entry-time features (one per event)."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import CACHE_ALL, first_per_event, fwd72, meta

m = meta()
r4, qv, yr = m["ret_4h"].to_numpy(), m["qv_base"].to_numpy(), m["year"].to_numpy()
import sys

from common_ls import scores

LONG = sys.argv[1:] == ["long"]
SIDE = 1 if LONG else -1
pct = scores("day5")
mask = (r4 <= -0.15) if LONG else (r4 >= 0.30)
idx = first_per_event((yr >= 2021) & mask & (qv >= 20_000))
seq = np.load(CACHE_ALL / "seq2.npy", mmap_mode="r")
s = np.asarray(seq[np.sort(idx)], dtype=np.float32)
idx = np.sort(idx)
f = np.asarray(fwd72()[idx], dtype=np.float64)
d = m[idx].with_columns(
    red=pl.Series(s[:, 0, -1] > 0),  # spike candle open above its close
    off_high=pl.Series(s[:, 1, -48:].max(1)),  # log(4h high / close)
    r1h=pl.Series(-s[:, 3, -13]),
    r15m=pl.Series(-s[:, 3, -4]),
    wick=pl.Series(s[:, 1, -1] - np.maximum(s[:, 0, -1], 0)),  # upper wick of the spike candle
    taker=pl.Series(s[:, 5, -1]),
    btc24=pl.Series(-s[:, 7, 0]),
    y24=pl.Series(SIDE * f[:, 2, 23] - 0.003), y72=pl.Series(SIDE * f[:, 2, 71] - 0.003),
    adv24=pl.Series(f[:, 0, :24].max(1) if SIDE < 0 else -f[:, 1, :24].min(1)),
    pct=pl.Series(pct[idx]),
)


def by(col: str, q: int = 5) -> None:
    g = d.with_columns(b=pl.col(col).qcut(q, allow_duplicates=True)).group_by("b").agg(
        n=pl.len(), y24=pl.col("y24").mean() * 1e4, y24med=pl.col("y24").median() * 1e4,
        y72=pl.col("y72").mean() * 1e4, sq50=(pl.col("adv24") > 0.5).mean() * 100,
        y24_2025=pl.col("y24").filter(pl.col("year") >= 2025).mean() * 1e4,
    ).sort("b")  # fmt: skip
    print(col); print(g)


pl.Config.set_tbl_width_chars(200)
for c in (["pct"] if LONG else []) + ["ret_4h", "ret_24h", "off_high", "r1h", "r15m", "wick", "taker", "vol_mult", "btc24", "btc_ret_4h", "listed_days", "qv_base"]:
    by(c)
print(d.group_by("red").agg(n=pl.len(), y24=pl.col("y24").mean() * 1e4, y72=pl.col("y72").mean() * 1e4))
d.write_parquet(f"/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/e06_{'crash' if LONG else 'pumps'}.parquet")
