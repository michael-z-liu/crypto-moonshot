"""X3 step 1: hourly spot and perp matrices (hours x symbols) for the lag/catch-up study."""

from __future__ import annotations

import os
import sys

os.environ["MOONSHOT_UNIVERSE"] = "all"
import numpy as np
import polars as pl
from common_ls import CACHE_ALL, OWN

sys.path.insert(0, str(CACHE_ALL.parent))
import common
from futures_ls import FUT, spot_name

EX = OWN / "explore"
EX.mkdir(exist_ok=True)
T0 = int(np.datetime64("2020-01-01", "s").astype(int))
T1 = int(np.datetime64("2026-11-01", "s").astype(int))
H = (T1 - T0) // 3600


def mat(frames: dict[str, pl.DataFrame], cols: list[str]) -> dict[str, np.ndarray]:
    syms = sorted(frames)
    out = {c: np.full((H, len(syms)), np.nan, dtype=np.float32) for c in cols}
    for j, s in enumerate(syms):
        f = frames[s]
        idx = ((f["t"].to_numpy() - T0) // 3600).astype(int)
        ok = (idx >= 0) & (idx < H)
        for c in cols:
            out[c][idx[ok], j] = f[c].to_numpy()[ok]
    out["syms"] = np.array(syms)  # type: ignore[assignment]
    return out


spot = {}
for s in common.universe():
    d = common.load(s)
    h = (
        d.with_columns(t=pl.col("open_time").dt.epoch("s") // 3600 * 3600)
        .group_by("t", maintain_order=True)
        .agg(
            open=pl.col("open").first(),
            high=pl.col("high").max(),
            low=pl.col("low").min(),
            close=pl.col("close").last(),
            qv=pl.col("quote_volume").sum(),
        )
        .filter(pl.col("qv") > 0)
    )
    if h.height > 100:
        spot[s] = h
M = mat(spot, ["open", "high", "low", "close", "qv"])
np.savez(EX / "spot_h.npz", **M)
print("spot", M["close"].shape)

p = pl.read_parquet(FUT / "perp_1h.parquet")
p = p.with_columns(
    spot=pl.col("symbol").map_elements(spot_name, return_dtype=pl.Utf8),
    t=pl.col("open_time").dt.epoch("s"),
).filter(pl.col("spot").is_in(list(spot)))
p = p.sort("symbol").unique(["spot", "t"], keep="first")
pf = {s: g for (s,), g in p.group_by(["spot"]) if g.height > 100}
P = mat(pf, ["open", "high", "low", "close", "quote_volume"])
np.savez(EX / "perp_h.npz", **P)
print("perp", P["close"].shape)
