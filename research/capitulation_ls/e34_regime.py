"""E34: regime splits for the sleeves (BTC 7d / 30d trend at entry, weekday)."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN, meta, scores
from sim import Rule, simulate

m = meta()
yr = m["year"].to_numpy()
btc = pl.read_parquet(OWN / "feats.parquet", columns=["row", "r7d", "r30d"])  # coin-level; need BTC
# BTC 7d/30d trend from the BTC rows of seq2 meta (BTCUSDT spikes are sparse), so use hourly BTC closes.
import os, sys
os.environ["MOONSHOT_UNIVERSE"] = "all"
from common_ls import CACHE_ALL
sys.path.insert(0, str(CACHE_ALL.parent))
from common import load  # noqa: E402

b = load("BTCUSDT").group_by_dynamic("open_time", every="1h").agg(c=pl.col("close").last()).sort("open_time")
bt, bc = b["open_time"].dt.epoch("s").to_numpy(), b["c"].to_numpy()


def btc_trend(t: np.ndarray, days: int) -> np.ndarray:
    k = np.searchsorted(bt, t - 3600, side="right") - 1
    j = np.searchsorted(bt, t - 3600 - days * 86400, side="right") - 1
    return bc[np.maximum(k, 0)] / bc[np.maximum(j, 0)] - 1


def split(name: str, tr: pl.DataFrame) -> None:
    t = tr["t"].to_numpy()
    tr = tr.with_columns(b7=pl.Series(btc_trend(t, 7)), b30=pl.Series(btc_trend(t, 30)), wd=pl.col("open_time").dt.weekday())
    for col in ("b7", "b30"):
        g = tr.with_columns(q=pl.col(col).qcut(3, labels=["down", "mid", "up"])).group_by("q").agg(n=pl.len(), bp=(pl.col("net").mean() * 1e4).round(), med=(pl.col("net").median() * 1e4).round()).sort("q")
        print(name, col, g.to_dicts())
    g = tr.group_by(pl.col("wd") >= 6).agg(n=pl.len(), bp=(pl.col("net").mean() * 1e4).round()).sort("wd")
    print(name, "weekend", g.to_dicts())


if __name__ == "__main__":
    pct = scores("day5")
    idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.995))
    L = simulate(pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]}), Rule(hold_min=4320, tp=0.2, slots=100_000, cost=0.004, liq_frac=None), start_equity=1e12).trades
    split("L", L)
    d = pl.read_parquet(OWN / "feats.parquet").filter(pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("ret_4h") >= 0.4) & (pl.col("open_time").dt.year() >= 2022))
    P = simulate(d.select("row", side=pl.lit(-1), prio="ret_4h"), Rule(hold_min=4320, slots=100_000, cost=0.003, sl=1.0, liq_frac=None), start_equity=1e12, perp_funding=True, source="perp").trades
    split("P", P)

    t = P["t"].to_numpy()
    P = P.with_columns(b7=pl.Series(btc_trend(t, 7)), wd=pl.col("open_time").dt.weekday(), year=pl.col("open_time").dt.year())
    for lab, f in (("all", pl.lit(True)), ("btc7>0", pl.col("b7") > 0), ("btc7>-0.03", pl.col("b7") > -0.03), ("weekday", pl.col("wd") <= 5), ("btc7>0 & weekday", (pl.col("b7") > 0) & (pl.col("wd") <= 5))):
        x = P.filter(f)
        print(lab, x.height, round(float(x["net"].mean()) * 1e4), x.group_by("year").agg(n=pl.len(), bp=(pl.col("net").mean() * 1e4).round()).sort("year").to_dicts())
