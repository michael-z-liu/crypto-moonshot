"""E07: walk-forward pump shorts. Arg: spot (spot paths, no perp gate, no funding) | perp."""

from __future__ import annotations

import itertools
import sys
from typing import Any

import numpy as np
import polars as pl
from common_ls import meta
from sim import Rule, simulate, simulate_yearly
from wf import summarize, table, walk_forward

MODE = sys.argv[1] if len(sys.argv) > 1 else "spot"


def cands(thr: float) -> pl.DataFrame:
    m = meta()
    r4, qv, yr = m["ret_4h"].to_numpy(), m["qv_base"].to_numpy(), m["year"].to_numpy()
    mask = (yr >= 2021) & (r4 >= thr) & (qv >= 20_000)
    if MODE == "perp":
        from futures_ls import has_perp

        idx0 = np.flatnonzero(mask)
        ok = has_perp(m["symbol"].to_numpy()[idx0], m["open_time"].dt.epoch("s").to_numpy()[idx0])
        mask = np.zeros(len(m), bool)
        mask[idx0[ok]] = True
    idx = np.flatnonzero(mask)
    return pl.DataFrame({"row": idx, "side": -1, "prio": r4[idx]})


def run(cfg: dict[str, Any]) -> pl.DataFrame:
    c = cands(cfg["thr"])
    rule = Rule(hold_min=cfg["hold"], tp=cfg["tp"], sl=cfg["sl"], slots=cfg["slots"], cost=0.003, stop_slip=0.01)
    return simulate_yearly(c, rule, perp_funding=MODE == "perp").years


if __name__ == "__main__":
    grid = [dict(thr=t, hold=h, tp=tp, sl=sl, slots=s) for t, h, tp, sl, s in itertools.product(
        (0.3, 0.4, 0.5, 0.6, 0.8), (240, 720, 1440, 2880, 4320), (None, 0.2, 0.3, 0.5), (None, 0.5), (3, 5, 10))]  # fmt: skip
    tab = table(run, grid)
    tab.write_parquet(f"/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/e07_{MODE}.parquet")
    wf = walk_forward(tab)
    print(MODE, "walk-forward:")
    print(summarize(wf))
    # Context: how many configs are positive in every year, and the median config.
    g = tab.group_by("cfg").agg(minret=pl.col("ret").min(), med=pl.col("ret").median(), bp=pl.col("bp").mean())
    print("configs:", g.height, "positive every year:", int((g["minret"] > 0).sum()), "median of yearly-median returns:", float(g["med"].median()))
