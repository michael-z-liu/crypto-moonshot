"""M2: does the raw volume-spike trigger beat random entries, and with which exits?"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common import RESULTS
from evalx import load_events, summary, triple_barrier

TPS = [0.01, 0.02, 0.03, 0.05, None]
SLS = [0.01, 0.02, 0.03, 0.05, None]
BARS = [3, 6, 12, 24, 48]


def main() -> None:
    events, fwd = load_events()
    train = (events["split"] == "train").to_numpy()
    groups = {
        "spike": events["is_spike"].to_numpy() & train,
        "random": events["is_random"].to_numpy() & train,
        "spike_up": (events["is_spike"] & (events["ret_bar"] > 0)).to_numpy() & train,
    }
    rows = []
    for (name, mask), tp, sl, bars in itertools.product(groups.items(), TPS, SLS, BARS):
        idx = np.flatnonzero(mask)
        r = triple_barrier(fwd[idx], tp, sl, bars)
        rows.append({"group": name, "tp": tp, "sl": sl, "bars": bars, **summary(r)})
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m2_baseline_grid.csv")
    with pl.Config(tbl_rows=40, tbl_cols=10):
        for g in groups:
            print(g)
            print(df.filter(pl.col("group") == g).sort("mean_bp", descending=True).head(8))
        print("user 2019 setup (+1% / -5%)")
        print(df.filter((pl.col("tp") == 0.01) & (pl.col("sl") == 0.05)))
        print("gross close-to-close by horizon, no barriers (cost added back)")
        print(
            df.filter(pl.col("tp").is_null() & pl.col("sl").is_null()).with_columns(
                gross_bp=pl.col("mean_bp") + 30
            )
        )


if __name__ == "__main__":
    main()
