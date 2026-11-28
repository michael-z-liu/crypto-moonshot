"""X3 step 3: landscape. Mean directional time-exit return by lag-signal bucket (before costs)."""

import numpy as np
import polars as pl
from common_ls import OWN

EX = OWN / "explore"
d = pl.read_parquet(EX / "lag_cands.parquet")
p = np.load(EX / "lag_paths.npz")
cl, fc = p["cl"], p["fc"]
side = d["side"].to_numpy()
for h in (4, 24, 72):
    d = d.with_columns(
        **{f"g{h}": pl.Series(side * cl[:, h - 1] - (side < 0) * fc[:, h - 1] * -1 * -1)}
    )
d = d.with_columns(
    b=pl.col("signal").cut(
        [0.5, 0.8, 0.9, 0.95], labels=["<.5", ".5-.8", ".8-.9", ".9-.95", ">.95"]
    )
)
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(200)
for combo in range(6):
    for s in (1, -1):
        g = d.filter((pl.col("combo") == combo) & (pl.col("side") == s))
        t = (
            g.group_by("b")
            .agg(
                n=pl.len(),
                g4=pl.col("g4").mean() * 1e4,
                g24=pl.col("g24").mean() * 1e4,
                g72=pl.col("g72").mean() * 1e4,
                med24=pl.col("g24").median() * 1e4,
                win24=(pl.col("g24") > 0.004).mean(),
            )
            .sort("b")
        )
        print(combo, s, g["w"][0], g["thr"][0])
        print(t.with_columns(pl.col(pl.Float64).round(1)))
