"""W5g: forward-window CNN scores (2026-11-02..27) per day: max pct, rows >= .98, crash rows."""

import polars as pl
from common_ls import OWN

f = pl.read_parquet(OWN / "forward_signals_day5.parquet")
print(f.schema)
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)
tcol = next(c for c in f.columns if "time" in c)
pc = next(c for c in f.columns if "pct" in c or "score" in c or "day5" in c)
print(tcol, pc)
g = (
    f.with_columns(day=pl.col(tcol).dt.date())
    .group_by("day")
    .agg(
        n=pl.len(),
        mx=pl.col(pc).max().round(3),
        n98=(pl.col(pc) >= 0.98).sum(),
        n95=(pl.col(pc) >= 0.95).sum(),
        r4min=pl.col("ret_4h").min().round(3) if "ret_4h" in f.columns else pl.lit(0),
    )
    .sort("day")
)
print(g)
