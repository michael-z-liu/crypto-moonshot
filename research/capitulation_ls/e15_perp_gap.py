"""E15: where does the perp-vs-spot gap of the pump short come from? And how often does a short
see a +100% adverse move (liquidation at 1x isolated) or end below -100% (cross margin)?"""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN
from sim import paths, perp_paths

d = pl.read_parquet(OWN / "feats.parquet").filter(
    pl.col("perp")
    & (pl.col("qv_base") >= 20_000)
    & (pl.col("ret_4h") >= 0.4)
    & (pl.col("open_time").dt.year() >= 2021)
)
from common_ls import first_per_event, meta  # noqa: E402

keep = first_per_event(np.isin(np.arange(meta().height), d["row"].to_numpy()))
rows = np.array(sorted(set(keep.tolist())))
pp, ok = perp_paths(rows)
rows = rows[ok]
sp = paths(rows)
k72p = 96 + (72 - 9)  # perp path index of hour 72
k72s = 96 + (72 - 9)
short_p = -pp[:, 2, k72p]
short_s = -sp[:, 2, k72s]
adv_p = pp[:, 0, : k72p + 1].max(1)
adv_s = sp[:, 0, : k72s + 1].max(1)
x = (
    d.filter(pl.col("row").is_in(rows.tolist()))
    .sort("row")
    .select("row", "symbol", "open_time", "ret_4h")
)
x = x.with_columns(
    short_perp=pl.Series(short_p),
    short_spot=pl.Series(short_s),
    adv_perp=pl.Series(adv_p),
    adv_spot=pl.Series(adv_s),
    gap=pl.Series(short_p - short_s),
)
print(x.height, "events (one per event)")
print(
    x.select(
        perp=pl.col("short_perp").mean() * 1e4,
        spot=pl.col("short_spot").mean() * 1e4,
        perp_med=pl.col("short_perp").median() * 1e4,
        spot_med=pl.col("short_spot").median() * 1e4,
        adv100_perp=(pl.col("adv_perp") >= 1).sum(),
        adv100_spot=(pl.col("adv_spot") >= 1).sum(),
        end_below_m100=(pl.col("short_perp") <= -1).sum(),
        gap_med=pl.col("gap").median() * 1e4,
        gap_mean_ex_top10=pl.col("gap").sort().slice(10, None).mean() * 1e4,
    )
)
pl.Config.set_tbl_width_chars(200)
print(x.sort("gap").head(12))
print(x.filter(pl.col("adv_perp") >= 1).sort("open_time"))
