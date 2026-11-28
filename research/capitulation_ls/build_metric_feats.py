"""Futures-metric features at entry for feats.parquet rows with a perp: open interest change over
the last 1h/4h/24h, OI value relative to spot volume, long/short ratios and their 4h change, taker
long/short ratio. Snapshots taken at or before the spike candle's open time (5 minutes stale, so
no look-ahead). Output: OWN/metric_feats.parquet (row + features)."""

from __future__ import annotations

import polars as pl
from common_ls import OWN
from futures_ls import spot_name

m = (
    pl.read_parquet(OWN / "futures" / "metrics.parquet")
    .with_columns(symbol=pl.col("perp").map_elements(spot_name, return_dtype=pl.Utf8))
    .sort("time")
)
d = (
    pl.read_parquet(
        OWN / "feats.parquet", columns=["row", "symbol", "open_time", "qv_base", "perp"]
    )
    .filter("perp")
    .sort("open_time")
)
cols = ["oi", "oi_value", "top_ls_acc", "top_ls_pos", "ls_acc", "taker_ls"]
base = d.join_asof(
    m.select("symbol", "time", *cols),
    left_on="open_time",
    right_on="time",
    by="symbol",
    strategy="backward",
    tolerance="30m",
)
out = base.rename({c: f"{c}_0" for c in cols}).drop("time")
for lag, lab in ((60, "1h"), (240, "4h"), (1440, "24h")):
    x = d.with_columns(tl=pl.col("open_time") - pl.duration(minutes=lag)).sort("tl")
    x = x.join_asof(
        m.select("symbol", "time", *cols),
        left_on="tl",
        right_on="time",
        by="symbol",
        strategy="backward",
        tolerance="30m",
    )
    out = out.join(
        x.select("row", *[pl.col(c).alias(f"{c}_{lab}") for c in cols]), on="row", how="left"
    )
out = out.with_columns(
    oi_ch1h=pl.col("oi_0") / pl.col("oi_1h") - 1, oi_ch4h=pl.col("oi_0") / pl.col("oi_4h") - 1, oi_ch24h=pl.col("oi_0") / pl.col("oi_24h") - 1,  # noqa: E501
    oi_to_vol=pl.col("oi_value_0") / (pl.col("qv_base") * 288),
    top_ls_acc=pl.col("top_ls_acc_0"), top_ls_pos=pl.col("top_ls_pos_0"), ls_acc=pl.col("ls_acc_0"), taker_ls=pl.col("taker_ls_0"),  # noqa: E501
    ls_acc_ch4h=pl.col("ls_acc_0") - pl.col("ls_acc_4h"), top_ls_pos_ch4h=pl.col("top_ls_pos_0") - pl.col("top_ls_pos_4h"),  # noqa: E501
    taker_ls_1h=pl.col("taker_ls_1h"),
).select("row", "oi_ch1h", "oi_ch4h", "oi_ch24h", "oi_to_vol", "top_ls_acc", "top_ls_pos", "ls_acc", "taker_ls",  # noqa: E501
         "ls_acc_ch4h", "top_ls_pos_ch4h", "taker_ls_1h")  # fmt: skip
out.write_parquet(OWN / "metric_feats.parquet")
print(out.shape, out.null_count())
