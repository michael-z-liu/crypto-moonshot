"""Equal-weight daily index of the universe (ex BTC), for market-regime features."""

from __future__ import annotations

import polars as pl
from common import CACHE, KLINES_5M, universe


def build() -> pl.DataFrame:
    frames = []
    for s in universe():
        if s == "BTCUSDT":
            continue
        d = (
            pl.scan_parquet(KLINES_5M / f"symbol={s}" / "*.parquet")
            .select("open_time", "close")
            .group_by_dynamic("open_time", every="1d")
            .agg(pl.col("close").last())
            .with_columns(r=pl.col("close").pct_change(), symbol=pl.lit(s))
            .collect()
        )
        frames.append(d.slice(7))  # skip each coin's first week
    daily = (
        pl.concat(frames)
        .filter(pl.col("r").is_finite() & (pl.col("r").abs() < 2))
        .group_by("open_time")
        .agg(pl.col("r").mean().alias("alt_r"), pl.len().alias("alt_n"))
        .sort("open_time")
        .with_columns(alt_index=(1 + pl.col("alt_r")).cum_prod())
        .with_columns(
            alt_ret_7d=pl.col("alt_index") / pl.col("alt_index").shift(7) - 1,
            alt_ret_30d=pl.col("alt_index") / pl.col("alt_index").shift(30) - 1,
            alt_ret_90d=pl.col("alt_index") / pl.col("alt_index").shift(90) - 1,
        )
        # values known at the end of day D are usable from day D+1
        .with_columns(day=(pl.col("open_time") + pl.duration(days=1)).dt.date())
        .select("day", "alt_ret_7d", "alt_ret_30d", "alt_ret_90d")
    )
    daily.write_parquet(CACHE / "alt_index.parquet")
    return daily


if __name__ == "__main__":
    print(build().tail())
