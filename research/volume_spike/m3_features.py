"""M3: gross forward return by feature decile, for spikes and for random bars (train only)."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from common import RESULTS
from evalx import load_events

FEATURES = [
    "vol_mult", "vol_mult_3h", "tbr", "tbr_excess", "ret_bar", "range_mult", "clv", "upper_wick",
    "runup_1h", "ret_4h", "ret_24h", "ret_7d", "dist_hi_24h", "dist_hi_7d", "dist_lo_24h",
    "vola_24h", "qv_base", "qv_usdt", "trades_mult", "avg_trade_mult", "prev_spikes_1h",
    "prev_spikes_24h", "tick_rel", "flat_share", "listed_days", "btc_ret_bar", "btc_runup_1h",
    "btc_ret_4h", "btc_ret_24h", "btc_ret_7d", "btc_vol_mult", "hour",
]  # fmt: skip


def main(group: str) -> None:
    events, fwd = load_events()
    mask = (events["split"] == "train") & events[f"is_{group}"]
    idx = np.flatnonzero(mask.to_numpy())
    df = events.filter(mask).with_columns(
        f1h=pl.Series(np.asarray(fwd[idx, 2, 11])),
        f4h=pl.Series(np.asarray(fwd[idx, 2, 47])),
        mfe4h=pl.Series(np.asarray(fwd[idx, 0, :]).max(1)),
    )
    out = []
    for f in FEATURES:
        d = df.filter(pl.col(f).is_finite()).with_columns(
            q=pl.col(f).qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True)
        )
        g = (
            d.group_by("q")
            .agg(
                pl.len().alias("n"),
                pl.col(f).cast(pl.Float64).median().alias("x"),
                (pl.col("f1h").mean() * 1e4).alias("f1h"),
                (pl.col("f4h").mean() * 1e4).alias("f4h"),
                (pl.col("fwd_24h").mean() * 1e4).alias("f24h"),
                (pl.col("mfe4h").median() * 1e4).alias("mfe4h_med"),
            )
            .sort("q")
            .with_columns(feature=pl.lit(f))
        )
        out.append(g)
        cells = " ".join(f"{a:+6.0f}" for a in g["f4h"].to_list())
        cells24 = " ".join(f"{a:+6.0f}" for a in g["f24h"].to_list())
        print(f"{f:16s} 4h: {cells}\n{'':16s} 24h:{cells24}")
    pl.concat(out).write_csv(RESULTS / f"m3_deciles_{group}.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "spike")
