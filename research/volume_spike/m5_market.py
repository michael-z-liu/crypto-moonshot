"""M5: capitulation during market-wide flushes: grid over drop, liquidity, BTC condition, hold."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common import RESULTS
from portfolio import Exit, load_long, simulate

BTC = {
    "btc_vm>=3": pl.col("btc_vol_mult") >= 3,
    "btc_vm>=4": pl.col("btc_vol_mult") >= 4,
    "btc_vm>=6": pl.col("btc_vol_mult") >= 6,
    "btc_4h<=-1.5%": pl.col("btc_ret_4h") <= -0.015,
    "btc_4h<=-3%": pl.col("btc_ret_4h") <= -0.03,
    "btc_vm>=3|4h<=-2%": (pl.col("btc_vol_mult") >= 3) | (pl.col("btc_ret_4h") <= -0.02),
}


def main() -> None:
    ev, fwd = load_long()
    rows = []
    grid = itertools.product([0.05, 0.06, 0.08, 0.10, 0.12], [5_000, 20_000], BTC.items(),
                             [12, 24, 48, 144])  # fmt: skip
    for drop, liq, (bname, bcond), hold in grid:
        cond = pl.col("is_spike") & (pl.col("ret_4h") <= -drop) & (pl.col("qv_base") >= liq) & bcond
        for split in ("train", "valid"):
            m = ev.select((cond & (pl.col("split") == split)).fill_null(False).alias("m"))["m"]
            idx = np.flatnonzero(m.to_numpy())
            s = simulate(ev[idx], fwd[idx], Exit(bars=hold)).stats()
            rows.append({"drop": drop, "liq": liq, "btc": bname, "hold": hold, "split": split, **s})
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m5_market.csv")
    wide = df.pivot(on="split", index=["drop", "liq", "btc", "hold"],
                    values=["trades_per_week", "mean_bp", "cagr", "max_dd", "sharpe"])  # fmt: skip
    with pl.Config(tbl_rows=40, tbl_cols=14, tbl_width_chars=250, float_precision=2):
        print(wide.sort("sharpe_train", descending=True).head(30))
        print(
            wide.filter(pl.col("trades_per_week_train") >= 5)
            .sort("sharpe_train", descending=True)
            .head(10)
        )


if __name__ == "__main__":
    main()
