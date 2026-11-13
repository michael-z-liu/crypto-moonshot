"""M5: capitulation rule variants through the portfolio simulation, by split."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common import RESULTS
from portfolio import Exit, load_long, simulate


def rule(
    ev: pl.DataFrame, drop: float, window: str, liq: float, extra: pl.Expr | None
) -> pl.Series:
    m = pl.col("is_spike") & (pl.col(window) <= -drop) & (pl.col("qv_base") >= liq)
    if extra is not None:
        m = m & extra
    return ev.select(m.fill_null(False).alias("m"))["m"]


def main() -> None:
    ev, fwd = load_long()
    exits = [
        Exit(bars=12), Exit(bars=48), Exit(bars=144), Exit(bars=288),
        Exit(bars=48, sl=0.10), Exit(bars=144, sl=0.10), Exit(bars=288, sl=0.15),
        Exit(bars=144, tp=0.10, sl=0.10), Exit(bars=288, tp=0.15, sl=0.15),
        Exit(bars=144, trail=0.05), Exit(bars=288, trail=0.08),
    ]  # fmt: skip
    rows = []
    for drop, window, liq, ex, slots in itertools.product(
        [0.08, 0.10, 0.15], ["ret_4h", "ret_24h"], [5_000, 20_000], exits, [5]
    ):
        m = rule(ev, drop, window, liq, None).to_numpy()
        for split in ("train", "valid"):
            mm = m & (ev["split"] == split).to_numpy()
            idx = np.flatnonzero(mm)
            res = simulate(ev[idx], fwd[idx], ex, slots=slots)
            rows.append({"drop": drop, "window": window, "liq": liq, "exit": str(ex),
                         "slots": slots, "split": split, **res.stats()})  # fmt: skip
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m5_capitulation.csv")
    wide = df.pivot(on="split", index=["drop", "window", "liq", "exit"],
                    values=["trades_per_week", "mean_bp", "cagr", "max_dd", "sharpe"])  # fmt: skip
    with pl.Config(tbl_rows=80, tbl_cols=14, tbl_width_chars=250, fmt_str_lengths=60):
        print(wide.sort("sharpe_train", descending=True).head(40))


if __name__ == "__main__":
    main()
