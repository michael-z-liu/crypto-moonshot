"""B4, exploratory (not pre-registered): pump-start minutes on quiet coins.

1m spikes from m23_quiet_events.parquet on quiet coins (Q7 decile <= 2) where the spike minute
itself rose >= 2% (or >= 5%); entry at the next minute's open; take profit +3/5/10% or a 5m/15m/1h
limit, no stop; the $50 book. Also without 2025-10-10/11.

Run: MOONSHOT_UNIVERSE=all uv run python research/volume_spike/m23b_pumps.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
from common import CACHE, RESULTS
from m23_minute import COST, MIN_NS, book_stats


def main() -> None:
    e = pl.read_parquet(CACHE / "m23_quiet_events.parquet").filter(pl.col("q7_dec") <= 2)
    rows = []
    for floor in (0.02, 0.05):
        d0 = e.filter(pl.col("spike_pct") >= floor)
        tns = d0["t"].dt.epoch("ns").to_numpy()
        lag = d0["lag_m1"].to_numpy()
        for tp, hcol in ((0.05, "hit5_m1"), (0.10, "hit10_m1"), (None, None)):
            for h, w in (("15m", 15), ("1h", 60), ("4h", 240)):
                close = d0[f"c{h}_m1"].to_numpy()
                if tp is None:
                    ret, off = close, np.full(d0.height, w)
                else:
                    hit = d0[hcol].to_numpy()
                    took = (hit >= 0) & (hit < w)
                    ret, off = np.where(took, tp, close), np.where(took, hit + 1, w)
                d = d0.with_columns(
                    r=pl.Series(ret - COST),
                    tin=pl.Series(tns + lag * MIN_NS),
                    tout=pl.Series(tns + (lag + off) * MIN_NS),
                )
                tp_text = "none" if tp is None else f"{tp:.0%}"
                label = f"spike minute >= {floor:.0%} | TP {tp_text} | {h}"
                rows += book_stats(d, "r", "tin", "tout", label)
                rows += book_stats(d, "r", "tin", "tout", label + " | 0.6% cost", 0.006)
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m23b_pumps.csv")
    with pl.Config(
        tbl_rows=80, tbl_cols=12, tbl_width_chars=220, fmt_str_lengths=45, float_precision=0
    ):
        print(
            df.filter(~pl.col("rule").str.contains("0.6%")).pivot(
                on="year", index="rule", values="pnl_usd"
            )
        )
        print(
            df.filter(~pl.col("rule").str.contains("0.6%"), pl.col("year") == 2025).select(
                "rule", "trades", "pnl_usd", "pnl_usd_ex_cascade"
            )
        )
        print(
            df.filter(~pl.col("rule").str.contains("0.6%")).pivot(
                on="year", index="rule", values="trades"
            )
        )


if __name__ == "__main__":
    main()
