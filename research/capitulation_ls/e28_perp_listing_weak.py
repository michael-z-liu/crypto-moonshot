"""E28: the 'already falling' filter (price at entry <= price at listing) on all new perps."""

from __future__ import annotations

import numpy as np
import polars as pl
from e18_perp_listings import book, perp, trades

for delay, hold, tp, sl in (
    (72, 30, 0.5, 0.5),
    (72, 14, 0.5, 0.5),
    (24, 14, 0.5, 0.5),
    (72, 30, None, 0.5),
):
    d = trades(delay, hold, tp, sl)
    since = []
    for r in d.iter_rows(named=True):
        h, o = perp[r["symbol"]][0], perp[r["symbol"]][1]
        i = np.searchsorted(h, r["t_in"])
        since.append(o[i] / o[0] - 1)
    d = d.with_columns(since=pl.Series(since))
    for lab, f in (
        ("all", pl.lit(True)),
        ("since<=0", pl.col("since") <= 0),
        ("since<=-0.2", pl.col("since") <= -0.2),
    ):
        x = d.filter(f)
        b = book(x, 10)
        yrs = " | ".join(
            f"{r['year']}: {r['trades']:>3}t {r['bp']:+5.0f}bp {r['ret'] * 100:+5.0f}%"
            for r in b.iter_rows(named=True)
        )
        print(
            f"d{delay} h{hold} tp{tp} sl{sl} {lab:11s} n={x.height:>3} {x['net'].mean() * 1e4:+5.0f}bp | {yrs}"  # noqa: E501
        )
