"""E43: stresses and deflated Sharpe for the LMC book (longs + weak listing shorts + crash shorts)."""  # noqa: E501

from __future__ import annotations

import numpy as np
import polars as pl
from dsr import daily_returns, dsr
from e25_combined import book, sleeve_crash_short, sleeve_listing_weak, sleeve_long

Y = (2022, 2023, 2024, 2025, 2026)


def show(lab: str, parts: list[pl.DataFrame], slots: int = 10) -> np.ndarray:
    tr: list = []
    b = book(pl.concat(parts, how="vertical_relaxed"), slots, trades_out=tr)
    n = int(b["trades"].sum())
    bp = float((b["bp"] * b["trades"]).sum() / n)
    yrs = " | ".join(
        f"{r['year']}: {r['ret'] * 100:+5.0f}% dd{r['dd'] * 100:+4.0f}"
        for r in b.iter_rows(named=True)
    )
    print(f"{lab:30s} {bp:+5.0f}bp n={n} min {b['ret'].min() * 100:+4.0f}% | {yrs}")
    return daily_returns(pl.DataFrame(tr), Y)


r = show("base", [sleeve_long(), sleeve_listing_weak(), sleeve_crash_short()])
show(
    "2x costs",
    [
        sleeve_long(cost=0.008),
        sleeve_listing_weak(extra_cost=0.003),
        sleeve_crash_short(cost=0.006),
    ],
)
show("L one candle late", [sleeve_long(delay=1), sleeve_listing_weak(), sleeve_crash_short()])
show("C calm 2%", [sleeve_long(), sleeve_listing_weak(), sleeve_crash_short(calm=0.02)])
show(
    "C r4 -20% 72h", [sleeve_long(), sleeve_listing_weak(), sleeve_crash_short(r4=-0.2, hold=4320)]
)
show("C cnn <= 0.3", [sleeve_long(), sleeve_listing_weak(), sleeve_crash_short(cnn=0.3)])
show("L volscale5", [sleeve_long(score="volscale5"), sleeve_listing_weak(), sleeve_crash_short()])
show(
    "L path_all5 (+C path_all5)",
    [sleeve_long(score="path_all5"), sleeve_listing_weak(), sleeve_crash_short()],
)
for n in (1000, 5000):
    print(
        "DSR", n, {k: round(float(v), 3) for k, v in dsr(r, n, (0.48 / np.sqrt(365)) ** 2).items()}
    )
