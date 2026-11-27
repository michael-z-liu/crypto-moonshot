"""E35: stresses on the LQM book (capitulation longs + regime pump shorts + weak listing shorts)."""

from __future__ import annotations

import polars as pl
from e25_combined import book, sleeve_listing_weak, sleeve_long, sleeve_pump_regime


def show(lab: str, parts: list[pl.DataFrame], slots: int = 10) -> None:
    b = book(pl.concat(parts, how="vertical_relaxed"), slots)
    n = int(b["trades"].sum())
    bp = float((b["bp"] * b["trades"]).sum() / n)
    yrs = " | ".join(f"{r['year']}: {r['ret'] * 100:+5.0f}% dd{r['dd'] * 100:+4.0f}" for r in b.iter_rows(named=True))
    print(f"{lab:34s} {bp:+5.0f}bp n={n} min {b['ret'].min() * 100:+4.0f}% | {yrs}")


show("base", [sleeve_long(), sleeve_pump_regime(), sleeve_listing_weak()])
show("2x costs", [sleeve_long(cost=0.008), sleeve_pump_regime(cost=0.006), sleeve_listing_weak(extra_cost=0.003)])
show("one candle late (L, P)", [sleeve_long(delay=1), sleeve_pump_regime(delay=1), sleeve_listing_weak()])
show("2x costs + one candle late", [sleeve_long(cost=0.008, delay=1), sleeve_pump_regime(cost=0.006, delay=1), sleeve_listing_weak(extra_cost=0.003)])
show("L q 0.99 tp 0.15", [sleeve_long(q=0.99, tp=0.15), sleeve_pump_regime(), sleeve_listing_weak()])
show("L volscale5 scores", [sleeve_long(score="volscale5"), sleeve_pump_regime(), sleeve_listing_weak()])
show("L path_all5 scores", [sleeve_long(score="path_all5"), sleeve_pump_regime(), sleeve_listing_weak()])
show("P thr 0.3", [sleeve_long(), sleeve_pump_regime(thr=0.3), sleeve_listing_weak()])
show("P btc7 > 0", [sleeve_long(), sleeve_pump_regime(btc7_min=0.0), sleeve_listing_weak()])
show("P hold 24h", [sleeve_long(), sleeve_pump_regime(hold=1440), sleeve_listing_weak()])
show("no P", [sleeve_long(), sleeve_listing_weak()])
show("no N", [sleeve_long(), sleeve_pump_regime()])
