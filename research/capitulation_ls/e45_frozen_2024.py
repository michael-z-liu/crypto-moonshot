"""E45: pseudo out-of-sample. Each sleeve's settings are chosen on 2022-24 only (best mean of
yearly log returns minus half the worst drawdown, per sleeve on its own 10-slot book), the book
is assembled from those choices with slots also chosen on 2022-24, then read on 2025-26.
2025-26 was seen by earlier studies and by this one, so this is a check of choice, not a test."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from e25_combined import book, sleeve_crash_short, sleeve_long

SEL = (2022, 2023, 2024)


def score(b: pl.DataFrame) -> float:
    s = b.filter(pl.col("year").is_in(SEL))
    if s.height < len(SEL):
        return -9.0
    return float(np.log1p(s["ret"].clip(-0.99, None)).mean() - 0.5 * abs(s["dd"].min()))


def best(name: str, gen, grid) -> tuple[dict, pl.DataFrame]:
    res = []
    for cfg in grid:
        t = gen(**cfg)
        b = book(t, 10)
        res.append((score(b), cfg, t))
    res.sort(key=lambda r: -r[0])
    print(f"{name}: best on 2022-24 {res[0][1]} score {res[0][0]:.3f}")
    return res[0][1], res[0][2]


Lc, L = best("L", sleeve_long, [dict(q=q, tp=tp, hold=h) for q, tp, h in itertools.product((0.99, 0.995, 0.997), (0.1, 0.15, 0.2, 0.3), (1440, 4320))])
Cc, C = best("C", sleeve_crash_short, [dict(r4=r, calm=c, cnn=n, hold=h) for r, c, n, h in itertools.product((-0.1, -0.15, -0.2), (0.01, 0.02), (0.3, 0.5, 1.0), (1440, 4320))])
# M: the weak-listing sleeve's settings from E13's grid restricted to 2022-24.
import e13_listing_book as e13
from e18_perp_listings import book as lbook

mres = []
x27 = pl.read_parquet("/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/e27_listings.parquet")
weak = set(zip(x27.filter(pl.col("since_list") <= 0)["symbol"], strict=True))
for delay, hold, tp, sl in itertools.product((24, 72), (7, 14, 30), (None, 0.5), (None, 0.5)):
    d = e13.trades(delay, hold, tp, sl).filter(pl.col("year") >= 2022)
    d = d.filter(pl.col("symbol").is_in([s for (s,) in weak]))
    b = lbook(d, 10)
    s = b.filter(pl.col("year").is_in(SEL))
    sc = float(np.log1p(s["ret"].clip(-0.99, None)).mean() - 0.5 * abs(s["dd"].min()))
    mres.append((sc, dict(delay=delay, hold=hold, tp=tp, sl=sl), d))
mres.sort(key=lambda r: -r[0])
print(f"M: best on 2022-24 {mres[0][1]} score {mres[0][0]:.3f} (weak filter fixed from E27)")
M = mres[0][2].select("symbol", adv=pl.lit(0.0), t_in="t_in", t_out="t_out", net="net", qv="qv", sleeve=pl.lit("N"), prio=pl.lit(1.0))
allb = []
for slots in (8, 10, 15):
    b = book(pl.concat([L, M, C], how="vertical_relaxed"), slots)
    allb.append((score(b), slots, b))
allb.sort(key=lambda r: -r[0])
sc, slots, b = allb[0]
print(f"slots chosen on 2022-24: {slots}")
for r in b.iter_rows(named=True):
    print(f"  {r['year']}: {r['trades']:>3}t {r['bp']:+5.0f}bp {r['ret'] * 100:+5.0f}% dd {r['dd'] * 100:+4.0f}%{'   <- read after the choice' if r['year'] >= 2025 else ''}")
