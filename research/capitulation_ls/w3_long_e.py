"""W3-E: tp20day vs day5 in the slot book; what the book takes."""

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, simulate_yearly

Y = (2022, 2023, 2024, 2025, 2026)
m = meta()
yr = m["year"].to_numpy()
tab = pl.read_parquet("/tmp/w3_tab.parquet").select("row", "o72", "day")
o = np.full(len(m), np.nan)
o[tab["row"].to_numpy()] = tab["o72"].to_numpy()
dayv = m["day"].to_numpy()
for nm in ("day5", "tp20day"):
    s = scores(nm)
    for q in (0.995, 0.98):
        idx = np.flatnonzero((yr >= 2022) & np.isfinite(s) & (s >= q))
        c = pl.DataFrame({"row": idx, "side": 1, "prio": s[idx]})
        b = simulate_yearly(c, Rule(hold_min=4320, tp=0.2, slots=10), years=Y)
        t = b.trades.with_columns(day=pl.col("open_time").dt.date())
        print(
            f"{nm} q{q}: raw rows {len(idx)} (days {len(set(dayv[idx]))}) raw mean o72 {np.nanmean(o[idx]) * 1e4:+.0f}; book n={b.summary['trades']} {b.summary['bp']:+.0f}bp days {t['day'].n_unique()} yearly {[round(v * 100) for v in b.years.sort('year')['ret'].to_list()]}; trades/yr {b.years.sort('year')['trades'].to_list()}; win {(t['net'] > 0).mean():.2f}"  # noqa: E501
        )
T = {}
for nm in ("day5", "tp20day"):
    s = scores(nm)
    idx = np.flatnonzero((yr >= 2022) & np.isfinite(s) & (s >= 0.995))
    c = pl.DataFrame({"row": idx, "side": 1, "prio": s[idx]})
    T[nm] = simulate_yearly(c, Rule(hold_min=4320, tp=0.2, slots=10), years=Y).trades.with_columns(
        day=pl.col("open_time").dt.date()
    )
d5d = set(T["day5"]["day"].to_list())
for nm in ("tp20day", "day5"):
    t = T[nm]
    inn = t.filter(pl.col("day").is_in(list(d5d)))
    out = t.filter(~pl.col("day").is_in(list(d5d)))
    print(
        nm,
        "on day5's trade days:",
        inn.height,
        "trades",
        f"{inn['net'].mean() * 1e4:+.0f}bp",
        "| other days:",
        out.height,
        "trades",
        f"{(out['net'].mean() or 0) * 1e4:+.0f}bp",
    )
t = T["tp20day"].filter(pl.col("day").is_in(list(d5d)))
u = T["day5"]
print(
    " same days only, tp20day coins also taken by day5:",
    t.join(u.select("row"), on="row").height,
    "of",
    t.height,
)
