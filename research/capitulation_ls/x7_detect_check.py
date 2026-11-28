"""X7a-3: why hourly-detected 4h pumps short worse than F. (1) Re-simulate F's own rows (d01/x6
pool) on hourly perp bars with the x7 simulator; (2) overlap of hourly gated 4h-40% shorts (2022+)
with F rows (same coin, signal hour within +-6 h) and the mean net of each part."""

from __future__ import annotations

import polars as pl
import x7_second_pump as sp
from x6_lib import pool, trades
from x7_lib import PERP_COST, T0, ep, fpaid, perp, simulate


def main() -> None:
    P = perp()
    pcol = {s: j for j, s in enumerate(P["syms"])}
    F = trades(pool()).with_columns(h=(pl.col("t") - T0) // 3600)
    mine: list[float | None] = []
    for r in F.iter_rows(named=True):
        pc, j = pcol.get(r["symbol"], -1), int(r["h"])
        s = (
            simulate(
                P["open"][:, pc],
                P["high"][:, pc],
                P["low"][:, pc],
                P["close"][:, pc],
                j + 1,
                72,
                -1,
                None,
                None,
            )
            if pc >= 0
            else None
        )
        mine.append(
            None
            if s is None
            else max(
                s["ret"] - PERP_COST - fpaid(r["symbol"], ep(j + 1), ep(s["k_exit"] + 1)), -1.0
            )
        )
    F = F.with_columns(hourly=pl.Series(mine, dtype=pl.Float64))
    print(
        "F rows re-simulated hourly:",
        F.select(n=pl.len(), F_net=pl.col("net").mean(), hourly_net=pl.col("hourly").mean()),
    )
    ev = sp.events(4, 0.4).filter(pl.col("year") >= 2022)
    t = sp.dedup(sp.trades(ev, -1, 72, None, None, True))
    fk: dict[str, list[int]] = {}
    for r in F.iter_rows(named=True):
        fk.setdefault(r["symbol"], []).append(int(r["h"]))
    t = t.with_columns(
        inF=pl.Series(
            [
                any(abs(h - x) <= 6 for x in fk.get(s, []))
                for s, h in zip(t["symbol"], t["h"], strict=True)
            ]
        )
    )
    print(
        t.group_by("inF", repeat=pl.col("n_prior") >= 1)
        .agg(n=pl.len(), net=pl.col("net").mean(), liq=(pl.col("kind") == "liq").mean())
        .sort("inF", "repeat")
    )
    mk: dict[str, list[int]] = {}
    for r in t.iter_rows(named=True):
        mk.setdefault(r["symbol"], []).append(int(r["h"]))
    F = F.with_columns(
        inHourly=pl.Series(
            [
                any(abs(h - x) <= 6 for x in mk.get(s, []))
                for s, h in zip(F["symbol"], F["h"], strict=True)
            ]
        )
    )
    print(F.group_by("inHourly").agg(n=pl.len(), net=pl.col("net").mean()))


if __name__ == "__main__":
    main()
