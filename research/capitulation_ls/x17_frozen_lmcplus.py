"""X17: the frozen-settings check (E45, corrected by the audit in a1_e45_fix.py) with the
perp-only long added.

E45/A1 choose every setting on 2022-24 and read 2025-26: L at pct >= 0.99 with TP 10% (72h), C at
r4 <= -10% (72h), M by the 2022-24 grid with the weak filter measured at entry, slots 8/10/15 by
the E45 score. Here the perp-only long uses the same frozen L settings (pct >= 0.99, TP 10%, 72h,
perp 1x, cost 0.3% + funding), and slots are re-chosen on 2022-24 for both books.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import e13_listing_book as e13
import numpy as np
import polars as pl
from a1_e45_fix import m_frame, since_at_entry
from e18_perp_listings import book as lbook
from e25_combined import book, sleeve_crash_short, sleeve_long
from e45_frozen_2024 import SEL, score
from p2_lib import perp_only_spikes, sim_perp, to_book
from sim import Rule

OUT = Path(__file__).resolve().parent / "results" / "explore"


def frozen_m() -> pl.DataFrame:
    best: tuple[float, pl.DataFrame] | None = None
    for delay, hold, tp, sl in itertools.product((24, 72), (7, 14, 30), (None, 0.5), (None, 0.5)):
        d = e13.trades(delay, hold, tp, sl).filter(pl.col("year") >= 2022)
        d = d.with_columns(since=since_at_entry(d)).filter(pl.col("since") <= 0)
        s = lbook(d, 10).filter(pl.col("year").is_in(SEL))
        sc = float(np.log1p(s["ret"].clip(-0.99, None)).mean() - 0.5 * abs(s["dd"].min()))
        if best is None or sc > best[0]:
            best = (sc, d)
    assert best is not None
    return m_frame(best[1])


def pick(parts: list[pl.DataFrame]) -> tuple[int, pl.DataFrame]:
    allb = []
    for slots in (8, 10, 15):
        b = book(pl.concat(parts, how="vertical_relaxed"), slots)
        allb.append((score(b), slots, b))
    allb.sort(key=lambda r: -r[0])
    return allb[0][1], allb[0][2]


if __name__ == "__main__":
    cols = ["symbol", "adv", "t_in", "t_out", "net", "qv", "sleeve", "prio"]
    lg = sleeve_long(q=0.99, tp=0.1, hold=4320).select(cols)
    cs = sleeve_crash_short(r4=-0.1, calm=0.01, cnn=0.5, hold=4320).select(cols)
    mm = frozen_m().select(cols)
    sp = perp_only_spikes().filter(pl.col("perp") & (pl.col("pct") >= 0.99))
    rule = Rule(hold_min=4320, tp=0.1, slots=100_000, cost=0.003, liq_frac=None)
    lp = to_book(sim_perp(sp.with_columns(prio=pl.col("pct")), 1, rule), "LP").select(cols)
    lines = ["# X17: frozen settings (chosen on 2022-24) with and without the perp-only long", "",
             "| book | slots (chosen on 2022-24) | trades | bp/trade | 2022 | 2023 | 2024 | 2025 | 2026 | 2025-26 bp/trade |",  # noqa: E501
             "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, parts in (("LMC frozen", [lg, mm, cs]), ("LMC+ frozen", [lg, mm, cs, lp])):
        slots, b = pick(parts)
        n = int(b["trades"].sum())
        bp = float((b["bp"] * b["trades"]).sum() / n)
        late = b.filter(pl.col("year") >= 2025)
        bpl = float((late["bp"] * late["trades"]).sum() / late["trades"].sum())
        yrs = " | ".join(
            f"{r['ret'] * 100:+.0f}% (dd {r['dd'] * 100:.0f})" for r in b.iter_rows(named=True)
        )
        row = f"| {name} | {slots} | {n} | {bp:+.0f} | {yrs} | {bpl:+.0f} |"
        lines.append(row)
        print(row, flush=True)
    lines += ["", f"Perp-only long at the frozen settings: {lp.height} trades, mean net "
              f"{lp['net'].mean() * 100:+.1f}% (2025-26: "
              f"{lp.filter(pl.from_epoch('t_in').dt.year() >= 2025)['net'].mean() * 100:+.1f}%)."]  # fmt: skip  # noqa: E501
    (OUT / "frozen_lmcplus.md").write_text("\n".join(lines) + "\n")
