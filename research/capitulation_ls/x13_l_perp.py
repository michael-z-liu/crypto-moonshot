"""X13: the long sleeve L on the perp instead of spot, for coins that have both (in-sample).

Same rows (day5 pct >= 0.995, 2022-26), TP 20%, 72h; perp paths, 1x (no liquidation within 72h at
1x), cost 0.3% instead of 0.4%, funding received/paid over the hold. Compared row by row with the
spot version on the rows that have a perp path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, simulate

OUT = Path(__file__).resolve().parent / "results" / "explore"

if __name__ == "__main__":
    m = meta()
    pct = scores("day5")
    yr = m["year"].to_numpy()
    idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.995))
    c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
    rule_s = Rule(hold_min=4320, tp=0.2, slots=100_000, cost=0.004, liq_frac=None)
    rule_p = Rule(hold_min=4320, tp=0.2, slots=100_000, cost=0.003, liq_frac=None)
    sp = simulate(c, rule_s, start_equity=1e12).trades
    pp = simulate(c, rule_p, start_equity=1e12, perp_funding=True, source="perp").trades
    j = sp.select("row", net_spot="net", year=pl.from_epoch("t").dt.year()).join(
        pp.select("row", net_perp="net", fund="fund", kind_p="kind"), on="row", how="inner"
    )
    lines = [
        f"# X13: L on perps vs spot (candidates {len(idx)}; spot trades {sp.height}; "
        f"perp trades {pp.height}; both {j.height})",
        "",
        "| year | n both | spot net bp | perp net bp | perp - spot bp | funding paid bp (neg = received) |",  # noqa: E501
        "|---|---|---|---|---|---|",
    ]
    for (y,), g in j.sort("year").group_by(["year"], maintain_order=True):
        lines.append(
            f"| {y} | {g.height} | {g['net_spot'].mean() * 1e4:+.0f} | {g['net_perp'].mean() * 1e4:+.0f} | "  # noqa: E501
            f"{(g['net_perp'] - g['net_spot']).mean() * 1e4:+.0f} | {g['fund'].mean() * 1e4:+.0f} |"
        )
    lines.append(
        f"| all | {j.height} | {j['net_spot'].mean() * 1e4:+.0f} | {j['net_perp'].mean() * 1e4:+.0f} | "  # noqa: E501
        f"{(j['net_perp'] - j['net_spot']).mean() * 1e4:+.0f} | {j['fund'].mean() * 1e4:+.0f} |"
    )
    txt = "\n".join(lines)
    print(txt)
    (OUT / "l_perp.md").write_text(txt + "\n")
