"""E01: how large are the moves after each event type, long and short, 1h to 72h?"""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import SPOT_COST, fwd2, fwd72, meta, scores

m = meta()
f2, f72 = fwd2(), fwd72()
pct = scores("day5")
r4, btcv, btc4 = m["ret_4h"].to_numpy(), m["btc_vol_mult"].to_numpy(), m["btc_ret_4h"].to_numpy()
yr = m["year"].to_numpy()


def stats(name: str, mask: np.ndarray, side: int) -> dict[str, object]:
    idx = np.flatnonzero(mask)
    a = np.asarray(f2[idx]).astype(np.float64)
    b = np.asarray(f72[idx]).astype(np.float64)
    row: dict[str, object] = {"set": name, "side": "long" if side > 0 else "short", "n": len(idx)}
    for lab, v in (("1h", a[:, 2, 11]), ("2h", a[:, 2, 23]), ("4h", a[:, 2, 47]),
                   ("8h", a[:, 2, 95]), ("24h", b[:, 2, 23]), ("72h", b[:, 2, 71])):  # fmt: skip
        row[lab] = round(float(np.nanmean(side * v - SPOT_COST)) * 1e4)
    fav = b[:, 0, :24].max(1) if side > 0 else -b[:, 1, :24].min(1)
    adv = -b[:, 1, :24].min(1) if side > 0 else b[:, 0, :24].max(1)
    row["fav24_med"] = round(float(np.nanmedian(fav)) * 100, 1)
    row["fav24>10%"] = round(float(np.nanmean(fav > 0.10)) * 100)
    row["adv24_med"] = round(float(np.nanmedian(adv)) * 100, 1)
    for y in (2022, 2023, 2024, 2025, 2026):
        s = idx[yr[idx] == y]
        row[f"24h_{y}"] = (
            round(float(np.nanmean(side * np.asarray(f72[s, 2, 23]) - SPOT_COST)) * 1e4)
            if len(s)
            else None
        )
    return row


sets = [
    ("all spikes", np.ones(len(m), bool), 1),
    ("crash r4<=-10%", r4 <= -0.10, 1),
    ("crash r4<=-20%", r4 <= -0.20, 1),
    ("crash r4<=-30%", r4 <= -0.30, 1),
    ("F2 r4<=-10% btcv>=6", (r4 <= -0.10) & (btcv >= 6), 1),
    ("r4<=-20% btc4<=-5%", (r4 <= -0.20) & (btc4 <= -0.05), 1),
    ("day5 pct>=0.98", pct >= 0.98, 1),
    ("day5 pct>=0.995", pct >= 0.995, 1),
    ("day5 pct>=0.999", pct >= 0.999, 1),
    ("pump r4>=+20%", r4 >= 0.20, -1),
    ("pump r4>=+40%", r4 >= 0.40, -1),
    ("pump r4>=+80%", r4 >= 0.80, -1),
    ("pump r4>=+20% btc calm", (r4 >= 0.20) & (np.abs(btc4) < 0.01), -1),
    ("day5 pct<=0.02 (short)", pct <= 0.02, -1),
    ("day5 pct<=0.002 (short)", pct <= 0.002, -1),
]
out = pl.DataFrame([stats(*s) for s in sets])
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
print(out)
out.write_csv(__file__.replace("e01_landscape.py", "results/e01_landscape.csv"))
