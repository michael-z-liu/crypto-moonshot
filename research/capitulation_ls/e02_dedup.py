"""E02: E01 with one entry per coin per event (first qualifying spike, then 24h lockout), medians."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from common_ls import SPOT_COST, first_per_event, fwd72, meta, scores

m = meta()
f72 = fwd72()
pct = scores("day5")
r4, btcv, btc4 = m["ret_4h"].to_numpy(), m["btc_vol_mult"].to_numpy(), m["btc_ret_4h"].to_numpy()
qv = m["qv_base"].to_numpy()
t = m["open_time"].dt.epoch("s").to_numpy()
sym = m["symbol"].to_numpy()
yr = m["year"].to_numpy()


def stats(name: str, mask: np.ndarray, side: int) -> dict[str, object]:
    idx = first_per_event(mask)
    b = np.asarray(f72[idx]).astype(np.float64)
    row: dict[str, object] = {"set": name, "side": side, "n": len(idx)}
    for lab, h in (("1h", 0), ("4h", 3), ("24h", 23), ("72h", 71)):
        v = side * b[:, 2, h] - SPOT_COST
        row[lab] = round(float(np.nanmean(v)) * 1e4)
        row[lab + "_med"] = round(float(np.nanmedian(v)) * 1e4)
    v = side * b[:, 2, 23] - SPOT_COST
    row["24h_trim1%"] = round(float(np.nanmean(np.clip(v, *np.nanpercentile(v, [1, 99])))) * 1e4)
    adv = -b[:, 1, :24].min(1) if side > 0 else b[:, 0, :24].max(1)
    row["adv24>50%"] = round(float(np.nanmean(adv > 0.5)) * 100, 1)
    for y in (2021, 2022, 2023, 2024, 2025, 2026):
        s = np.flatnonzero(yr[idx] == y)
        row[f"{y}"] = f"{round(float(np.nanmean(v[s])) * 1e4)}/{len(s)}" if len(s) else ""
    return row


liq = qv >= 20_000
sets = [
    ("crash r4<=-10%", r4 <= -0.10, 1),
    ("crash r4<=-20%", r4 <= -0.20, 1),
    ("crash r4<=-30%", r4 <= -0.30, 1),
    ("F2 r4<=-10% btcv>=6 liq", (r4 <= -0.10) & (btcv >= 6) & liq, 1),
    ("day5 pct>=0.98", pct >= 0.98, 1),
    ("day5 pct>=0.995", pct >= 0.995, 1),
    ("pump r4>=+20%", r4 >= 0.20, -1),
    ("pump r4>=+40%", r4 >= 0.40, -1),
    ("pump r4>=+80%", r4 >= 0.80, -1),
    ("pump r4>=+20% liq", (r4 >= 0.20) & liq, -1),
    ("pump r4>=+40% liq", (r4 >= 0.40) & liq, -1),
    ("day5 pct<=0.02", pct <= 0.02, -1),
    ("day5 pct<=0.002", pct <= 0.002, -1),
]
out = pl.DataFrame([stats(*s) for s in sets])
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(300)
pl.Config.set_tbl_rows(40)
print(out)
out.write_csv("results/e02_dedup.csv")
