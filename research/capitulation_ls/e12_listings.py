"""E12: do new Binance spot listings fall afterwards? Short at listing + delay, hold N days."""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
from common_ls import CACHE_ALL, OWN

os.environ["MOONSHOT_UNIVERSE"] = "all"
sys.path.insert(0, str(CACHE_ALL.parent))
from common import ARCHIVE_5M, KLINES_5M, universe
from futures_ls import perp_active

act = perp_active()
rows = []
for sym in universe():
    src = ARCHIVE_5M / f"symbol={sym}"
    src = src if src.exists() else KLINES_5M / f"symbol={sym}"
    files = sorted(src.glob("*.parquet"))
    if not files:
        continue
    df = pl.concat(
        [
            pl.read_parquet(f, columns=["open_time", "open", "close", "quote_volume"])
            for f in files[:3]
        ]
    ).sort("open_time")
    t0 = df["open_time"][0]
    if t0.year < 2021:
        continue
    t = df["open_time"].dt.epoch("s").to_numpy()
    c = df["close"].to_numpy()
    o = df["open"].to_numpy()
    rec = {"symbol": sym, "listed": t0, "perp_at_list": False}
    a = act.get(sym)
    if a is not None:
        rec["perp_at_list"] = bool(a[0][0] <= t[0] + 86400)
    for dl, dlab in ((1, "1h"), (24, "1d"), (72, "3d")):
        i = np.searchsorted(t, t[0] + dl * 3600)
        if i >= len(t):
            continue
        rec[f"qv_{dlab}"] = float(df["quote_volume"][:i].sum())
        for hd in (1, 3, 7, 14, 30):
            j = np.searchsorted(t, t[i] + hd * 86400)
            if j < len(t):
                rec[f"s{dlab}_{hd}d"] = float(-(c[j] / o[i] - 1))
    rows.append(rec)
d = pl.DataFrame(rows).with_columns(year=pl.col("listed").dt.year())
d.write_parquet(OWN / "e12_listings.parquet")
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
cols = [c for c in d.columns if c[0] == "s" and c[1] in "13" and "_" in c]
print(d.select(pl.len(), *[(pl.col(c).mean() * 1e4).round().alias(c) for c in cols]))
print(
    d.group_by("year")
    .agg(
        pl.len(),
        *[
            (pl.col(c).mean() * 1e4).round().alias(c)
            for c in ["s1d_7d", "s1d_14d", "s1d_30d", "s3d_14d", "s3d_30d"]
        ],
        med30=(pl.col("s1d_30d").median() * 1e4).round(),
    )
    .sort("year")
)
print(
    d.group_by("perp_at_list").agg(
        pl.len(), *[(pl.col(c).mean() * 1e4).round().alias(c) for c in ["s1d_7d", "s1d_30d"]]
    )
)
