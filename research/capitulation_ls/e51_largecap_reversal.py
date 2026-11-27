"""E51 (new family): weekly cross-sectional reversal among large caps. Each Monday 00:00 UTC rank
30 liquid large coins by their past-7-day return; buy the 5 worst (and, separately, short the 5
best) for 7 days; 0.4% round trip. Compared with the equal-weight basket of all 30."""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
from common_ls import CACHE_ALL

os.environ["MOONSHOT_UNIVERSE"] = "all"
sys.path.insert(0, str(CACHE_ALL.parent))
from common import load

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",  # noqa: E501
         "DOTUSDT", "LTCUSDT", "TRXUSDT", "BCHUSDT", "NEARUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT", "FILUSDT",  # noqa: E501
         "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT", "AAVEUSDT", "XLMUSDT", "HBARUSDT", "ICPUSDT",  # noqa: E501
         "VETUSDT", "ALGOUSDT", "SANDUSDT"]  # fmt: skip
closes = {}
for s in COINS:
    d = load(s).filter(pl.col("open_time").dt.hour().eq(0) & pl.col("open_time").dt.minute().eq(0))
    closes[s] = dict(zip(d["open_time"].dt.epoch("s").to_list(), d["open"].to_list(), strict=True))
mondays = pl.datetime_range(
    pl.datetime(2021, 1, 4, time_zone="UTC"),
    pl.datetime(2026, 10, 19, time_zone="UTC"),
    "7d",
    eager=True,
)
rows = []
for mon in mondays:
    t = int(mon.timestamp())
    past, fut = {}, {}
    for s, c in closes.items():
        a, b, f = c.get(t - 7 * 86400), c.get(t), c.get(t + 7 * 86400)
        if a and b and f:
            past[s], fut[s] = b / a - 1, f / b - 1
    if len(past) < 15:
        continue
    order = sorted(past, key=past.get)
    lo, hi = order[:5], order[-5:]
    rows.append({"year": mon.year, "losers": np.mean([fut[s] for s in lo]) - 0.004, "winners_short": -np.mean([fut[s] for s in hi]) - 0.004,  # noqa: E501
                 "basket": np.mean(list(fut.values())) - 0.004})  # fmt: skip
d = pl.DataFrame(rows)
print(
    d.group_by("year")
    .agg((pl.col(c).mean() * 1e4).round().alias(c) for c in ("losers", "winners_short", "basket"))
    .sort("year")
)
print(
    "all weeks (bp/week):",
    {c: round(float(d[c].mean()) * 1e4) for c in ("losers", "winners_short", "basket")},
    "weeks",
    d.height,
)
print(
    "losers minus basket:",
    round(float((d["losers"] - d["basket"]).mean()) * 1e4),
    "bp/week; t =",
    round(
        float(
            (d["losers"] - d["basket"]).mean()
            / (d["losers"] - d["basket"]).std()
            * np.sqrt(d.height)
        ),
        2,
    ),
)
