"""E47: short perps after extreme positive funding (crowded longs) / long after extreme negative
funding (crowded shorts). Entry at the next hour's open after the funding print, hold 24/72h,
liquidation at +100% against, 0.3% cost plus funding over the hold. One entry per coin per 3 days."""  # noqa: E501

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN
from futures_ls import funding, funding_paid, spot_name

P = pl.read_parquet(
    OWN / "futures" / "perp_1h.parquet",
    columns=["symbol", "open_time", "open", "high", "low", "close", "quote_volume"],
)
P = P.with_columns(h=pl.col("open_time").dt.epoch("s")).sort("symbol", "h")
perp = {
    k[0]: tuple(g[c].to_numpy() for c in ("h", "open", "high", "low", "close", "quote_volume"))
    for k, g in P.group_by(["symbol"], maintain_order=True)
}
f = funding().filter(pl.col("time").dt.year() >= 2021).with_columns(t=pl.col("time").dt.epoch("s"))
for side, lab, cond in ((-1, "short after rate >= +0.10%", pl.col("rate") >= 0.001), (-1, "short after rate >= +0.30%", pl.col("rate") >= 0.003),  # noqa: E501
                        (1, "long after rate <= -0.30%", pl.col("rate") <= -0.003), (1, "long after rate <= -1.00%", pl.col("rate") <= -0.01)):  # fmt: skip  # noqa: E501
    ev = f.filter(cond).sort("symbol", "t")
    rows, last = [], {}
    for r in ev.iter_rows(named=True):
        if r["t"] - last.get(r["symbol"], -1e18) < 3 * 86400:
            continue
        a = perp.get(r["symbol"])
        if a is None:
            continue
        h, o, hi, lo, c, qv = a
        i = np.searchsorted(h, r["t"] + 1)
        if i >= len(h) - 1 or i < 1 or qv[i - 1] < 50_000:
            continue
        last[r["symbol"]] = r["t"]
        for hold in (24, 72):
            j = min(i + hold - 1, len(h) - 1)
            adv = (
                (hi[i : j + 1].max() / o[i] - 1) if side < 0 else -(lo[i : j + 1].min() / o[i] - 1)
            )
            ret = -1.0 if adv >= 0.99 else side * (c[j] / o[i] - 1)
            rows.append(
                {
                    "hold": hold,
                    "symbol": r["symbol"],
                    "t_in": int(h[i]),
                    "t_out": int(h[j]) + 3600,
                    "ret": ret,
                    "year": int(
                        np.datetime64(int(h[i]), "s").astype("datetime64[Y]").astype(int) + 1970
                    ),
                }
            )
    d = pl.DataFrame(rows)
    fp = funding_paid(
        np.array([spot_name(s) for s in d["symbol"].to_list()]),
        d["t_in"].to_numpy(),
        d["t_out"].to_numpy(),
        np.full(d.height, side),
    )
    d = d.with_columns(net=pl.col("ret") - 0.003 - pl.Series(fp))
    for hold in (24, 72):
        x = d.filter(pl.col("hold") == hold)
        print(
            f"{lab:28s} {hold}h n={x.height:>5} {x['net'].mean() * 1e4:+5.0f}bp med {x['net'].median() * 1e4:+5.0f} |",  # noqa: E501
            x.group_by("year")
            .agg(bp=(pl.col("net").mean() * 1e4).round(), n=pl.len())
            .sort("year")
            .to_dicts(),
        )
