"""E18: short every new USDT perp after its listing (perp 1h candles, delisted perps included).

Listing = the perp's first hourly candle (2021 onward). Entry: open of the hour at listing + delay
(the perp must have traded >= 50k USDT in the previous hour). Exit: hold N days at the close, a
take profit, a stop, or liquidation at +100% (isolated 1x), adverse extreme first within an hour.
Funding paid. Book: K equal slots of equity, each year starts at $10k, stake <= 0.5% of the
perp's 24h quote volume at entry.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import polars as pl
from common_ls import OWN
from futures_ls import funding_paid, spot_name

P = pl.read_parquet(OWN / "futures" / "perp_1h.parquet").with_columns(h=pl.col("open_time").dt.epoch("s")).sort("symbol", "h")
perp = {k[0]: tuple(g[c].to_numpy() for c in ("h", "open", "high", "low", "close", "quote_volume"))
        for k, g in P.group_by(["symbol"], maintain_order=True)}  # fmt: skip
first = {s: int(a[0][0]) for s, a in perp.items()}
COST = 0.003


def trades(delay_h: int, hold_d: int, tp: float | None, sl: float | None, min_qv: float = 50_000) -> pl.DataFrame:
    out = []
    for sym, a in perp.items():
        h, o, hi, lo, c, qv = a
        t0 = first[sym]
        if t0 < 1609459200:  # listed before 2021
            continue
        i = np.searchsorted(h, t0 + delay_h * 3600)
        if i >= len(h) - 1 or i < 1 or qv[i - 1] < min_qv or h[i] - (t0 + delay_h * 3600) > 6 * 3600:
            continue
        j_end = min(np.searchsorted(h, h[i] + hold_d * 86400), len(h) - 1)
        e = o[i]
        up = hi[i:j_end + 1] / e - 1
        dn = lo[i:j_end + 1] / e - 1
        ret, k_exit = -(c[j_end] / e - 1), j_end
        stop = min(sl, 1.0) if sl is not None else 1.0
        hs = np.flatnonzero(up >= stop)
        ht = np.flatnonzero(-dn >= tp) if tp is not None else np.array([], int)
        ks = hs[0] if len(hs) else 10**9
        kt = ht[0] if len(ht) else 10**9
        if ks <= kt and ks < 10**9:
            ret, k_exit = -stop - (0.0 if stop >= 1.0 else 0.02), i + ks
        elif kt < 10**9:
            ret, k_exit = tp, i + kt
        out.append({"symbol": sym, "t_in": int(h[i]), "t_out": int(h[k_exit]) + 3600, "ret": float(ret),
                    "year": int(np.datetime64(int(t0), "s").astype("datetime64[Y]").astype(int) + 1970),
                    "qv": float(qv[max(0, i - 24):i].sum())})  # fmt: skip
    d = pl.DataFrame(out)
    f = funding_paid(np.array([spot_name(s) for s in d["symbol"].to_list()]), d["t_in"].to_numpy(), d["t_out"].to_numpy(), -np.ones(d.height))
    return d.with_columns(net=(pl.col("ret") - COST - pl.Series(f)).clip(-1.0, None), fund=pl.Series(f))


def book(d: pl.DataFrame, slots: int) -> pl.DataFrame:
    res = []
    for (y,), g in d.sort("t_in").group_by(["year"], maintain_order=True):
        eq, cash, pos, nets, peak, dd = 10_000.0, 10_000.0, [], [], 10_000.0, 0.0
        for r in g.iter_rows(named=True):
            keep = []
            for p in pos:
                if p[0] <= r["t_in"]:
                    cash += p[1] * (1 + p[2])
                    eq += p[1] * p[2]
                    peak = max(peak, eq)
                    dd = min(dd, eq / peak - 1)
                else:
                    keep.append(p)
            pos = keep
            if len(pos) >= slots:
                continue
            stake = min(eq / slots, cash, r["qv"] * 0.005)
            if stake < 1:
                continue
            cash -= stake
            pos.append((r["t_out"], stake, r["net"]))
            nets.append(r["net"])
        for p in sorted(pos):
            eq += p[1] * p[2]
            peak = max(peak, eq)
            dd = min(dd, eq / peak - 1)
        res.append({"year": y, "trades": len(nets), "bp": float(np.mean(nets)) * 1e4 if nets else 0.0,
                    "med": float(np.median(nets)) * 1e4 if nets else 0.0, "ret": eq / 10_000 - 1, "dd": dd})  # fmt: skip
    return pl.DataFrame(res).sort("year")


if __name__ == "__main__":
    rows = []
    for delay, hold, tp, sl, slots in itertools.product((1, 24, 72, 168), (3, 7, 14, 30), (None, 0.3, 0.5, 0.7), (None, 0.5), (5, 10, 20)):
        d = trades(delay, hold, tp, sl)
        b = book(d, slots)
        rows.append({"delay": delay, "hold": hold, "tp": tp, "sl": sl, "slots": slots, "n_all": d.height, "bp_all": float(d["net"].mean()) * 1e4,
                     "min_ret": float(b.filter(pl.col("year") >= 2022)["ret"].min()), "med_ret": float(b["ret"].median()),
                     "worst_dd": float(b["dd"].min()), **{f"r{r['year']}": round(r["ret"] * 100) for r in b.iter_rows(named=True)},
                     **{f"n{r['year']}": r["trades"] for r in b.iter_rows(named=True)}})  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_csv("results/e18_perp_listings.csv")
    pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_rows(30)
    print(out.sort("min_ret", descending=True).head(20))
    print(out.group_by("hold").agg(pl.col("bp_all").mean(), pl.col("min_ret").median(), pl.col("med_ret").median()).sort("hold"))
    print(out.group_by("delay").agg(pl.col("bp_all").mean(), pl.col("min_ret").median(), pl.col("med_ret").median()).sort("delay"))
