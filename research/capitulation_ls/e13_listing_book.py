"""E13: book for shorting new listings on the perp (1h perp candles), one day after spot listing.

Entry: perp open of the first full hour >= listing + delay. Exit: N days later at the close, or at
a take profit / stop on hourly highs and lows (adverse first). At 1x a +100% move liquidates.
Slots: K equal shares of equity (each year starts at $10k), funding paid per position.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common_ls import OWN
from futures_ls import funding_paid, spot_name

L = pl.read_parquet(OWN / "e12_listings.parquet")
P = pl.read_parquet(OWN / "futures" / "perp_1h.parquet").with_columns(
    spot=pl.col("symbol").map_elements(spot_name, return_dtype=pl.Utf8), h=pl.col("open_time").dt.epoch("s"))
perp = {k[0]: (g["h"].to_numpy(), g["open"].to_numpy(), g["high"].to_numpy(), g["low"].to_numpy(), g["close"].to_numpy(), g["quote_volume"].to_numpy())
        for k, g in P.sort("h").group_by(["spot"], maintain_order=True)}  # fmt: skip
COST = 0.003


def trades(delay_h: int, hold_d: int, tp: float | None, sl: float | None) -> pl.DataFrame:
    out = []
    for r in L.iter_rows(named=True):
        a = perp.get(r["symbol"])
        if a is None:
            continue
        h, o, hi, lo, c, qv = a
        t_e = (int(r["listed"].timestamp()) // 3600 + delay_h) * 3600
        i = np.searchsorted(h, t_e)
        if i >= len(h) or h[i] - t_e > 6 * 3600 or i < 2 or qv[i - 1] < 50_000:
            continue
        j_end = min(np.searchsorted(h, h[i] + hold_d * 86400), len(h) - 1)
        e = o[i]
        up = hi[i:j_end + 1] / e - 1  # adverse for a short
        dn = lo[i:j_end + 1] / e - 1
        ret, k_exit = -(c[j_end] / e - 1), j_end
        lvl_liq = 1.0
        stop = sl if sl is not None else lvl_liq
        hit_sl = np.flatnonzero(up >= stop)
        hit_tp = np.flatnonzero(-dn >= tp) if tp is not None else np.array([], int)
        ks = hit_sl[0] if len(hit_sl) else 10**9
        kt = hit_tp[0] if len(hit_tp) else 10**9
        if ks <= kt and ks < 10**9:
            ret, k_exit = -stop, i + ks
        elif kt < 10**9:
            ret, k_exit = tp, i + kt
        out.append({"symbol": r["symbol"], "t_in": int(h[i]), "t_out": int(h[k_exit]) + 3600, "ret": float(ret),
                    "year": r["listed"].year, "qv": float(qv[i - 1])})  # fmt: skip
    d = pl.DataFrame(out)
    f = funding_paid(d["symbol"].to_numpy(), d["t_in"].to_numpy(), d["t_out"].to_numpy(), -np.ones(d.height))
    return d.with_columns(net=pl.col("ret") - COST - pl.Series(f), fund=pl.Series(f))


def book(d: pl.DataFrame, slots: int) -> pl.DataFrame:
    res = []
    for y, g in d.sort("t_in").group_by(["year"], maintain_order=True):
        eq, cash, pos, n, pnl_list = 10_000.0, 10_000.0, [], 0, []
        for r in g.iter_rows(named=True):
            keep = []
            for p in pos:
                if p[0] <= r["t_in"]:
                    cash += p[1] * (1 + p[2]); eq += p[1] * p[2]
                else:
                    keep.append(p)
            pos = keep
            if len(pos) >= slots:
                continue
            stake = min(eq / slots, cash, r["qv"] * 24 * 0.005)
            cash -= stake
            pos.append((r["t_out"], stake, r["net"]))
            n += 1; pnl_list.append(r["net"])
        for p in pos:
            eq += p[1] * p[2]
        res.append({"year": y[0], "trades": n, "bp": float(np.mean(pnl_list)) * 1e4 if pnl_list else 0.0, "ret": eq / 10_000 - 1})
    return pl.DataFrame(res).sort("year")


for delay, hold, tp, sl, slots in itertools.product((24, 72), (3, 7, 14, 30), (None, 0.5), (None, 0.5), (5, 10)):
    d = trades(delay, hold, tp, sl)
    b = book(d, slots)
    yrs = " | ".join(f"{r['year']}: {r['trades']:>3}t {r['bp']:+6.0f}bp {r['ret'] * 100:+5.0f}%" for r in b.iter_rows(named=True))
    print(f"delay {delay:>2}h hold {hold:>2}d tp {tp} sl {sl} slots {slots:>2} | all {d['net'].mean() * 1e4:+5.0f}bp n={d.height} | {yrs}")
