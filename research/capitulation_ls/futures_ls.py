"""Perp availability and funding for the short side (M1 data from fetch_futures.py)."""

from __future__ import annotations

from functools import cache

import numpy as np
import polars as pl
from common_ls import OWN

FUT = OWN / "futures"
PREFIXES = ("1000000", "1000", "1M")


def spot_name(perp: str) -> str:
    for p in PREFIXES:
        if perp.startswith(p) and len(perp) > len(p) + 4:
            return perp[len(p):]
    return perp


@cache
def funding() -> pl.DataFrame:
    f = pl.read_parquet(FUT / "funding.parquet")
    return f.with_columns(spot=pl.col("symbol").map_elements(spot_name, return_dtype=pl.Utf8))


@cache
def funding_cum() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """spot symbol -> (funding times in epoch s, cumulative rate after each time)."""
    out = {}
    for (sym,), g in funding().sort("time").group_by(["spot"], maintain_order=True):
        t = g["time"].dt.epoch("s").to_numpy()
        out[str(sym)] = (t, np.cumsum(g["rate"].to_numpy()))
    return out


@cache
def perp_window() -> dict[str, tuple[int, int]]:
    """spot symbol -> (first, last) epoch s of the perp's funding history (its trading window)."""
    g = funding().group_by("spot").agg(a=pl.col("time").min(), b=pl.col("time").max())
    return {r["spot"]: (int(r["a"].timestamp()), int(r["b"].timestamp())) for r in g.iter_rows(named=True)}


@cache
def perp_active() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """spot symbol -> (hour starts in epoch s, perp quote volume that hour), from perp_1h."""
    p = pl.read_parquet(FUT / "perp_1h.parquet", columns=["symbol", "open_time", "quote_volume", "high", "low"])
    p = p.with_columns(spot=pl.col("symbol").map_elements(spot_name, return_dtype=pl.Utf8))
    # A frozen perp prints flat candles: count only hours that traded and moved.
    p = p.with_columns(qv=pl.when(pl.col("high") > pl.col("low")).then(pl.col("quote_volume")).otherwise(0.0))
    out = {}
    for (sym,), g in p.sort("open_time").group_by(["spot"], maintain_order=True):
        out[str(sym)] = (g["open_time"].dt.epoch("s").to_numpy(), g["qv"].to_numpy())
    return out


def has_perp(symbols: np.ndarray, t: np.ndarray, min_qv: float = 50_000, warmup_s: int = 86400) -> np.ndarray:
    """True if the perp traded at least `min_qv` USDT (with a moving price) in each of the last
    two closed hours before `t`, and has a history of at least `warmup_s`."""
    act = perp_active()
    out = np.zeros(len(symbols), dtype=bool)
    for i, (s, ti) in enumerate(zip(symbols, t, strict=True)):
        a = act.get(str(s))
        if a is None:
            continue
        hours, qv = a
        k = np.searchsorted(hours, (ti // 3600) * 3600) - 1  # last closed hour
        if k < 1 or hours[0] + warmup_s > ti or hours[k] < ti - 2 * 3600:
            continue
        out[i] = qv[k] >= min_qv and qv[k - 1] >= min_qv
    return out


def funding_paid(symbols: np.ndarray, t_in: np.ndarray, t_out: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Funding paid by each position as a fraction of notional (negative = received).

    A positive rate means longs pay shorts, so a long pays +rate and a short pays -rate.
    """
    cum = funding_cum()
    out = np.zeros(len(symbols))
    for i, s in enumerate(symbols):
        c = cum.get(str(s))
        if c is None:
            continue
        t, cs = c
        a = np.searchsorted(t, t_in[i], side="right") - 1
        b = np.searchsorted(t, t_out[i], side="right") - 1
        va = cs[a] if a >= 0 else 0.0
        vb = cs[b] if b >= 0 else 0.0
        out[i] = side[i] * (vb - va)
    return out
