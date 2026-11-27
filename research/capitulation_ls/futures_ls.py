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


def has_perp(symbols: np.ndarray, t: np.ndarray, warmup_s: int = 86400) -> np.ndarray:
    w = perp_window()
    out = np.zeros(len(symbols), dtype=bool)
    for i, (s, ti) in enumerate(zip(symbols, t, strict=True)):
        ab = w.get(str(s))
        out[i] = ab is not None and ab[0] + warmup_s <= ti <= ab[1]
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
