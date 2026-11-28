"""F1 (pump shorts on perp-only coins): shared pieces. The F / P pump-short candidates of x8_lib
(`sleeves()`: 5m liquid spike rows with ret_4h >= 40%, perp active, qv_base >= 20k; F adds the last
settled funding rate >= 0 at the signal bar, null = 0) built on the p1 perp-only spikes and the p2
pre-spot spikes, simulated on the perp (isolated 1x: stop at +100% adverse, 72h, cost 0.3% +
funding, one position per coin) with p2_lib.sim_perp.

Funding is keyed by the perp's own symbol here (futures_ls keys it by the spot name, which merges
BOBUSDT / 1000000BOBUSDT and CATUSDT / 1000CATUSDT); for every other coin the two are identical.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import p1_lib
import polars as pl
from futures_ls import funding, spot_name
from p1_lib import P1
from p2_lib import BOOK_COLS, P2, perp_only_spikes, sim_perp
from sim import Rule

RF = Rule(hold_min=4320, slots=100_000, cost=0.003, sl=1.0, liq_frac=None)  # x8 F / P rule
R4 = 0.4
QV = 20_000.0
F1 = P1.parent / "f1"
F1.mkdir(parents=True, exist_ok=True)


@cache
def funding_by_perp() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """perp symbol -> (settlement times in epoch s, rate, cumulative rate after each time)."""
    out = {}
    for (sym,), g in funding().sort("time").group_by(["symbol"], maintain_order=True):
        r = g["rate"].to_numpy()
        out[str(sym)] = (g["time"].dt.epoch("s").to_numpy(), r, np.cumsum(r))
    return out


def last_rate(syms: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Last funding rate settled at or before t (the signal bar's open time, epoch s); nan if
    none (x8 treats a missing rate as 0, i.e. the gate passes)."""
    fb = funding_by_perp()
    out = np.full(len(syms), np.nan)
    for i, (s, ti) in enumerate(zip(syms, t, strict=True)):
        c = fb.get(str(s))
        if c is None:
            continue
        k = np.searchsorted(c[0], ti, side="right") - 1
        if k >= 0:
            out[i] = c[1][k]
    return out


def funding_paid_perp(
    syms: np.ndarray, t_in: np.ndarray, t_out: np.ndarray, side: np.ndarray
) -> np.ndarray:
    """futures_ls.funding_paid keyed by the perp symbol (negative = received)."""
    fb = funding_by_perp()
    out = np.zeros(len(syms))
    for i, s in enumerate(syms):
        c = fb.get(str(s))
        if c is None:
            continue
        t, _, cs = c
        a = np.searchsorted(t, t_in[i], side="right") - 1
        b = np.searchsorted(t, t_out[i], side="right") - 1
        out[i] = side[i] * ((cs[b] if b >= 0 else 0.0) - (cs[a] if a >= 0 else 0.0))
    return out


def pumps(sp: pl.DataFrame, lo: float = R4, hi: float = np.inf) -> pl.DataFrame:
    """Spike rows with lo <= ret_4h < hi, perp active, qv_base >= 20k, plus the funding rate."""
    d = sp.filter(
        pl.col("perp") & (pl.col("qv_base") >= QV) & (pl.col("ret_4h") >= lo)
        & (pl.col("ret_4h") < hi) & (pl.col("t") >= 1640995200)
    )  # fmt: skip
    rate = last_rate(d["symbol"].to_numpy(), d["t"].to_numpy())
    return d.with_columns(rate=pl.Series(rate), prio=pl.col("ret_4h"))


def short(c: pl.DataFrame, prespot: bool = False) -> pl.DataFrame:
    """Every candidate on its perp path (one position per coin), funding by perp symbol."""
    if c.height == 0:
        return pl.DataFrame()
    orig = p1_lib.P5
    if prespot:
        p1_lib.P5 = p1_lib.FUT / "perp5m_prespot"
    try:
        t = sim_perp(c, -1, RF)
    finally:
        p1_lib.P5 = orig
    if t.height == 0:
        return t
    syms = t["symbol"].to_numpy()
    fund = funding_paid_perp(syms, t["t_in"].to_numpy(), t["t_out"].to_numpy(), -np.ones(t.height))
    net = np.maximum(RF.lev * (t["ret"].to_numpy() - RF.cost - fund), RF.loss_floor)
    return t.with_columns(fund=pl.Series(fund), net=pl.Series(net))


def gate(c: pl.DataFrame) -> pl.DataFrame:
    return c.filter(pl.col("rate").fill_nan(None).fill_null(0) >= 0)


def perp_only_pool(lo: float = R4, hi: float = np.inf) -> pl.DataFrame:
    return pumps(perp_only_spikes(), lo, hi)


def prespot_pool(lo: float = R4, hi: float = np.inf) -> pl.DataFrame:
    sp = pl.read_parquet(P2 / "prespot_spikes.parquet")
    return pumps(sp, lo, hi)


def to_book(t: pl.DataFrame, sleeve: str = "F", spot_sym: bool = False) -> pl.DataFrame:
    """Pump-short trades -> e25 book rows (x8 F priority 2 + ret_4h; stake cap from the perp's own
    volume). spot_sym: book symbol = spot name (pre-spot coins later trade as spot coins)."""
    sym = (
        pl.col("symbol").map_elements(spot_name, return_dtype=pl.Utf8)
        if spot_sym
        else pl.col("symbol")
    )
    return t.select(
        symbol=sym, adv=pl.col("adv").clip(0, 1), t_in="t_in", t_out="t_out", net="net",
        qv=pl.col("qv_base") * 288, sleeve=pl.lit(sleeve), prio=2.0 + pl.col("ret_4h").clip(0, 0.99),  # noqa: E501
        w=pl.lit(1.0),
    ).select(BOOK_COLS)  # fmt: skip


def diag_frame(t: pl.DataFrame) -> pl.DataFrame:
    """diag.py trade frame (x8 F conventions: signal 1, threshold 0.9, no TP, stop 1.0)."""
    return t.select(
        signal=pl.lit(1.0), threshold=pl.lit(0.9), net="net", mfe="mfe", mae="adv",
        tp=pl.lit(None, dtype=pl.Float64), stop=pl.lit(1.0), exit_kind="exit_kind", year="year",
    )  # fmt: skip
