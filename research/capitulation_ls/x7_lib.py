# ruff: noqa: E501
"""X7 shared helpers: hourly spot/perp matrices (x3_build), an hourly long/short exit simulator
with funding and isolated-1x liquidation, diag scorecards and the 10-slot books (alone and added
to LMC = L + M + C). Used by x7_second_pump.py and x7_listing_month2.py (in-sample exploration).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import polars as pl
from common_ls import OWN
from diag import annotate, scorecard
from futures_ls import funding_cum

EX = OWN / "explore"
T0 = int(np.datetime64("2020-01-01", "s").astype(int))
SPOT_COST, PERP_COST = 0.004, 0.003
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)


@cache
def spot() -> dict[str, np.ndarray]:
    z = np.load(EX / "spot_h.npz")
    return {k: z[k] for k in z.files}


@cache
def perp() -> dict[str, np.ndarray]:
    z = np.load(EX / "perp_h.npz")
    return {k: z[k] for k in z.files}


def ep(h: int) -> int:
    """Hour index -> epoch seconds of the hour start."""
    return T0 + int(h) * 3600


def fpaid(sym: str, a_t: int, b_t: int) -> float:
    """Funding paid by a short over (a_t, b_t] as a fraction of notional (negative = received)."""
    c = funding_cum().get(sym)
    if c is None:
        return 0.0
    t, cs = c
    a = np.searchsorted(t, a_t, side="right") - 1
    b = np.searchsorted(t, b_t, side="right") - 1
    return -float((cs[b] if b >= 0 else 0.0) - (cs[a] if a >= 0 else 0.0))


def last_rate(sym: str, t: int) -> float:
    """Last funding rate settled at or before t (nan if none)."""
    c = funding_cum().get(sym)
    if c is None:
        return float("nan")
    tt, cs = c
    j = np.searchsorted(tt, t, side="right") - 1
    if j < 0:
        return float("nan")
    return float(cs[j] - (cs[j - 1] if j >= 1 else 0.0))


def perp_ok(col: int, j: int, min_qv: float = 50_000, warm: int = 24) -> bool:
    """Perp traded >= min_qv (with a moving price) in each of hours j and j-1 (the last two closed
    hours before an entry at j+1) and has at least `warm` hours of history."""
    if col < 0 or j < warm + 2:
        return False
    p = perp()
    q, hi, lo = p["quote_volume"][:, col], p["high"][:, col], p["low"][:, col]
    if not np.isfinite(p["close"][j - warm, col]):
        return False
    return all(bool(np.isfinite(q[k]) and q[k] >= min_qv and hi[k] > lo[k]) for k in (j, j - 1))


def simulate(
    o: np.ndarray,
    hi: np.ndarray,
    lo: np.ndarray,
    c: np.ndarray,
    e: int,
    hold: int,
    side: int,
    tp: float | None,
    stop: float | None,
) -> dict[str, float] | None:
    """Hourly exit simulation from the open of bar e for `hold` bars, adverse side first.

    side +1 long (spot), -1 short (perp, isolated 1x: liquidation at +100% if no tighter stop).
    Gaps through a level fill at the bar open. A missing bar (delisting/freeze) exits at the last
    valid close. Returns gross return, exit bar, kind, mfe, mae (all as fractions of entry)."""
    if e >= len(o) or not np.isfinite(o[e]) or o[e] <= 0:
        return None
    e0 = float(o[e])
    end = min(e + hold, len(o))
    h_, l_, o_, c_ = hi[e:end], lo[e:end], o[e:end], c[e:end]
    bad = ~(np.isfinite(h_) & np.isfinite(l_) & np.isfinite(c_))
    if bad.any():  # delisted / missing: stop the path at the last valid bar
        m = int(bad.argmax())
        if m == 0:
            return None
        h_, l_, o_, c_ = h_[:m], l_[:m], o_[:m], c_[:m]
    if side > 0:
        fav, adv, oadv = h_ / e0 - 1, 1 - l_ / e0, 1 - o_ / e0
    else:
        fav, adv, oadv = 1 - l_ / e0, h_ / e0 - 1, o_ / e0 - 1
    n = len(fav)
    stop_lvl = stop if stop is not None else (1.0 if side < 0 else None)
    ks = int((adv >= stop_lvl).argmax()) if stop_lvl is not None and (adv >= stop_lvl).any() else n
    kt = int((fav >= tp).argmax()) if tp is not None and (fav >= tp).any() else n
    if ks < n and ks <= kt:  # adverse first within a bar
        a = max(float(oadv[ks]), stop_lvl)  # gap through the stop fills at the open
        liq = side < 0 and (stop is None or a >= 1.0)
        ret = -1.0 if liq else -a
        mfe = float(fav[:ks].max()) if ks > 0 else 0.0
        return {"ret": ret, "k_exit": e + ks, "kind": "liq" if liq else "stop",
                "mfe": max(mfe, 0.0), "mae": min(a, 1.0)}  # fmt: skip
    if kt < n:
        mae = float(adv[: kt + 1].max())
        return {
            "ret": float(tp),
            "k_exit": e + kt,
            "kind": "tp",
            "mfe": float(tp),
            "mae": min(max(mae, 0.0), 1.0),
        }
    ret = side * (float(c_[-1]) / e0 - 1)
    return {"ret": ret, "k_exit": e + n - 1, "kind": "time", "mfe": max(float(fav.max()), 0.0),
            "mae": min(max(float(adv.max()), 0.0), 1.0)}  # fmt: skip


def card(t: pl.DataFrame, tp: float | None, stop: float | None) -> dict[str, float]:
    f = t.select(
        signal=pl.col("signal") if "signal" in t.columns else pl.lit(1.0),
        threshold=pl.lit(0.0), net="net", mfe="mfe", mae="mae",
        tp=pl.lit(tp, dtype=pl.Float64), stop=pl.lit(stop, dtype=pl.Float64),
        exit_kind="kind", year="year",
    )  # fmt: skip
    return scorecard(annotate(f))


def diag_frame(t: pl.DataFrame, theta: float, tp: float | None, stop: float | None) -> pl.DataFrame:
    return t.select(
        signal="signal", threshold=pl.lit(theta), net="net", mfe="mfe", mae="mae",
        tp=pl.lit(tp, dtype=pl.Float64), stop=pl.lit(stop, dtype=pl.Float64),
        exit_kind="kind", year="year",
    )  # fmt: skip


def per_year(t: pl.DataFrame) -> dict[int, tuple[int, float]]:
    g = t.group_by("year").agg(n=pl.len(), e=pl.col("net").mean())
    return {int(r["year"]): (int(r["n"]), float(r["e"])) for r in g.iter_rows(named=True)}


def yr_cells(t: pl.DataFrame) -> str:
    py = per_year(t)
    return " | ".join(f"{py[y][1] * 100:+.1f} ({py[y][0]})" if y in py else "-" for y in YEARS)


def split_mean(t: pl.DataFrame, a: int, b: int) -> tuple[int, float]:
    """n and mean of the per-year mean net over years a..b (years without trades skipped)."""
    py = per_year(t)
    ys = [py[y] for y in range(a, b + 1) if y in py]
    if not ys:
        return 0, float("nan")
    return sum(n for n, _ in ys), float(np.mean([e for _, e in ys]))


BOOK_COLS = ["symbol", "adv", "t_in", "t_out", "net", "qv", "sleeve", "prio", "w"]


def as_book(t: pl.DataFrame, sleeve: str = "X", prio: float = 2.0) -> pl.DataFrame:
    """Trade frame (symbol, t_in, t_out, net, mae, qv24) -> e25 book rows (one per coin held)."""
    return t.select(
        "symbol", adv="mae", t_in="t_in", t_out="t_out", net="net", qv="qv24",
        sleeve=pl.lit(sleeve), prio=pl.lit(prio), w=pl.lit(1.0),
    )  # fmt: skip


@cache
def lmc_rows() -> pl.DataFrame:
    """L + M + C candidate trades (e25 sleeves, 2022+), cached on disk."""
    p = EX / "x7_lmc.parquet"
    if p.exists():
        return pl.read_parquet(p)
    from e25_combined import sleeve_crash_short, sleeve_listing_weak, sleeve_long

    fr = [
        s.with_columns(w=pl.lit(1.0)).select(BOOK_COLS)
        for s in (sleeve_long(), sleeve_listing_weak(), sleeve_crash_short())
    ]
    d = pl.concat(fr, how="vertical_relaxed")
    d.write_parquet(p)
    return d


def books(t: pl.DataFrame, sleeve: str = "X", prio: float = 2.0) -> dict[str, str]:
    """10-slot per-year $10k books: X alone (2021+), LMC, LMC + X (2022+). Returns table cells."""
    from e25_combined import book

    x = as_book(t, sleeve, prio)
    out: dict[str, str] = {}

    def fmt(b: pl.DataFrame) -> str:
        if b.height == 0:
            return "no trades"
        n = int(b["trades"].sum())
        bp = float((b["bp"] * b["trades"]).sum() / n)
        yrs = " ".join(f"{int(r['year'])}:{r['ret'] * 100:+.0f}%" for r in b.iter_rows(named=True))
        return f"{n}t, {bp:+.0f}bp/trade, min yr {b['ret'].min() * 100:+.0f}%, maxDD {b['dd'].min() * 100:+.0f}% | {yrs}"

    out["alone"] = fmt(book(x, 10)) if x.height else "no trades"
    base = lmc_rows()
    out["lmc"] = fmt(book(base, 10))
    out["lmc_plus"] = fmt(
        book(
            pl.concat([base, x.filter(pl.col("t_in") >= ep_year(2022))], how="vertical_relaxed"), 10
        )
    )
    return out


def ep_year(y: int) -> int:
    return int(np.datetime64(f"{y}-01-01", "s").astype(int))
