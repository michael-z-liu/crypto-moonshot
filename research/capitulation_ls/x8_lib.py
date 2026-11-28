"""X8 library (crazy-idea explorer): the four sleeves as trade frames with entry times and diag
columns, hourly perp/spot matrices of the whole universe, a day-clustered bootstrap and book
helpers. Shared by `x8_calendar.py` and `x8_baskets.py`. In-sample exploration only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from common_ls import OWN, meta, scores
from diag import annotate, scorecard
from e25_combined import book
from futures_ls import FUT, funding_cum, spot_name
from sim import Rule, simulate

EX = OWN / "explore"
EX.mkdir(exist_ok=True)
OUT = Path(__file__).resolve().parent / "results" / "explore"
T0 = int(np.datetime64("2020-01-01", "s").astype(int))
T1 = int(np.datetime64("2026-11-01", "s").astype(int))
NH = (T1 - T0) // 3600
KIND = {0: "time", 1: "stop", 2: "tp", 3: "liq", 4: "stop"}
BOOK_COLS = ["symbol", "adv", "t_in", "t_out", "net", "qv", "sleeve", "prio"]
DIAG_COLS = ["signal", "threshold", "net", "mfe", "mae", "tp", "stop", "exit_kind", "year"]


# ---------------------------------------------------------------- sleeves ----------------------
def _sim(c: pl.DataFrame, rule: Rule, **kw: object) -> pl.DataFrame:
    t = simulate(c, rule, start_equity=1e12, **kw).trades  # type: ignore[arg-type]
    qv = meta()["qv_base"].to_numpy()
    return t.with_columns(
        year=pl.col("open_time").dt.year(),
        exit_kind=pl.col("kind").replace_strict(KIND, default="time"),
        t_in=pl.col("t") + 300,
        qv=pl.col("row").map_elements(lambda r: float(qv[r]) * 288, return_dtype=pl.Float64),
        mae=pl.col("adv"),
    )


def sleeves(refresh: bool = False) -> pl.DataFrame:
    """L, C, F, M trades (every candidate, no slot limit) with book and diag columns.

    Identical to e25 sleeve_long / sleeve_crash_short / x6 F pool / sleeve_listing_weak."""
    p = EX / "x8_sleeves.parquet"
    if p.exists() and not refresh:
        return pl.read_parquet(p)
    m = meta()
    yr = m["year"].to_numpy()
    pct = scores("day5")
    out = []
    # L: capitulation longs, day5 >= 0.995, TP 20%, 72h, spot
    ie = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.995))
    t = _sim(
        pl.DataFrame({"row": ie, "side": 1, "prio": pct[ie]}),
        Rule(hold_min=4320, tp=0.2, slots=100_000, cost=0.004, liq_frac=None),
    )
    out.append(
        t.with_columns(
            sleeve=pl.lit("L"),
            prio=3.0 + pl.col("prio"),
            signal=pl.col("prio"),
            threshold=pl.lit(0.995),
            tp=pl.lit(0.2),
            stop=pl.lit(None, dtype=pl.Float64),
        )
    )
    feats = pl.read_parquet(OWN / "feats.parquet")
    base = pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("open_time").dt.year() >= 2022)
    # C: idiosyncratic crash shorts, 24h, perp 1x
    dc = feats.filter(base & (pl.col("ret_4h") <= -0.15) & (pl.col("btc_ret_4h").abs() <= 0.01))
    dc = dc.filter(pl.Series(pct[dc["row"].to_numpy()] <= 0.5))
    t = _sim(
        dc.select("row", side=pl.lit(-1), prio=-pl.col("ret_4h")),
        Rule(hold_min=1440, slots=100_000, cost=0.003, sl=1.0, liq_frac=None),
        perp_funding=True,
        source="perp",
    )
    out.append(
        t.with_columns(
            sleeve=pl.lit("C"),
            prio=pl.lit(1.5),
            signal=pl.lit(1.0),
            threshold=pl.lit(0.5),
            tp=pl.lit(None, dtype=pl.Float64),
            stop=pl.lit(1.0),
        )
    )
    # F: pump shorts r4 >= 40% with funding >= 0, 72h, perp 1x
    df = feats.filter(base & (pl.col("ret_4h") >= 0.4) & (pl.col("rate").fill_null(0) >= 0))
    t = _sim(
        df.select("row", side=pl.lit(-1), prio="ret_4h"),
        Rule(hold_min=4320, slots=100_000, cost=0.003, sl=1.0, liq_frac=None),
        perp_funding=True,
        source="perp",
    )
    out.append(
        t.with_columns(
            sleeve=pl.lit("F"),
            prio=2.0 + pl.col("prio").clip(0, 0.99),
            signal=pl.lit(1.0),
            threshold=pl.lit(0.9),
            tp=pl.lit(None, dtype=pl.Float64),
            stop=pl.lit(1.0),
        )
    )
    cols = ["symbol", "adv", "t_in", "t_out", "net", "qv", "sleeve", "prio", "signal",
            "threshold", "mfe", "mae", "tp", "stop", "exit_kind", "year"]  # fmt: skip
    out = [o.select(cols) for o in out]
    # M: weak spot-listing shorts (e27, since_list <= 0), 30d, TP/stop 50%, perp 1x
    x = pl.read_parquet(OWN / "e27_listings.parquet").filter(
        (pl.col("year") >= 2022) & (pl.col("since_list") <= 0)
    )
    P = perp_h()
    col = {s: j for j, s in enumerate(P["spot"])}
    mfe = []
    for r in x.iter_rows(named=True):
        j = col[r["symbol"]]
        i, k = (r["t_in"] - T0) // 3600, (r["t_out"] - 3600 - T0) // 3600
        lo = P["low"][i : k + 1, j]
        mfe.append(float(-(np.nanmin(lo) / P["open"][i, j] - 1)))
    out.append(
        x.with_columns(mfe=pl.Series(mfe)).select(
            "symbol",
            "adv",
            "t_in",
            "t_out",
            "net",
            "qv",
            sleeve=pl.lit("M"),
            prio=pl.lit(1.0),
            signal=pl.lit(1.0),
            threshold=pl.lit(0.5),
            mfe="mfe",
            mae="adv",
            tp=pl.lit(0.5),
            stop=pl.lit(0.5),
            exit_kind=pl.when(pl.col("ret") >= 0.5)
            .then(pl.lit("tp"))
            .when(pl.col("ret") <= -0.5)
            .then(pl.lit("stop"))
            .otherwise(pl.lit("time")),
            year="year",
        )
    )
    s = pl.concat(out, how="vertical_relaxed").with_columns(pl.col("t_in", "t_out").cast(pl.Int64))
    s.write_parquet(p)
    return s


def as_book(t: pl.DataFrame) -> pl.DataFrame:
    """Sleeve frame -> e25 book input (M is called N there; book() treats names the same)."""
    return t.select(BOOK_COLS).with_columns(pl.col("t_in", "t_out").cast(pl.Int64))


def card(t: pl.DataFrame) -> dict[str, float]:
    return scorecard(annotate(t.select(DIAG_COLS)))


# ---------------------------------------------------------------- hourly matrices --------------
_PH: dict[str, np.ndarray] | None = None
_SH: dict[str, np.ndarray] | None = None


def perp_h() -> dict[str, np.ndarray]:
    """Hourly perp matrices [hours since 2020-01-01, symbol] of every USD-M perp (delisted
    included): open, high, low, close, qv (0 for flat/frozen candles); keys sym (perp name),
    spot (spot name used by the funding cache)."""
    global _PH
    if _PH is not None:
        return _PH
    p = EX / "x8_perp_h.npz"
    if not p.exists():
        d = pl.read_parquet(FUT / "perp_1h.parquet")
        d = d.with_columns(
            spot=pl.col("symbol").map_elements(spot_name, return_dtype=pl.Utf8),
            h=((pl.col("open_time").dt.epoch("s") - T0) // 3600).cast(pl.Int64),
        ).filter(~pl.col("spot").is_in(["BOBUSDT", "CATUSDT"]))  # two perps per spot name
        d = d.filter((pl.col("h") >= 0) & (pl.col("h") < NH)).unique(["symbol", "h"])
        syms = sorted(d["symbol"].unique().to_list())
        jj = {s: j for j, s in enumerate(syms)}
        j = d["symbol"].replace_strict(jj).to_numpy()
        h = d["h"].to_numpy()
        M: dict[str, np.ndarray] = {}
        for c, src in (("open", "open"), ("high", "high"), ("low", "low"), ("close", "close")):
            a = np.full((NH, len(syms)), np.nan, dtype=np.float32)
            a[h, j] = d[src].to_numpy()
            M[c] = a
        q = np.zeros((NH, len(syms)), dtype=np.float32)
        q[h, j] = np.where(
            d["high"].to_numpy() > d["low"].to_numpy(), d["quote_volume"].to_numpy(), 0.0
        )
        M["qv"] = q
        M["sym"] = np.array(syms)
        M["spot"] = np.array([spot_name(s) for s in syms])
        np.savez(p, **M)
    z = np.load(p)
    _PH = {k: z[k] for k in z.files}
    return _PH


def spot_h() -> dict[str, np.ndarray]:
    """Hourly spot matrices of universe "all" (x3_build): open, high, low, close, qv, syms."""
    global _SH
    if _SH is None:
        z = np.load(EX / "spot_h.npz")
        _SH = {k: z[k] for k in z.files}
    return _SH


def funding_matrix(spots: np.ndarray, hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each spot name and each hour index: (cumulative funding rate settled at or before the
    hour start, last settled rate at or before the hour start); nan where unknown."""
    fc = funding_cum()
    cum = np.full((len(hours), len(spots)), np.nan)
    last = np.full((len(hours), len(spots)), np.nan)
    ts = T0 + hours.astype(np.int64) * 3600
    for j, s in enumerate(spots):
        c = fc.get(str(s))
        if c is None:
            continue
        t, cs = c
        k = np.searchsorted(t, ts, side="right") - 1
        ok = k >= 0
        cum[ok, j] = cs[k[ok]]
        rate = np.diff(np.concatenate([[0.0], cs]))
        last[ok, j] = rate[k[ok]]
        cum[~ok, j] = 0.0
    return cum, last


# ---------------------------------------------------------------- statistics -------------------
def boot_diff(
    o: np.ndarray, day: np.ndarray, mask: np.ndarray, reps: int = 2000, seed: int = 0
) -> tuple[float, float, float, float, float]:
    """Mean o in mask, mean o outside, and the day-clustered bootstrap 95% CI of the bucket mean
    and of the difference (bucket - rest). Days are resampled with replacement."""
    rng = np.random.default_rng(seed)
    ud, inv = np.unique(day, return_inverse=True)
    nd = len(ud)
    s_in = np.bincount(inv, weights=o * mask, minlength=nd)
    n_in = np.bincount(inv, weights=mask.astype(float), minlength=nd)
    s_out = np.bincount(inv, weights=o * ~mask, minlength=nd)
    n_out = np.bincount(inv, weights=(~mask).astype(float), minlength=nd)
    w = rng.multinomial(nd, np.full(nd, 1.0 / nd), size=reps).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mi = (w @ s_in) / (w @ n_in)
        mo = (w @ s_out) / (w @ n_out)
    d = mi - mo
    a, b = np.nanpercentile(mi, [2.5, 97.5])
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return float(lo), float(hi), float(a), float(b), float(np.mean(np.isfinite(d)))


def book_stats(b: pl.DataFrame) -> dict[str, object]:
    n = int(b["trades"].sum())
    bp = float((b["bp"] * b["trades"]).sum() / n) if n else 0.0
    rets = b["ret"].to_numpy()
    return {
        "trades": n,
        "bp": bp,
        "min": float(rets.min()) if len(rets) else 0.0,
        "geo": float(np.prod(1 + rets) ** (1 / len(rets)) - 1) if len(rets) else 0.0,
        "dd": float(b["dd"].min()) if len(rets) else 0.0,
        "years": " ".join(f"{r['year']}:{r['ret'] * 100:+.0f}%" for r in b.iter_rows(named=True)),
    }


def run_book(frames: list[pl.DataFrame], slots: int = 10) -> dict[str, object]:
    tr = pl.concat([f.select(BOOK_COLS) for f in frames], how="vertical_relaxed")
    tr = tr.with_columns(pl.col("t_in", "t_out").cast(pl.Int64))
    return book_stats(book(tr, slots))
