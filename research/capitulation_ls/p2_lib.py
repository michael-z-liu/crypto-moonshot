"""P2 (perp-only follow-ups): shared pieces. The LMC / LMC+ trade frames (LMC+ = LMC + L on
perp-only coins, as p1_eval.py), a perp simulator with an entry delay, kline fetching from
data.binance.vision into a given raw directory, and in-memory spike scoring with the day5
walk-forward folds (the p1_lib pipeline, parity-checked in p1_parity.py).
"""

from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from futures_ls import funding_paid, has_perp, spot_name
from p1_lib import (
    CHANNELS,
    P1,
    SEQ,
    Context,
    FoldModel,
    fold_name,
    perp_paths,
    spikes_and_coin_channels,
)
from sim import PERP_STEP_END_MIN, Rule, delayed, exits

P2 = P1.parent / "p2"
P2.mkdir(parents=True, exist_ok=True)
DL = "https://data.binance.vision/"
KIND = {0: "time", 1: "stop", 2: "tp", 3: "liq", 4: "stop"}
RL = Rule(hold_min=4320, tp=0.2, slots=100_000, cost=0.003, liq_frac=None)
BOOK_COLS = ["symbol", "adv", "t_in", "t_out", "net", "qv", "sleeve", "prio", "w"]
YEARS = (2022, 2023, 2024, 2025, 2026)
PathFn = Callable[[str, np.ndarray], tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------- simulation -------------------
def sim_perp(
    c: pl.DataFrame, side: int, rule: Rule, delay: int = 0, paths: PathFn = perp_paths
) -> pl.DataFrame:
    """p1_eval.sim with an optional entry delay (in 5m candles): every candidate of `c` (symbol,
    t, prio, ...) on its perp path, cost + funding, one position per coin."""
    parts = []
    for (sym,), g in c.group_by(["symbol"]):
        ok, p = paths(str(sym), g["t"].to_numpy())
        if not ok.any():
            continue
        g = g.filter(pl.Series(ok))
        ends = PERP_STEP_END_MIN
        if delay:
            p, ends, _ = delayed(p, delay, PERP_STEP_END_MIN)
        sd = np.full(g.height, side)
        ret, t_exit, kind = exits(p, sd, rule, ends)
        hold = ends[t_exit]
        last = np.minimum(t_exit, p.shape[2] - 1)
        advp = np.nan_to_num(-p[:, 1] if side > 0 else p[:, 0], nan=0.0)
        favp = np.nan_to_num(p[:, 0] if side > 0 else -p[:, 1], nan=0.0)
        adv = np.maximum.accumulate(advp, axis=1)[np.arange(g.height), last]
        mfe = np.maximum.accumulate(favp, axis=1)[np.arange(g.height), last]
        t_in = g["t"].to_numpy() + 300 + 300 * delay
        t_out = t_in + hold * 60
        fund = funding_paid(np.array([spot_name(str(sym))] * g.height), t_in, t_out, sd)
        net = np.maximum(rule.lev * (ret - rule.cost - fund), rule.loss_floor)
        parts.append(g.with_columns(
            t_in=pl.Series(t_in), t_out=pl.Series(t_out), net=pl.Series(net), ret=pl.Series(ret),
            fund=pl.Series(fund), kind=pl.Series(kind), adv=pl.Series(adv.astype(np.float64)),
            mfe=pl.Series(mfe.astype(np.float64)),
        ))  # fmt: skip
    if not parts:
        return pl.DataFrame()
    t = pl.concat(parts).sort(["t_in", "prio"], descending=[False, True])
    keep, busy = [], {}
    for r in t.select("symbol", "t_in", "t_out").iter_rows():
        if busy.get(r[0], -1) > r[1]:
            keep.append(False)
            continue
        busy[r[0]] = r[2]
        keep.append(True)
    return t.filter(pl.Series(keep)).with_columns(
        exit_kind=pl.col("kind").replace_strict(KIND, default="time"),
        year=pl.from_epoch("t", "s").dt.year(),
        day=pl.from_epoch("t", "s").dt.date(),
    )


def to_book(t: pl.DataFrame, sleeve: str) -> pl.DataFrame:
    """Perp L trades -> e25 book rows (stake cap from the perp's own volume, as p1_eval)."""
    return t.select(
        "symbol", adv=pl.col("adv").clip(0, 1), t_in="t_in", t_out="t_out", net="net",
        qv=pl.col("qv_base") * 288, sleeve=pl.lit(sleeve), prio=3.0 + pl.col("pct"), w=pl.lit(1.0),
    )  # fmt: skip


def perp_only_spikes() -> pl.DataFrame:
    sp = pl.read_parquet(P1 / "spikes.parquet")
    sc = pl.read_parquet(P1 / "scores.parquet").sort("row")
    return sp.with_columns(pct=sc["pct"].cast(pl.Float64))


def lperp(delay: int = 0) -> pl.DataFrame:
    """L on perp-only coins (p1): pct >= 0.995 with an active perp."""
    d = perp_only_spikes().filter(pl.col("perp") & (pl.col("pct") >= 0.995))
    return sim_perp(d.with_columns(prio=pl.col("pct")), 1, RL, delay)


def crash_short_delayed(delay: int) -> pl.DataFrame:
    """e25 `sleeve_crash_short` with an entry delay (the same candidates and rule)."""
    from common_ls import OWN, meta, scores
    from sim import simulate

    m = meta()
    d = pl.read_parquet(OWN / "feats.parquet").filter(
        pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("open_time").dt.year() >= 2022)
        & (pl.col("ret_4h") <= -0.15) & (pl.col("btc_ret_4h").abs() <= 0.01)
    )  # fmt: skip
    pct = scores("day5")
    d = d.filter(pl.Series(pct[d["row"].to_numpy()] <= 0.5))
    b = simulate(
        d.select("row", side=pl.lit(-1), prio=-pl.col("ret_4h")),
        Rule(hold_min=1440, slots=100_000, cost=0.003, sl=1.0, liq_frac=None, delay=delay),
        start_equity=1e12, perp_funding=True, source="perp",
    )  # fmt: skip
    t = b.trades
    qv = m["qv_base"].to_numpy()
    return t.select(
        "symbol", "adv", t_in=pl.col("t") + 300 + 300 * delay, t_out="t_out", net="net",
        qv=pl.col("row").map_elements(lambda r: float(qv[r]) * 288, return_dtype=pl.Float64),
        sleeve=pl.lit("C"), prio=pl.lit(1.5),
    )  # fmt: skip


def lmc(delay: int = 0) -> list[pl.DataFrame]:
    """The LMC sleeves (e25 sleeve_long, sleeve_listing_weak, sleeve_crash_short) as book rows.
    `delay` delays the 5m entries of L and C (M enters on hourly candles and is left as is)."""
    from e25_combined import sleeve_crash_short, sleeve_listing_weak, sleeve_long

    c = sleeve_crash_short() if delay == 0 else crash_short_delayed(delay)
    parts = [sleeve_long(delay=delay), sleeve_listing_weak(), c]
    return [p.with_columns(w=pl.lit(1.0)).select(BOOK_COLS) for p in parts]


def book_row(name: str, b: pl.DataFrame) -> dict[str, object]:
    n = int(b["trades"].sum())
    bp = float((b["bp"] * b["trades"]).sum() / n) if n else 0.0
    yrs = {int(r["year"]): r for r in b.iter_rows(named=True)}
    row: dict[str, object] = {"book": name, "trades": n, "bp": round(bp)}
    for y in YEARS:
        r = yrs.get(y)
        row[str(y)] = f"{r['ret'] * 100:+.0f}% (dd {r['dd'] * 100:.0f})" if r else "-"
    row["min yr"] = f"{min(float(r['ret']) for r in yrs.values()) * 100:+.0f}%"
    return row


def md(df: pl.DataFrame, digits: int = 3) -> str:
    out = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for r in df.iter_rows():
        cells = [f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- data -------------------------
def fetch_csv(key: str, raw_dir: Path) -> bytes | None:
    """Download one archive zip into `raw_dir`, return its CSV and delete the zip."""
    dst = raw_dir / key.split("/")[-1]
    for _ in range(3):
        try:
            urllib.request.urlretrieve(DL + key, dst)
            with zipfile.ZipFile(dst) as z:
                names = z.namelist()
                if len(names) != 1 or not names[0].endswith(".csv"):
                    return None
                return z.read(names[0])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            continue
        finally:
            if dst.exists():
                os.remove(dst)
    print("failed", key, file=sys.stderr)
    return None


def _num(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float("nan")


def parse_klines(raw: bytes) -> pl.DataFrame | None:
    """Perp kline CSV -> the p1 perp5m columns (ms, OHLC, qv, vol, tbv, n)."""
    rows = [ln.split(",") for ln in raw.decode("ascii", "replace").splitlines()]
    rows = [r for r in rows if len(r) >= 11 and r[0].isdigit()]
    if not rows:
        return None
    df = pl.DataFrame({
        "ms": [int(r[0]) for r in rows],
        **{c: [_num(r[i]) for r in rows] for i, c in enumerate(("open", "high", "low", "close", "vol"), 1)},  # noqa: E501
        "qv": [_num(r[7]) for r in rows],
        "n": [int(r[8]) if r[8].isdigit() else 0 for r in rows],
        "tbv": [_num(r[9]) for r in rows],
    })  # fmt: skip
    return df.with_columns(
        ms=pl.when(pl.col("ms") > 10**14).then(pl.col("ms") // 1000).otherwise(pl.col("ms"))
    )


def finish(parts: list[pl.DataFrame]) -> pl.DataFrame:
    cols = ["ms", "open", "high", "low", "close", "qv", "vol", "tbv", "n"]
    parts = [
        p.with_columns(
            pl.col("ms").cast(pl.Int64),
            pl.col("open", "high", "low", "close", "qv", "vol", "tbv").cast(pl.Float32),
            pl.col("n").cast(pl.Int32),
        ).select(cols)
        for p in parts
    ]
    return pl.concat(parts).unique("ms", keep="last").sort("ms")


# ---------------------------------------------------------------- scoring ----------------------
def score_coins(
    raws: dict[str, pl.DataFrame], start: dict[str, datetime], end: dict[str, datetime],
    ctx: Context, fold: Callable[[int], str] = fold_name,
) -> pl.DataFrame:  # fmt: skip
    """Liquid spikes of each coin in [start, end), scored in memory with the fold of their year
    (`fold(year)`); adds btc_ret_4h, t, year, day and the perp activity gate."""
    metas, seqs = [], []
    for sym, raw in raws.items():
        got = spikes_and_coin_channels(sym, raw, start[sym], end[sym])
        if got is None:
            continue
        meta, coin = got
        seq = np.zeros((meta.height, CHANNELS, SEQ), dtype=np.float16)
        seq[:, :7] = coin
        btc4 = ctx.fill(seq, meta["open_time"])
        metas.append(meta.with_columns(btc_ret_4h=pl.Series(btc4)))
        seqs.append(seq)
    if not metas:
        return pl.DataFrame()
    meta = pl.concat(metas, how="vertical_relaxed")
    seq = np.concatenate(seqs)
    yr = meta["open_time"].dt.year().to_numpy()
    pct = np.full(meta.height, np.nan)
    raw_s = np.full(meta.height, np.nan)
    for y in sorted(set(yr.tolist())):
        try:
            fm = FoldModel(fold(int(y)))
        except FileNotFoundError:
            continue
        r = np.flatnonzero(yr == y)
        raw_s[r], pct[r] = fm.score(seq[r], meta["qv_base"].to_numpy()[r])
    t = meta["open_time"].dt.epoch("s").to_numpy()
    spot = np.array([spot_name(s) for s in meta["symbol"].to_list()])
    return meta.with_columns(
        raw=pl.Series(raw_s), pct=pl.Series(pct), t=pl.Series(t),
        year=pl.col("open_time").dt.year(), day=pl.col("open_time").dt.date(),
        perp=pl.Series(has_perp(spot, t)),
    )  # fmt: skip
