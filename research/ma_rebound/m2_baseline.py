"""M2: does buying every MA touch pay, and are the major MAs special? No model.

    uv run --group research python m2_baseline.py [--workers 8] [--rebuild]
    uv run --group research python m2_baseline.py --holdout   # M6 reveal context, see below

Stages (each cached under `cache/m2/`, a cut-off run resumes):
1. `trades.parquet`: every base/stress label of a fill before 2025-01-01, one row per trade after
   dropping duplicates (same coin, MA, entry, scenario, exit and fill candle: the earliest
   trigger is kept), with the candidate's attributes and a majors flag.
2. `targets.parquet` and `controls/<SYMBOL>.parquet`: matched random entries, coin by coin.
3. Tables in `results/m2/*.csv`, the charts, and `results/m2/tables.md` (the main tables as
   markdown, pasted into `results/M2.md`).

Holdout: 2025-01-01 and later is sealed. `read_cands`, `read_fills` and `read_labels` are the only
readers of the event tables and each applies the filter on fill time (decision time for
candidates) before anything else touches the rows. Nothing after the cut is printed or written.

`--holdout` (M6 reveal only, after the pick was committed and pushed): the same readers take the
window [2025-01-01, data end) instead, tables 2 (timeframe x fill year) and 3 (study vs placebo)
are computed for fill years 2025 and 2026 and written to `results/m6/m2_*.csv`, with trades cached
in `cache/m2/trades_holdout.parquet`. Labels still open at the data end (`censored`) are left out,
as M2 leaves out unfinished labels, and counted. The default run is unchanged.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from common import (
    CACHE,
    HOLDOUT_START,
    MA_LENGTHS,
    MIN_DAILY_QUOTE_VOLUME,
    RESULTS,
    TIMEFRAMES,
)
from events import DAY, FIVE, TF_NAMES, exit_hold_us, hold_us, read_5m, read_bars
from runner import END_US

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]

HOLD_US = int(HOLDOUT_START.timestamp()) * 1_000_000
LO_US: int | None = None  # lower bound of the window (holdout mode only)
TRADES_FILE = "trades.parquet"
M2_CACHE = CACHE / "m2"
OUT = RESULTS / "m2"
SEED = 20261113
DRAWS = 2000
CTRL_PER_CAND = 3
DEPTH_TOL = 0.25  # pullback depth tolerance, in units of u
FAR_FROM_MA = 0.5  # a control's close is more than this x u from every study MA level
YEARS = (2020, 2021, 2022, 2023, 2024)
ENTRIES = ("L0", "L0.05", "L0.15", "touch")
EXITS_ALL = ("E1", "E2", "E3", "E4", "E5", "E6")
BASE_ENTRIES = ("L0.05", "touch")
BASE_EXITS = ("E1", "E3")
CTRL_EXITS = {"half": "E2", "full": "E1", "long": "E6"}  # time exits at T/2, T, min(2T, 28 d)
MAJOR_SYMBOLS = ("BTCUSDT", "ETHUSDT")
MAJOR_TOP = 20
BP = 10_000.0


# --- holdout-safe readers (the only way to the event tables) -----------------------------------


def in_window(col: str) -> pl.Expr:
    """Times before the cut (default), or in [LO_US, data end) in holdout mode."""
    e = pl.col(col) < HOLD_US
    return e if LO_US is None else e & (pl.col(col) >= LO_US)


def read_cands() -> pl.DataFrame:
    """Candidates decided before the holdout, with the end of their order window (decision + T/2)
    and `in_denom`: the window ends by 2025-01-01, so the fill rate has no sealed fills in it."""
    cols = ["cid", "family", "placebo", "symbol", "tf", "n", "trigger_t", "decision_t", "u", "year"]
    half = pl.col("tf").replace_strict(
        {tf: hold_us(tf) // 2 for tf in TF_NAMES}, return_dtype=pl.Int64
    )
    df = (
        pl.read_parquet(CACHE / "candidates.parquet", columns=cols)
        .filter(in_window("decision_t"))
        .with_columns(window_end=pl.col("decision_t") + half)
        .with_columns(in_denom=pl.col("window_end") <= HOLD_US)
        .rename({"year": "trig_year"})
    )
    return df


def read_fills() -> pl.DataFrame:
    """Base-scenario filled orders with a fill before the holdout."""
    return (
        pl.scan_parquet(CACHE / "fills.parquet")
        .filter(pl.col("scenario") == "base", pl.col("status") == "filled")
        .filter(in_window("fill_t"))
        .select("cid", "buffer", "fill_t")
        .collect()
    )


def read_labels() -> pl.DataFrame:
    """Finished labels (both scenarios) of fills before the holdout."""
    lf = pl.scan_parquet(CACHE / "labels.parquet").filter(in_window("entry_t"))
    return lf.filter(pl.col("status") == "done").drop("status").collect()


def entry_name() -> pl.Expr:
    names = {0.0: "L0", 0.05: "L0.05", 0.15: "L0.15"}
    return (
        pl.when(pl.col("buffer").is_null())
        .then(pl.lit("touch"))
        .otherwise(pl.col("buffer").replace_strict(names, default=None, return_dtype=pl.String))
    )


def dedupe(df: pl.DataFrame, subset: list[str]) -> pl.DataFrame:
    """One row per fill candle: of the orders that fill in the same candle, the earliest trigger."""
    return df.sort("trigger_t", maintain_order=True).unique(
        subset=subset, keep="first", maintain_order=True
    )


# --- stage 1: trades ---------------------------------------------------------------------------


def major_flags(cands: pl.DataFrame) -> pl.DataFrame:
    """(symbol, day) -> BTC, ETH or among the 20 highest `liq` across all coins on that day."""
    from runner import PARTS

    frames = [
        pl.read_parquet(f, columns=["t", "liq"]).with_columns(symbol=pl.lit(f.stem))
        for f in sorted((PARTS / "bars_1d").glob("*.parquet"))
    ]
    liq = pl.concat(frames).filter(pl.col("liq").is_not_null())
    liq = liq.with_columns(rank=pl.col("liq").rank("ordinal", descending=True).over("t"))
    top = (
        liq.filter(pl.col("rank") <= MAJOR_TOP)
        .select(pl.col("t").alias("day"), "symbol")
        .with_columns(major=pl.lit(True))
    )
    day = pl.col("trigger_t") - pl.col("trigger_t") % DAY
    out = (
        cands.select("cid", "symbol", day=day)
        .join(top, on=["day", "symbol"], how="left")
        .with_columns(
            major=pl.col("major").fill_null(False) | pl.col("symbol").is_in(MAJOR_SYMBOLS)
        )
    )
    return out.select("cid", "major")


def build_trades() -> pl.DataFrame:
    t0 = time.perf_counter()
    cands = read_cands()
    cands = cands.join(major_flags(cands), on="cid", how="left")
    lab = read_labels()
    n_cens = (
        pl.scan_parquet(CACHE / "labels.parquet")
        .filter(in_window("entry_t"), pl.col("status") != "done")
        .select(pl.len())
        .collect()
        .item()
    )
    if LO_US is None:
        assert n_cens == 0, f"{n_cens} censored labels before 2025"
    else:
        print(f"holdout: {n_cens} censored labels (open at the data end) left out", flush=True)
    attrs = cands.select(
        "cid", "symbol", "tf", "n", "placebo", "trigger_t", "u", "trig_year", "major"
    )
    df = (
        lab.with_columns(entry=entry_name())
        .join(attrs, on="cid", how="inner")
        .with_columns(
            gross=pl.col("exit_price") / pl.col("entry_price") - 1,
            net_u=pl.col("net") / pl.col("u"),
            hold_d=(pl.col("exit_t") - pl.col("entry_t")) / DAY,
            day=pl.col("entry_t") // DAY,
            fyear=pl.from_epoch(pl.col("entry_t"), time_unit="us").dt.year(),
        )
    )
    df = dedupe(df, ["symbol", "tf", "n", "entry", "scenario", "exit", "entry_t"])
    df = df.select(
        "cid", "symbol", "tf", "n", "placebo", "entry", "exit", "scenario", "trigger_t", "entry_t",
        "entry_price", "exit_price", "net", "gross", "u", "net_u", "hold_d", "day", "fyear",
        "trig_year", "major",
    )  # fmt: skip
    print(f"trades: {df.height} rows in {time.perf_counter() - t0:.0f}s", flush=True)
    return df


def load_trades(rebuild: bool) -> pl.DataFrame:
    path = M2_CACHE / TRADES_FILE
    if path.exists() and not rebuild:
        return pl.read_parquet(path)
    M2_CACHE.mkdir(parents=True, exist_ok=True)
    df = build_trades()
    df.write_parquet(path)
    return df


# --- fill rates --------------------------------------------------------------------------------


def fill_rates(keys: list[str], extra: pl.DataFrame | None = None) -> pl.DataFrame:
    """Study fills / candidates per key, over candidates whose order window ends by the cut.
    `keys` may use `entry`, `tf`, `n`, `trig_year`. With `extra` (`cid`, `major`) only major
    candidates count. Duplicate fills (same coin, MA, entry and fill candle) count once; the
    candidates are all counted, so a refreshed order that triggered twice lowers the rate."""
    cands = read_cands().filter(pl.col("in_denom"), ~pl.col("placebo"))
    if extra is not None:
        cands = cands.join(extra, on="cid", how="left").filter(pl.col("major"))
    # a limit candidate serves all three buffers, a touch candidate one entry
    ent = pl.DataFrame({"entry": list(ENTRIES)}).with_columns(
        family=pl.when(pl.col("entry") == "touch").then(pl.lit("touch")).otherwise(pl.lit("limit"))
    )
    n_c = cands.join(ent, on="family").group_by(keys).len().rename({"len": "candidates"})
    fills = read_fills().join(cands, on="cid", how="inner").with_columns(entry=entry_name())
    fills = dedupe(fills, ["symbol", "tf", "n", "entry", "fill_t"])
    n_f = fills.group_by(keys).len().rename({"len": "fills"})
    out = n_c.join(n_f, on=keys, how="left").with_columns(pl.col("fills").fill_null(0))
    return out.with_columns(fill_rate=pl.col("fills") / pl.col("candidates"))


# --- bootstrap and summaries -------------------------------------------------------------------


def boot_ratios(sums: F64, counts: F64) -> F64:
    """Day bootstrap: `sums` and `counts` are (days, k). Each draw resamples the days with
    replacement; returns (DRAWS, k) of sum/count. Fixed seed."""
    d = sums.shape[0]
    rng = np.random.default_rng(SEED)
    w = rng.multinomial(d, np.full(d, 1.0 / d), size=DRAWS).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w @ sums) / (w @ counts)


def ci(draws: F64) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns of thin groups
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def _num(x: float | None) -> float:
    return float("nan") if x is None else float(x)


def summarize(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Per key: trades, mean net bp with day-bootstrap CI, median, win rate, mean net/u, hold."""
    stats = df.group_by(keys).agg(
        trades=pl.len(),
        days=pl.col("day").n_unique(),
        mean_bp=pl.col("net").mean() * BP,
        median_bp=pl.col("net").median() * BP,
        win_rate=(pl.col("net") > 0).mean(),
        mean_u=pl.col("net_u").mean(),
        mean_hold_d=pl.col("hold_d").mean(),
    )
    perday = (
        df.group_by([*keys, "day"])
        .agg(s=pl.col("net").sum() * BP, c=pl.len().cast(pl.Float64))
        .sort("day")  # a fixed day order makes the bootstrap draws reproducible
    )
    rows: list[dict[str, object]] = []
    for key, g in perday.partition_by(keys, as_dict=True, include_key=False).items():
        s = g["s"].to_numpy()[:, None]
        c = g["c"].to_numpy()[:, None]
        lo, hi = ci(boot_ratios(s, c)) if len(s) >= 2 else (float("nan"), float("nan"))
        rows.append({**dict(zip(keys, key, strict=True)), "ci_lo_bp": lo, "ci_hi_bp": hi})
    ci_df = pl.DataFrame(rows, schema_overrides={k: stats.schema[k] for k in keys})
    return stats.join(ci_df, on=keys, how="left").sort(keys)


def diff_table(a: pl.DataFrame, b: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Group A against group B per key, resampling the union of their days jointly."""
    sa = a.group_by([*keys, "day"]).agg(
        sa=pl.col("net").sum() * BP, ua=pl.col("net_u").sum(), ca=pl.len().cast(pl.Float64)
    )
    sb = b.group_by([*keys, "day"]).agg(
        sb=pl.col("net").sum() * BP, ub=pl.col("net_u").sum(), cb=pl.len().cast(pl.Float64)
    )
    j = sa.join(sb, on=[*keys, "day"], how="full", coalesce=True).fill_null(0.0).sort("day")
    rows: list[dict[str, object]] = []
    for key, g in j.partition_by(keys, as_dict=True, include_key=False).items():
        ca, cb = g["ca"].to_numpy(), g["cb"].to_numpy()
        sums = np.column_stack([g["sa"], g["sb"], g["ua"], g["ub"]])
        cnt = np.column_stack([ca, cb, ca, cb])
        na, nb = float(cnt[:, 0].sum()), float(cnt[:, 1].sum())
        row: dict[str, object] = {**dict(zip(keys, key, strict=True))}
        row.update(trades_a=int(na), trades_b=int(nb))
        if na == 0 or nb == 0 or len(g) < 2:
            row.update(
                a_bp=np.nan, b_bp=np.nan, diff_bp=np.nan, diff_lo_bp=np.nan, diff_hi_bp=np.nan,
                a_u=np.nan, b_u=np.nan, diff_u=np.nan, diff_lo_u=np.nan, diff_hi_u=np.nan,
            )  # fmt: skip
        else:
            tot_s, tot_c = sums.sum(axis=0), cnt.sum(axis=0)
            m = tot_s / tot_c
            r = boot_ratios(sums, cnt)
            lo_bp, hi_bp = ci(r[:, 0] - r[:, 1])
            lo_u, hi_u = ci(r[:, 2] - r[:, 3])
            row.update(
                a_bp=m[0], b_bp=m[1], diff_bp=m[0] - m[1], diff_lo_bp=lo_bp, diff_hi_bp=hi_bp,
                a_u=m[2], b_u=m[3], diff_u=m[2] - m[3], diff_lo_u=lo_u, diff_hi_u=hi_u,
            )  # fmt: skip
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).sort(keys)


def mean_bp_by(df: pl.DataFrame, keys: list[str], name: str) -> pl.DataFrame:
    return df.group_by(keys).agg(
        pl.col("net").mean().mul(BP).alias(name), pl.len().alias(f"{name}_trades")
    )


# --- tables 1, 2, 3, 5 -------------------------------------------------------------------------


def expand(df: pl.DataFrame) -> pl.DataFrame:
    """Four copies: per timeframe or all, per fill year or all (columns `tf_g`, `year_g`)."""
    yr = pl.col("fyear").cast(pl.String)
    parts = [
        df.with_columns(tf_g=pl.col("tf"), year_g=yr),
        df.with_columns(tf_g=pl.col("tf"), year_g=pl.lit("all")),
        df.with_columns(tf_g=pl.lit("all"), year_g=yr),
        df.with_columns(tf_g=pl.lit("all"), year_g=pl.lit("all")),
    ]
    return pl.concat(parts)


def table1(base: pl.DataFrame, stress: pl.DataFrame) -> pl.DataFrame:
    keys = ["tf", "n", "entry", "exit"]
    t = summarize(base, keys).join(mean_bp_by(stress, keys, "stress_mean_bp"), on=keys, how="left")
    fr = fill_rates(["tf", "n", "entry"])
    t = t.join(fr, on=["tf", "n", "entry"], how="left")
    return sort_tf(t, "tf", ["n", "entry", "exit"]).select(
        "tf", "n", "entry", "exit", "candidates", "fill_rate", "trades", "mean_bp", "ci_lo_bp",
        "ci_hi_bp", "median_bp", "win_rate", "mean_u", "mean_hold_d", "stress_mean_bp",
        "stress_mean_bp_trades", "days",
    )  # fmt: skip


TF_RANK = {tf: i for i, tf in enumerate((*TF_NAMES, "all"))}


def sort_tf(df: pl.DataFrame, tf_col: str, then: list[str]) -> pl.DataFrame:
    return df.with_columns(_r=pl.col(tf_col).replace_strict(TF_RANK)).sort("_r", *then).drop("_r")


def table6(base: pl.DataFrame, stress: pl.DataFrame) -> pl.DataFrame:
    """All study MAs of a timeframe pooled, every entry and exit (CSV only, plus an E1 excerpt)."""
    keys = ["tf", "entry", "exit"]
    pooled = pl.concat([base, base.with_columns(tf=pl.lit("all"))])
    pooled_s = pl.concat([stress, stress.with_columns(tf=pl.lit("all"))])
    t = summarize(pooled, keys).join(
        mean_bp_by(pooled_s, keys, "stress_mean_bp"), on=keys, how="left"
    )
    return sort_tf(t, "tf", ["entry", "exit"])


def table2(study: pl.DataFrame) -> pl.DataFrame:
    sub = expand(
        study.filter(
            pl.col("entry").is_in(BASE_ENTRIES), pl.col("exit").is_in(BASE_EXITS),
            pl.col("scenario") == "base", pl.col("fyear").is_in(YEARS),
        )
    )  # fmt: skip
    keys = ["tf_g", "year_g", "entry", "exit"]
    t = summarize(sub, keys)
    # fill rate by trigger year (the candidates' own year), `all` pooled
    fr = fill_rates(["tf", "trig_year", "entry"])
    fr = fr.with_columns(tf_g=pl.col("tf"), year_g=pl.col("trig_year").cast(pl.String))
    parts = [fr.select("tf_g", "year_g", "entry", "candidates", "fills")]
    for tf_all, yr_all in ((False, True), (True, False), (True, True)):
        g = fr.with_columns(
            tf_g=pl.lit("all") if tf_all else pl.col("tf"),
            year_g=pl.lit("all") if yr_all else pl.col("trig_year").cast(pl.String),
        )
        parts.append(
            g.group_by("tf_g", "year_g", "entry")
            .agg(pl.col("candidates").sum(), pl.col("fills").sum())
            .select("tf_g", "year_g", "entry", "candidates", "fills")
        )
    frate = pl.concat(parts).with_columns(fill_rate=pl.col("fills") / pl.col("candidates"))
    frate = frate.rename({"candidates": "cands_by_trigger_year", "fills": "fills_by_trigger_year"})
    t = t.join(frate, on=["tf_g", "year_g", "entry"], how="left")
    return sort_tf(t, "tf_g", ["year_g", "entry", "exit"])


def table3(study: pl.DataFrame, placebo: pl.DataFrame) -> pl.DataFrame:
    combos = [("L0.05", "E1"), ("L0.05", "E3"), ("touch", "E1")]
    cond = pl.any_horizontal(*[(pl.col("entry") == e) & (pl.col("exit") == x) for e, x in combos])
    parts: list[pl.DataFrame] = []
    for df in (study, placebo):
        d = df.filter(cond, pl.col("scenario") == "base", pl.col("fyear").is_in(YEARS))
        parts.append(
            pl.concat(
                [
                    d.with_columns(tf_g=pl.col("tf")),
                    d.with_columns(tf_g=pl.lit("all")),
                ]
            )
        )
    t = diff_table(parts[0], parts[1], ["tf_g", "entry", "exit"])
    t = t.rename({"a_bp": "study_bp", "b_bp": "placebo_bp", "a_u": "study_u", "b_u": "placebo_u"})
    t = t.rename({"trades_a": "study_trades", "trades_b": "placebo_trades"})
    return sort_tf(t, "tf_g", ["entry", "exit"])


def table5(study: pl.DataFrame) -> pl.DataFrame:
    sub = study.filter(
        pl.col("major"), pl.col("entry").is_in(BASE_ENTRIES), pl.col("exit").is_in(BASE_EXITS),
        pl.col("scenario") == "base", pl.col("fyear").is_in(YEARS),
    )  # fmt: skip
    per_ma = summarize(sub, ["tf", "n", "entry", "exit"]).with_columns(
        tf_g=pl.col("tf"), n=pl.col("n").cast(pl.Int64)
    )
    pooled = (
        summarize(
            pl.concat(
                [
                    sub.with_columns(tf_g=pl.col("tf")),
                    sub.with_columns(tf_g=pl.lit("all")),
                ]
            ),
            ["tf_g", "entry", "exit"],
        )
        .with_columns(tf=pl.col("tf_g"), n=pl.lit(0, dtype=pl.Int64))
        .select(per_ma.columns)
    )
    t = pl.concat([per_ma, pooled]).rename({"n": "ma_n"})
    fr = fill_rates(["tf", "n", "entry"], extra=major_cands()).with_columns(
        pl.col("n").cast(pl.Int64)
    )
    tot = pl.concat([fr, fr.with_columns(tf=pl.lit("all"))])
    pooled_fr = tot.group_by("tf", "entry").agg(pl.col("candidates").sum(), pl.col("fills").sum())
    pooled_fr = pooled_fr.with_columns(
        n=pl.lit(0, dtype=pl.Int64), fill_rate=pl.col("fills") / pl.col("candidates")
    )
    fr = pl.concat([fr, pooled_fr.select(fr.columns)]).rename({"n": "ma_n"})
    t = t.join(fr, on=["tf", "ma_n", "entry"], how="left")
    return sort_tf(t, "tf", ["ma_n", "entry", "exit"]).drop("tf_g")


def major_cands() -> pl.DataFrame:
    return pl.read_parquet(M2_CACHE / "trades.parquet", columns=["cid", "major"]).unique("cid")


# --- stage 2: matched random entries -----------------------------------------------------------


def _hours(tf: str) -> int:
    return hold_us(tf) // (3600 * 1_000_000)


def _empty_controls() -> pl.DataFrame:
    schema: dict[str, pl.DataType | type[pl.DataType]] = {
        "cid": pl.String,
        "n_pool": pl.Int32,
        "n_ctrl": pl.Int32,
    }
    for h in CTRL_EXITS:
        schema[f"c_{h}"] = pl.Float64
        schema[f"cu_{h}"] = pl.Float64
    return pl.DataFrame(schema=schema)


def control_part(symbol: str) -> pl.DataFrame:
    """Controls for one coin's targets: per candidate the number drawn and the mean gross return
    (and in units of the control's own u) of the controls at T/2, T and min(2T, 28 d)."""
    tg = pl.read_parquet(M2_CACHE / "targets.parquet").filter(pl.col("symbol") == symbol)
    if tg.is_empty():
        return _empty_controls()
    c5 = read_5m(symbol, END_US)
    bars = read_bars(symbol)
    t = c5["t"].to_numpy().astype(np.int64)
    op = c5["open"].to_numpy().astype(np.float64)
    close = c5["close"].to_numpy().astype(np.float64)
    n = len(t)
    known_until = int(t[-1]) + FIVE
    ended = known_until <= END_US - 2 * DAY
    frame = pl.DataFrame({"t": t})
    day_bars = bars["1d"].select("t", "atr_d", "liq")
    daily = frame.join_asof(day_bars, on="t")
    atr = daily["atr_d"].to_numpy().astype(np.float64)
    liq = daily["liq"].to_numpy().astype(np.float64)
    levels: list[F64] = []
    for tf in TF_NAMES:
        cols = [f"L{k}" for k in MA_LENGTHS[tf]]
        j = frame.join_asof(bars[tf].select("t", *cols), on="t")
        levels.extend(j[c].to_numpy().astype(np.float64) for c in cols)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.abs(close[None, :] / np.stack(levels) - 1.0)
    mindist = np.where(np.isnan(dist), np.inf, dist).min(axis=0)
    month = t.astype("datetime64[us]").astype("datetime64[M]").astype(np.int64)
    entry_ok = np.zeros(n, dtype=bool)
    entry_ok[:-1] = (t[1:] == t[:-1] + FIVE) & (t[:-1] + FIVE < HOLD_US)
    dts = pl.DataFrame({"dt": pl.Series(t).cast(pl.Datetime("us")), "close": close})
    rng = np.random.default_rng([SEED, zlib.crc32(symbol.encode())])
    outs: list[pl.DataFrame] = []
    for tf in TF_NAMES:
        sub = tg.filter(pl.col("tf") == tf)
        if sub.is_empty():
            continue
        hrs = _hours(tf)
        rollmax = (
            dts.select(pl.col("close").rolling_max_by("dt", window_size=f"{hrs}h", closed="right"))[
                "close"
            ]
            .to_numpy()
            .astype(np.float64)
        )
        u_c = atr * np.sqrt(TIMEFRAMES[tf].hold_days)  # each candle's own day's u
        with np.errstate(invalid="ignore", divide="ignore"):
            depth_u = (close / rollmax - 1.0) / u_c
            valid = (
                entry_ok & np.isfinite(depth_u) & (u_c > 0) & (liq >= MIN_DAILY_QUOTE_VOLUME)
                & (mindist > FAR_FROM_MA * u_c)
            )  # fmt: skip
        pool = np.flatnonzero(valid)
        comp = month[pool] * 10_000.0 + np.clip(depth_u[pool], -4000, 4000)
        order = np.argsort(comp, kind="stable")
        pool, comp = pool[order], comp[order]
        ci_ = np.searchsorted(t, sub["trigger_t"].to_numpy())
        assert np.array_equal(t[ci_], sub["trigger_t"].to_numpy()), "trigger candle exists"
        cand_depth = (close[ci_] / rollmax[ci_] - 1.0) / sub["u"].to_numpy()
        key = month[ci_] * 10_000.0 + np.clip(cand_depth, -4000, 4000)
        lo = np.searchsorted(comp, key - DEPTH_TOL, side="left")
        hi = np.searchsorted(comp, key + DEPTH_TOL, side="right")
        owner: list[int] = []
        picked: list[int] = []
        for i in range(sub.height):
            cnt = int(hi[i] - lo[i])
            if cnt <= 0:
                continue
            pick = (
                np.arange(lo[i], hi[i])
                if cnt <= CTRL_PER_CAND
                else lo[i] + rng.choice(cnt, CTRL_PER_CAND, replace=False)
            )
            owner.extend([i] * len(pick))
            picked.extend(pool[pick].tolist())
        own = np.asarray(owner, dtype=np.int64)
        idx = np.asarray(picked, dtype=np.int64)
        res: dict[str, pl.Series | F64] = {
            "n_pool": (hi - lo).astype(np.int64),
            "n_ctrl": np.bincount(own, minlength=sub.height),
        }
        ent_p, ent_t = op[idx + 1], t[idx + 1]
        for h, ex in CTRL_EXITS.items():
            deadline = ent_t + exit_hold_us(tf, ex)
            jx = np.searchsorted(t, deadline, side="left") - 1
            ok = ended | (deadline <= known_until)
            g = np.where(ok, close[jx] / ent_p - 1.0, np.nan)
            gu = g / u_c[idx]
            good = ~np.isnan(g)
            cn = np.bincount(own[good], minlength=sub.height).astype(np.float64)
            for name, v in ((f"c_{h}", g), (f"cu_{h}", gu)):
                sm = np.bincount(own[good], weights=v[good], minlength=sub.height)
                with np.errstate(invalid="ignore", divide="ignore"):
                    res[name] = np.where(cn > 0, sm / cn, np.nan)
        outs.append(
            pl.DataFrame({"cid": sub["cid"], **res}).with_columns(
                pl.col("n_pool", "n_ctrl").cast(pl.Int32)
            )
        )
    return pl.concat(outs) if outs else _empty_controls()


def _control_work(symbol: str) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    df = control_part(symbol)
    path = M2_CACHE / "controls" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)
    return symbol, df.height, time.perf_counter() - t0


def run_controls(trades: pl.DataFrame, workers: int) -> pl.DataFrame:
    tg = (
        trades.filter(
            ~pl.col("placebo"), pl.col("scenario") == "base", pl.col("exit") == "E1",
            pl.col("entry").is_in(BASE_ENTRIES),
        )
        .select("cid", "symbol", "tf", "trigger_t", "u")
    )  # fmt: skip
    tg.write_parquet(M2_CACHE / "targets.parquet")
    symbols = sorted(tg["symbol"].unique().to_list())
    todo = [s for s in symbols if not (M2_CACHE / "controls" / f"{s}.parquet").exists()]
    print(f"controls: {len(symbols)} coins, {len(todo)} to do", flush=True)
    t0 = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            futs = [pool.submit(_control_work, s) for s in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                s, rows, secs = fut.result()
                print(f"[{i}/{len(todo)}] {s} {rows} {secs:.1f}s ({time.perf_counter() - t0:.0f}s)",
                      flush=True)  # fmt: skip
    parts = [pl.read_parquet(M2_CACHE / "controls" / f"{s}.parquet") for s in symbols]
    return pl.concat(parts).fill_nan(None)


def table4(study: pl.DataFrame, controls: pl.DataFrame) -> pl.DataFrame:
    base = study.filter(pl.col("scenario") == "base", pl.col("entry").is_in(BASE_ENTRIES))
    rows: list[pl.DataFrame] = []
    for h, ex in CTRL_EXITS.items():
        e = base.filter(pl.col("exit") == ex).select(
            "cid", "tf", "entry", "day", "u", g=pl.col("gross"), gu=pl.col("gross") / pl.col("u")
        )
        j = e.join(controls.select("cid", "n_ctrl", f"c_{h}", f"cu_{h}"), on="cid", how="left")
        rows.append(
            j.rename({f"c_{h}": "c", f"cu_{h}": "cu"}).with_columns(hold=pl.lit(h), exit=pl.lit(ex))
        )
    allr = pl.concat(rows).with_columns(has=pl.col("c").is_not_null())
    allr = pl.concat([allr.with_columns(tf_g=pl.col("tf")), allr.with_columns(tf_g=pl.lit("all"))])
    gkeys = ["tf_g", "entry", "hold", "exit"]
    cnt = allr.group_by(gkeys).agg(
        candidates=pl.len(), no_control=(pl.col("n_ctrl").fill_null(0) == 0).sum()
    )
    pairs = allr.filter(pl.col("has")).with_columns(
        d=(pl.col("g") - pl.col("c")) * BP,
        gb=pl.col("g") * BP,
        cb=pl.col("c") * BP,
        du=pl.col("gu") - pl.col("cu"),
    )
    agg = pairs.group_by(gkeys).agg(
        pairs=pl.len(),
        cand_bp=pl.col("gb").mean(),
        ctrl_bp=pl.col("cb").mean(),
        diff_bp=pl.col("d").mean(),
        cand_u=pl.col("gu").mean(),
        ctrl_u=pl.col("cu").mean(),
        diff_u=pl.col("du").mean(),
    )
    perday = (
        pairs.group_by([*gkeys, "day"])
        .agg(sd=pl.col("d").sum(), su=pl.col("du").sum(), c=pl.len().cast(pl.Float64))
        .sort("day")
    )
    ci_rows: list[dict[str, object]] = []
    for key, g in perday.partition_by(gkeys, as_dict=True, include_key=False).items():
        sums = np.column_stack([g["sd"], g["su"]])
        c = np.column_stack([g["c"], g["c"]])
        row: dict[str, object] = dict(zip(gkeys, key, strict=True))
        if len(g) >= 2:
            r = boot_ratios(sums, c)
            row["diff_lo_bp"], row["diff_hi_bp"] = ci(r[:, 0])
            row["diff_lo_u"], row["diff_hi_u"] = ci(r[:, 1])
        ci_rows.append(row)
    ci_df = pl.DataFrame(ci_rows, infer_schema_length=None)
    t = cnt.join(agg, on=gkeys, how="left").join(ci_df, on=gkeys, how="left")
    t = t.rename({"tf_g": "tf"})
    hrank = {h: i for i, h in enumerate(CTRL_EXITS)}
    t = t.with_columns(_b=pl.col("hold").replace_strict(hrank))
    return sort_tf(t, "tf", ["entry", "_b"]).drop("_b")


# --- output ------------------------------------------------------------------------------------


def fmt(v: object, digits: int = 1) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return f"{v:,.{digits}f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def md_table(df: pl.DataFrame, cols: dict[str, str], digits: dict[str, int] | None = None) -> str:
    """Markdown table of selected columns, `{column: header}`; `digits` per column (default 1)."""
    digits = digits or {}
    lines = ["| " + " | ".join(cols.values()) + " |", "|" + " --- |" * len(cols)]
    for r in df.iter_rows(named=True):
        cells = [fmt(r[c], digits.get(c, 1)) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ci_cell(df: pl.DataFrame, lo: str, hi: str, name: str) -> pl.DataFrame:
    return df.with_columns(
        pl.format("[{}, {}]", pl.col(lo).round(0).cast(pl.Int64, strict=False),
                  pl.col(hi).round(0).cast(pl.Int64, strict=False)).alias(name)
    )  # fmt: skip


def write_markdown(
    t1: pl.DataFrame,
    t2: pl.DataFrame,
    t3: pl.DataFrame,
    t4: pl.DataFrame,
    t5: pl.DataFrame,
    t6: pl.DataFrame,
) -> None:
    out: list[str] = []
    # Table 1: the primary rows only (limit 0.05 and touch, E1 and E3), all 15 study MAs
    a = ci_cell(t1.filter(pl.col("entry").is_in(BASE_ENTRIES), pl.col("exit").is_in(BASE_EXITS)),
                "ci_lo_bp", "ci_hi_bp", "ci")  # fmt: skip
    a = a.with_columns(ma=pl.col("tf") + " " + pl.col("n").cast(pl.String))
    out += ["### Table 1 (excerpt): study MA x entry x exit, 2020-24, base scenario\n",
            md_table(a, {"ma": "MA", "entry": "Entry", "exit": "Exit", "candidates": "Cands",
                         "fill_rate": "Fill rate", "trades": "Trades", "mean_bp": "Mean bp",
                         "ci": "95% CI", "median_bp": "Median bp", "win_rate": "Win",
                         "mean_u": "Net/u", "mean_hold_d": "Hold d", "stress_mean_bp": "Stress bp"},
                     {"fill_rate": 2, "win_rate": 2, "mean_u": 3, "candidates": 0, "trades": 0,
                      "mean_hold_d": 2}), ""]  # fmt: skip
    b = ci_cell(t2, "ci_lo_bp", "ci_hi_bp", "ci")
    out += ["### Table 2 (excerpt, exit E1): per timeframe and fill year\n",
            md_table(b.filter(pl.col("tf_g") != "all", pl.col("exit") == "E1"),
                     {"tf_g": "TF", "year_g": "Year", "entry": "Entry", "exit": "Exit",
                      "trades": "Trades", "mean_bp": "Mean bp", "ci": "95% CI",
                      "win_rate": "Win", "mean_u": "Net/u", "fill_rate": "Fill rate"},
                     {"win_rate": 2, "mean_u": 3, "trades": 0, "fill_rate": 2}), ""]  # fmt: skip
    c = ci_cell(t3, "diff_lo_bp", "diff_hi_bp", "ci_bp")
    c = ci_cell(c, "diff_lo_u", "diff_hi_u", "ci_u")
    out += ["### Table 3: study MAs minus placebo MAs (pooled within timeframe)\n",
            md_table(c, {"tf_g": "TF", "entry": "Entry", "exit": "Exit",
                         "study_trades": "Study n", "placebo_trades": "Placebo n",
                         "study_bp": "Study bp", "placebo_bp": "Placebo bp", "diff_bp": "Diff bp",
                         "ci_bp": "95% CI", "diff_u": "Diff net/u", "diff_lo_u": "CI lo",
                         "diff_hi_u": "CI hi"},
                     {"study_trades": 0, "placebo_trades": 0, "diff_u": 3, "diff_lo_u": 3,
                      "diff_hi_u": 3}), ""]  # fmt: skip
    d = ci_cell(t4, "diff_lo_bp", "diff_hi_bp", "ci")
    out += ["### Table 4: study entries against matched random entries (gross, no costs)\n",
            md_table(d, {"tf": "TF", "entry": "Entry", "hold": "Exit at", "candidates": "Cands",
                         "no_control": "No control", "pairs": "Pairs", "cand_bp": "Cand bp",
                         "ctrl_bp": "Ctrl bp", "diff_bp": "Diff bp", "ci": "95% CI",
                         "diff_u": "Diff u", "cand_u": "Cand u", "ctrl_u": "Ctrl u"},
                     {"candidates": 0, "no_control": 0, "pairs": 0, "diff_u": 3, "cand_u": 3,
                      "ctrl_u": 3}), ""]  # fmt: skip
    f = ci_cell(t6.filter(pl.col("exit") == "E1"), "ci_lo_bp", "ci_hi_bp", "ci")
    out += ["### Table 6 (excerpt): all study MAs of a timeframe pooled, exit E1, every entry\n",
            md_table(f, {"tf": "TF", "entry": "Entry", "trades": "Trades", "mean_bp": "Mean bp",
                         "ci": "95% CI", "median_bp": "Median bp", "win_rate": "Win",
                         "mean_u": "Net/u", "stress_mean_bp": "Stress bp"},
                     {"trades": 0, "win_rate": 2, "mean_u": 3}), ""]  # fmt: skip
    e = ci_cell(t5, "ci_lo_bp", "ci_hi_bp", "ci")
    e = e.with_columns(ma=pl.when(pl.col("ma_n") == 0).then(pl.lit("pooled"))
                       .otherwise(pl.col("ma_n").cast(pl.String)))  # fmt: skip
    title5 = "### Table 5 (excerpt, pooled MAs): majors cut (BTC, ETH, top 20 `liq` that day)\n"
    e = e.filter(pl.col("ma_n") == 0)
    out += [title5,
            md_table(e, {"tf": "TF", "ma": "MA n", "entry": "Entry", "exit": "Exit",
                         "candidates": "Cands", "trades": "Trades", "mean_bp": "Mean bp",
                         "ci": "95% CI", "win_rate": "Win", "mean_u": "Net/u"},
                     {"candidates": 0, "trades": 0, "win_rate": 2, "mean_u": 3}), ""]  # fmt: skip
    (OUT / "tables.md").write_text("\n".join(out))


def charts(t2: pl.DataFrame, t3: pl.DataFrame, t4: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    d = t3.filter(pl.col("tf_g") != "all")
    labels, mids = [], []
    for i, r in enumerate(d.iter_rows(named=True)):
        lo_err = r["diff_bp"] - r["diff_lo_bp"]
        hi_err = r["diff_hi_bp"] - r["diff_bp"]
        ax.errorbar(r["diff_bp"], i, xerr=[[lo_err], [hi_err]], fmt="o", color="C0")
        labels.append(f"{r['tf_g']} {r['entry']} {r['exit']}")
        mids.append(i)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(mids, labels, fontsize=7)
    ax.set_title("Study minus placebo MAs: mean net bp (95% CI)")
    ax = axes[1]
    for (entry, ex), g in t2.filter(pl.col("tf_g") == "all", pl.col("year_g") != "all").group_by(
        ["entry", "exit"], maintain_order=True
    ):  # fmt: skip
        ax.plot(g["year_g"].to_list(), g["mean_bp"].to_list(), marker="o", label=f"{entry} {ex}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("All study MAs: mean net bp per fill year")
    ax.legend(fontsize=7)
    ax = axes[2]
    d4 = t4.filter(pl.col("tf") != "all", pl.col("hold") == "full")
    for k, (entry, g) in enumerate(d4.group_by("entry", maintain_order=True)):
        x = np.arange(g.height) + 0.2 * k
        ax.errorbar(x, g["diff_bp"].to_numpy(),
                    yerr=[(g["diff_bp"] - g["diff_lo_bp"]).to_numpy(),
                          (g["diff_hi_bp"] - g["diff_bp"]).to_numpy()],
                    fmt="o", label=str(entry[0]))  # fmt: skip
        ax.set_xticks(np.arange(g.height), g["tf"].to_list())
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Candidate minus matched control, gross bp at T")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "m2_summary.png", dpi=110)


def holdout(rebuild: bool) -> None:
    """Tables 2 and 3 on fills from 2025-01-01 to the data end (module docstring)."""
    global HOLD_US, LO_US, YEARS, TRADES_FILE
    LO_US, HOLD_US = HOLD_US, END_US
    YEARS = (2025, 2026)
    TRADES_FILE = "trades_holdout.parquet"
    out = RESULTS / "m6"
    out.mkdir(parents=True, exist_ok=True)
    trades = load_trades(rebuild)
    assert int(trades["entry_t"].min()) >= LO_US  # type: ignore[arg-type]
    study = trades.filter(~pl.col("placebo"))
    placebo = trades.filter(pl.col("placebo"))
    table2(study).write_csv(out / "m2_t2_timeframe_year.csv")
    table3(study, placebo).write_csv(out / "m2_t3_study_vs_placebo.csv")
    print(f"wrote {out}/m2_t2_timeframe_year.csv and m2_t3_study_vs_placebo.csv", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true", help="rebuild trades and controls")
    ap.add_argument("--holdout", action="store_true", help="M6 reveal: tables 2 and 3 on 2025-26")
    args = ap.parse_args()
    if args.holdout:
        holdout(args.rebuild)
        return
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.rebuild:
        for f in (M2_CACHE / "controls").glob("*.parquet"):
            f.unlink()
    trades = load_trades(args.rebuild)
    assert int(trades["entry_t"].max()) < HOLD_US  # type: ignore[arg-type]
    study = trades.filter(~pl.col("placebo"))
    placebo = trades.filter(pl.col("placebo"))
    base = study.filter(pl.col("scenario") == "base", pl.col("fyear").is_in(YEARS))
    stress = study.filter(pl.col("scenario") == "stress", pl.col("fyear").is_in(YEARS))
    print(f"trades loaded {time.perf_counter() - t0:.0f}s", flush=True)
    t1 = table1(base, stress)
    t1.write_csv(OUT / "t1_ma_entry_exit.csv")
    print(f"table1 {time.perf_counter() - t0:.0f}s", flush=True)
    t2 = table2(study)
    t2.write_csv(OUT / "t2_timeframe_year.csv")
    t3 = table3(study, placebo)
    t3.write_csv(OUT / "t3_study_vs_placebo.csv")
    print(f"table2,3 {time.perf_counter() - t0:.0f}s", flush=True)
    t6 = table6(base, stress)
    t6.write_csv(OUT / "t6_timeframe_all_combos.csv")
    t5 = table5(study)
    t5.write_csv(OUT / "t5_majors.csv")
    controls = run_controls(trades, args.workers)
    t4 = table4(study, controls)
    t4.write_csv(OUT / "t4_matched_controls.csv")
    write_markdown(t1, t2, t3, t4, t5, t6)
    charts(t2, t3, t4)
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
