"""M3: single-factor results on discovery (2020-01-31 .. 2023-12-01).

    uv run --group research python m3_single.py

Reads `cache/labels.parquet` (discovery rows only), `cache/factors.parquet` and
`cache/exclusions.parquet`. It never opens `cache/sealed/`: labels are filtered to
`split == "discovery"` and joined with an inner join, so no entry with D from 2024 can enter.

Writes `results/m3_single.csv` (long format, every number), `results/m3_single.md` (generated from
that csv), `results/m3_redundancy.csv` (Spearman matrix of the factors) and
`results/m3_shuffle.csv` (the hit10-shuffled-within-day null for the coin factors).

Conventions
- Rates, lifts, excesses and returns are fractions (0.05 = 5 percentage points) in the csv.
- Lift = bucket rate minus the base rate of the same sample (entries with a non-null factor, in
  the same scope: all years, or one year). For within-day excess the base is the same-day rate of
  the entries with a non-null factor (days with at least 30 of them).
- Buckets are rank based (average rank, bucket = floor(pct * n_buckets)), so they are equal-sized
  unless the factor has ties. Binary flags use their two values instead.
- Market factors: quintiles over days (each day once); the assignment is fixed from all
  discovery days and reused for the per-year view. Coin factors: deciles over entries, recomputed
  within year for the per-year view.
- Bootstrap: 500 draws of the calendar months with replacement, seed fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl
from common import (
    CACHE,
    DISCOVERY_END_US,
    F64,
    I64,
    LABELS,
    RESULTS,
    SEALED_START_US,
)

FACTORS_PQ = CACHE / "factors.parquet"
EXCLUSIONS_PQ = CACHE / "exclusions.parquet"

YEARS = (2020, 2021, 2022, 2023)
LABELS3 = ("hit10", "hit20", "hit_ns")
KNAMES = (*LABELS3, "ret10")  # the four outcome columns, in this order
YCOLS = ("hit10", "hit20", "hit_ns", "ret_10")
FLAGS = ("c_touch_ma", "c_below_200d")
MIN_DAY_ENTRIES = 30
MIN_QUINTILE_DAYS = 20
N_BOOT = 500
N_SHUFFLE = 20
SEED = 20261114
CLUSTER_RHO = 0.8

type B64 = npt.NDArray[np.bool_]


# --- data ------------------------------------------------------------------------------------


@dataclass
class Frame:
    df: pl.DataFrame
    factors: list[str]
    market: list[str]
    coin: list[str]
    day: I64  # day index per entry (0..n_days-1)
    n_days: int
    day_us: I64  # d_us per day index
    month: I64  # month index per entry
    n_months: int
    year_i: I64  # year - 2020 per entry
    Y: F64  # (n, 4): hit10, hit20, hit_ns, ret_10
    peak30: F64


def load() -> Frame:
    labels = pl.read_parquet(LABELS).filter(pl.col("split") == "discovery")
    labels = labels.filter(pl.col("d_us") <= DISCOVERY_END_US)
    fac = pl.read_parquet(FACTORS_PQ)
    excl = pl.read_parquet(EXCLUSIONS_PQ).select("symbol", "d_us")
    names = [c for c in fac.columns if c.startswith(("m_", "c_"))]
    fac = fac.with_columns(pl.col(names).fill_nan(None))
    n_disc = labels.height
    n_excl = labels.join(excl, on=["symbol", "d_us"], how="semi").height
    df = labels.join(fac, on=["symbol", "d_us"], how="inner").join(
        excl, on=["symbol", "d_us"], how="anti"
    )
    # self-checks: discovery minus exclusions, nothing from the sealed period
    assert df.height == n_disc - n_excl, (df.height, n_disc, n_excl)
    assert int(df.select(pl.col("d_us").max()).item()) < SEALED_START_US
    assert int(df.select(pl.col("d_us").max()).item()) <= DISCOVERY_END_US
    assert set(df["split"].unique().to_list()) == {"discovery"}
    n_all = df.height
    df = df.filter(pl.col("hit10").is_not_null()).sort("d_us", "symbol")
    n_no_outcome = n_all - df.height
    df = df.with_columns(
        month=pl.from_epoch("d_us", time_unit="us").dt.year() * 12
        + pl.from_epoch("d_us", time_unit="us").dt.month()
        - 1
    )
    day_us, day = np.unique(df["d_us"].to_numpy(), return_inverse=True)
    mon_u, mon = np.unique(df["month"].to_numpy(), return_inverse=True)
    year_i = df["year"].to_numpy().astype(np.int64) - YEARS[0]
    assert year_i.min() == 0
    assert year_i.max() == len(YEARS) - 1
    Y = np.column_stack(
        [df[c].cast(pl.Float64).to_numpy().astype(np.float64) for c in YCOLS]
    ).astype(np.float64)
    assert not np.isnan(Y[:, :3]).any()
    print(
        f"discovery {n_disc}, excluded {n_excl}, used {n_all}, "
        f"without outcome {n_no_outcome}, analysed {df.height}"
    )
    return Frame(
        df=df.with_columns(day=pl.Series(day.astype(np.int64))),
        factors=names,
        market=[c for c in names if c.startswith("m_")],
        coin=[c for c in names if c.startswith("c_")],
        day=day.astype(np.int64),
        n_days=len(day_us),
        day_us=day_us.astype(np.int64),
        month=mon.astype(np.int64),
        n_months=len(mon_u),
        year_i=year_i,
        Y=Y,
        peak30=df["peak30"].to_numpy().astype(np.float64),
    )


# --- aggregation -----------------------------------------------------------------------------


@dataclass
class Agg:
    n: F64  # (G, B) entries
    s: F64  # (G, B, K) sums of the outcome
    c: F64  # (G, B, K) non-null counts


def agg(b: I64, nb: int, Y: F64, grp: I64, ng: int) -> Agg:
    """Per group and bucket: entry count, outcome sums and counts. Entries with b < 0 are out."""
    ok = b >= 0
    idx = grp[ok] * nb + b[ok]
    size = ng * nb
    k = Y.shape[1]
    y = Y[ok]
    v = ~np.isnan(y)
    yz = np.where(v, y, 0.0)
    n = np.bincount(idx, minlength=size).astype(np.float64).reshape(ng, nb)
    s = np.empty((ng, nb, k))
    c = np.empty((ng, nb, k))
    for j in range(k):
        s[:, :, j] = np.bincount(idx, weights=yz[:, j], minlength=size).reshape(ng, nb)
        c[:, :, j] = np.bincount(idx, weights=v[:, j].astype(np.float64), minlength=size).reshape(
            ng, nb
        )
    return Agg(n, s, c)


def rates(a: Agg) -> F64:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(a.c > 0, a.s / a.c, np.nan)


def bases(a: Agg) -> F64:
    cs = a.c.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cs > 0, a.s.sum(axis=1) / cs, np.nan)


def boot_weights(n_months: int) -> F64:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_months, size=(N_BOOT, n_months))
    w = np.zeros((N_BOOT, n_months))
    for i in range(N_BOOT):
        w[i] = np.bincount(draws[i], minlength=n_months)
    return w


def boot_lifts(b: I64, nb: int, Y: F64, fr: Frame, w: F64) -> F64:
    """(N_BOOT, nb, K) bucket minus base, resampling calendar months."""
    a = agg(b, nb, Y, fr.month, fr.n_months)
    s = np.einsum("dm,mbk->dbk", w, a.s)
    c = np.einsum("dm,mbk->dbk", w, a.c)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(c > 0, s / c, np.nan)
        base = np.where(c.sum(axis=1) > 0, s.sum(axis=1) / c.sum(axis=1), np.nan)
    return r - base[:, None, :]


# --- ranks -----------------------------------------------------------------------------------


def rank_avg(x: F64) -> F64:
    """Average ranks of x with NaN kept as NaN."""
    return pl.Series(x).fill_nan(None).rank("average").cast(pl.Float64).to_numpy()


def spearman(x: F64, y: F64) -> float:
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan")
    rx = rank_avg(x[ok])
    ry = rank_avg(y[ok])
    return float(np.corrcoef(rx, ry)[0, 1])


def bucket_of(pct: F64, nb: int) -> I64:
    ok = ~np.isnan(pct)
    b = np.minimum(np.floor(np.nan_to_num(pct) * nb), nb - 1).astype(np.int64)
    return np.where(ok, b, -1).astype(np.int64)


def pct_rank(df: pl.DataFrame, col: str, over: str | None) -> tuple[F64, F64]:
    """Percentile (rank - 0.5) / n within `over` (all rows if None), and the group's non-null n."""
    r_e = pl.col(col).rank("average").cast(pl.Float64)
    n_e = pl.col(col).count().cast(pl.Float64)
    if over is not None:
        r_e, n_e = r_e.over(over), n_e.over(over)
    out = df.select(r=r_e, n=n_e)
    r = out["r"].to_numpy().astype(np.float64)
    n = out["n"].to_numpy().astype(np.float64)
    return (r - 0.5) / n, n


# --- results store ---------------------------------------------------------------------------


class Rows:
    def __init__(self) -> None:
        self.cols: dict[str, list[object]] = {
            k: []
            for k in ("view", "factor", "scope", "bucket", "metric", "value", "n_entries", "n_days")
        }

    def add(
        self,
        view: str,
        factor: str,
        scope: str,
        bucket: str,
        metric: str,
        value: float,
        n_entries: float | None = None,
        n_days: float | None = None,
    ) -> None:
        if isinstance(value, float) and math.isnan(value):
            return
        self.cols["view"].append(view)
        self.cols["factor"].append(factor)
        self.cols["scope"].append(scope)
        self.cols["bucket"].append(bucket)
        self.cols["metric"].append(metric)
        self.cols["value"].append(float(value))
        self.cols["n_entries"].append(None if n_entries is None else int(n_entries))
        self.cols["n_days"].append(None if n_days is None else int(n_days))

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            self.cols,
            schema={
                "view": pl.String,
                "factor": pl.String,
                "scope": pl.String,
                "bucket": pl.String,
                "metric": pl.String,
                "value": pl.Float64,
                "n_entries": pl.Int64,
                "n_days": pl.Int64,
            },
        )


SCOPES = ("all", *(str(y) for y in YEARS))


def emit_buckets(
    rows: Rows,
    view: str,
    factor: str,
    nb: int,
    a_all: Agg,
    a_year: Agg,
    excess: bool,
    days: F64 | None = None,
) -> None:
    """One row per scope x bucket x outcome: rate (or excess) and the entry count."""
    for gi, scope in enumerate(SCOPES):
        a = a_all if gi == 0 else a_year
        g = 0 if gi == 0 else gi - 1
        r = rates(a)[g]
        for j in range(nb):
            n_e = a.n[g, j]
            if n_e == 0:
                continue
            nd = None if days is None else float(days[gi, j])
            for k, name in enumerate(KNAMES):
                if k == 3:
                    metric = "ret10_excess" if excess else "ret10_mean"
                else:
                    metric = f"{name}_excess" if excess else f"{name}_rate"
                rows.add(view, factor, scope, str(j + 1), metric, float(r[j, k]), n_e, nd)


def emit_base(rows: Rows, view: str, factor: str, a_all: Agg, a_year: Agg) -> None:
    for gi, scope in enumerate(SCOPES):
        a = a_all if gi == 0 else a_year
        g = 0 if gi == 0 else gi - 1
        base = bases(a)[g]
        for k, name in enumerate(KNAMES):
            metric = "ret10_mean" if k == 3 else f"{name}_rate"
            rows.add(view, factor, scope, "base", metric, float(base[k]), a.n[g].sum())


def emit_lifts(
    rows: Rows,
    view: str,
    factor: str,
    nb: int,
    a_all: Agg,
    a_year: Agg,
    ci: F64,
) -> None:
    """Top and bottom bucket lift over the base, per scope; the bootstrap interval on `all`."""
    for gi, scope in enumerate(SCOPES):
        a = a_all if gi == 0 else a_year
        g = 0 if gi == 0 else gi - 1
        lift = rates(a)[g] - bases(a)[g][None, :]
        for end, j in (("top", nb - 1), ("bottom", 0)):
            for k, name in enumerate(KNAMES):
                rows.add(view, factor, scope, end, f"lift_{name}", float(lift[j, k]), a.n[g, j])
    for end, j in (("top", nb - 1), ("bottom", 0)):
        for k, name in enumerate(KNAMES):
            lo, hi = np.nanpercentile(ci[:, j, k], [2.5, 97.5])
            rows.add(view, factor, "all", end, f"ci_lo_lift_{name}", float(lo))
            rows.add(view, factor, "all", end, f"ci_hi_lift_{name}", float(hi))
    sp = ci[:, nb - 1, 0] - ci[:, 0, 0]
    lo, hi = np.nanpercentile(sp, [2.5, 97.5])
    rows.add(view, factor, "all", "spread", "ci_lo_spread_hit10", float(lo))
    rows.add(view, factor, "all", "spread", "ci_hi_spread_hit10", float(hi))
    r_all = rates(a_all)[0]
    rows.add(view, factor, "all", "spread", "spread_hit10", float(r_all[nb - 1, 0] - r_all[0, 0]))


def beat_years(lift_year: F64, ok_year: B64) -> tuple[int, int]:
    """(years where the lift is positive, years that count)."""
    beats = ok_year & (lift_year > 0)
    return int(beats.sum()), int(ok_year.sum())


# --- market factors --------------------------------------------------------------------------


def market_view(
    fr: Frame, rows: Rows, w: F64, factor: str, flags: dict[str, dict[str, float]]
) -> None:
    nb = 5
    df = fr.df
    dd = (
        df.select("day", "year", "hit10", pl.col(factor))
        .group_by("day")
        .agg(
            v=pl.col(factor).drop_nulls().first(),
            nv=pl.col(factor).drop_nulls().n_unique(),
            year=pl.col("year").first(),
            h=pl.col("hit10").mean(),
            n=pl.len(),
        )
        .sort("day")
    )
    assert dd["nv"].max() <= 1  # type: ignore[operator]  # a market factor is one value a day
    v = dd["v"].to_numpy().astype(np.float64)
    day_pct = (
        dd.select(r=pl.col("v").rank("average").cast(pl.Float64), n=pl.col("v").count())
        .with_columns(p=(pl.col("r") - 0.5) / pl.col("n"))["p"]
        .to_numpy()
        .astype(np.float64)
    )
    bd = bucket_of(day_pct, nb)  # per day (index = day number, dd sorted by day)
    assert len(bd) == fr.n_days
    nd_valid = int((bd >= 0).sum())
    cnt_days = np.bincount(bd[bd >= 0], minlength=nb)
    if len(np.unique(v[~np.isnan(v)])) > 500:
        assert cnt_days.min() >= 0.8 * nd_valid / nb, cnt_days
        assert cnt_days.max() <= 1.2 * nd_valid / nb, cnt_days
    b = bd[fr.day]
    a_all = agg(b, nb, fr.Y, np.zeros(len(b), dtype=np.int64), 1)
    a_year = agg(b, nb, fr.Y, fr.year_i, len(YEARS))
    days = np.zeros((len(SCOPES), nb))
    day_year = dd["year"].to_numpy().astype(np.int64) - YEARS[0]
    for j in range(nb):
        days[0, j] = float((bd == j).sum())
        for yi in range(len(YEARS)):
            days[yi + 1, j] = float(((bd == j) & (day_year == yi)).sum())
    ci = boot_lifts(b, nb, fr.Y, fr, w)
    emit_buckets(rows, "mkt_quintile", factor, nb, a_all, a_year, False, days)
    emit_base(rows, "mkt_quintile", factor, a_all, a_year)
    emit_lifts(rows, "mkt_summary", factor, nb, a_all, a_year, ci)
    h_day = dd["h"].to_numpy().astype(np.float64)
    rows.add(
        "mkt_summary", factor, "all", "", "spearman_day_hit10", spearman(v, h_day), None, nd_valid
    )
    # stability: best quintile by overall hit10 rate; years with >= 20 days in that quintile
    r_all = rates(a_all)[0][:, 0]
    top_better = bool(r_all[nb - 1] >= r_all[0])
    j = nb - 1 if top_better else 0
    lift_y = rates(a_year)[:, j, 0] - bases(a_year)[:, 0]
    ok_y = days[1:, j] >= MIN_QUINTILE_DAYS
    beats, elig = beat_years(np.nan_to_num(lift_y), ok_y)
    stable = elig >= 3 and beats >= 3
    flags[factor] = {
        "best_end_top": float(top_better),
        "years_beat": float(beats),
        "years_eligible": float(elig),
        "stable": float(stable),
    }
    for m, val in flags[factor].items():
        rows.add("flags", factor, "all", "", m, val)


# --- coin factors ----------------------------------------------------------------------------


@dataclass
class CoinPre:
    """Within-day buckets and excess outcomes of one coin factor."""

    nb: int
    bucket: I64  # -1 when the entry is out (null factor or fewer than 30 valid that day)
    excess: F64  # (n, 4)


def coin_buckets(fr: Frame, f: str, nb_raw: int, over: str | None) -> I64:
    df = fr.df
    if f in FLAGS:
        vals = df[f].to_numpy().astype(np.float64)
        assert set(np.unique(vals[~np.isnan(vals)]).tolist()) <= {0.0, 1.0}
        return np.where(np.isnan(vals), -1, vals).astype(np.int64)
    p, _ = pct_rank(df, f, over)
    return bucket_of(p, nb_raw)


def within_day(fr: Frame, f: str) -> CoinPre:
    df = fr.df
    nb = 2 if f in FLAGS else 10
    vals = df[f].to_numpy().astype(np.float64)
    valid = ~np.isnan(vals)
    n_valid_day = np.bincount(fr.day[valid], minlength=fr.n_days)
    elig = valid & (n_valid_day[fr.day] >= MIN_DAY_ENTRIES)
    b = coin_buckets(fr, f, 10, "day")
    b = np.where(elig, b, -1).astype(np.int64)
    e = np.empty_like(fr.Y)
    for k in range(fr.Y.shape[1]):
        y = fr.Y[:, k]
        ok = elig & ~np.isnan(y)
        s = np.bincount(fr.day[ok], weights=y[ok], minlength=fr.n_days)
        c = np.bincount(fr.day[ok], minlength=fr.n_days)
        with np.errstate(invalid="ignore", divide="ignore"):
            dm = np.where(c > 0, s / c, np.nan)
        e[:, k] = y - dm[fr.day]
    e[~elig] = np.nan
    return CoinPre(nb, b, e)


def rank_ic(fr: Frame, f: str) -> tuple[float, float, int, F64]:
    """Mean daily Spearman(factor, peak30) over days with >= 30 entries; t-stat on monthly means.

    Returns (mean IC, t-stat, months, mean IC per year).
    """
    d = fr.df.select("day", "month", "year", f, "peak30").filter(
        pl.col(f).is_not_null() & pl.col("peak30").is_not_null()
    )
    d = d.with_columns(
        rf=pl.col(f).rank("average").over("day"),
        rp=pl.col("peak30").rank("average").over("day"),
    )
    g = (
        d.group_by("day")
        .agg(
            n=pl.len(),
            ic=pl.corr("rf", "rp"),
            month=pl.col("month").first(),
            year=pl.col("year").first(),
        )
        .filter((pl.col("n") >= MIN_DAY_ENTRIES) & pl.col("ic").is_not_null())
        .filter(pl.col("ic").is_not_nan())
    )
    mm = g.group_by("month").agg(ic=pl.col("ic").mean())["ic"].to_numpy().astype(np.float64)
    mean = float(g["ic"].mean())  # type: ignore[arg-type]
    t = float("nan")
    if len(mm) > 2 and mm.std(ddof=1) > 0:
        t = float(mm.mean() / (mm.std(ddof=1) / math.sqrt(len(mm))))
    per_year = np.full(len(YEARS), np.nan)
    gy = g.group_by("year").agg(ic=pl.col("ic").mean())
    for yv, icv in gy.iter_rows():
        per_year[int(yv) - YEARS[0]] = float(icv)
    return mean, t, len(mm), per_year


def coin_view(
    fr: Frame,
    rows: Rows,
    w: F64,
    f: str,
    flags: dict[str, dict[str, float]],
    pre_out: dict[str, CoinPre],
    inter: Rows,
    terc: I64,
) -> None:
    nb = 2 if f in FLAGS else 10
    zeros = np.zeros(fr.df.height, dtype=np.int64)
    vals = fr.df[f].to_numpy().astype(np.float64)
    valid = ~np.isnan(vals)
    # raw
    b_all = coin_buckets(fr, f, 10, None)
    b_year = coin_buckets(fr, f, 10, "year")
    if f not in FLAGS and len(np.unique(vals[valid])) > 1000:
        cnt = np.bincount(b_all[b_all >= 0], minlength=nb)
        assert cnt.min() >= 0.8 * valid.sum() / nb, cnt
        assert cnt.max() <= 1.2 * valid.sum() / nb, cnt
    a_all = agg(b_all, nb, fr.Y, zeros, 1)
    a_year = agg(b_year, nb, fr.Y, fr.year_i, len(YEARS))
    ci = boot_lifts(b_all, nb, fr.Y, fr, w)
    emit_buckets(rows, "coin_decile", f, nb, a_all, a_year, False)
    emit_base(rows, "coin_decile", f, a_all, a_year)
    emit_lifts(rows, "coin_summary", f, nb, a_all, a_year, ci)
    rows.add("coin_summary", f, "all", "", "spearman_peak30", spearman(vals, fr.peak30))
    r_all = rates(a_all)[0][:, 0]
    raw_top = bool(r_all[nb - 1] >= r_all[0])
    j = nb - 1 if raw_top else 0
    lift_y = rates(a_year)[:, j, 0] - bases(a_year)[:, 0]
    raw_beats, _ = beat_years(np.nan_to_num(lift_y), np.ones(len(YEARS), dtype=bool))
    # within-day rank
    pre = within_day(fr, f)
    pre_out[f] = pre
    rb_all = agg(pre.bucket, nb, pre.excess, zeros, 1)
    rb_year = agg(pre.bucket, nb, pre.excess, fr.year_i, len(YEARS))
    rci = boot_lifts(pre.bucket, nb, pre.excess, fr, w)
    emit_buckets(rows, "coin_rank_decile", f, nb, rb_all, rb_year, True)
    emit_lifts(rows, "coin_rank_summary", f, nb, rb_all, rb_year, rci)
    ic, t, n_m, ic_y = rank_ic(fr, f)
    rows.add("coin_rank_summary", f, "all", "", "rank_ic_mean", ic)
    rows.add("coin_rank_summary", f, "all", "", "rank_ic_t", t, None, None)
    rows.add("coin_rank_summary", f, "all", "", "rank_ic_months", float(n_m))
    for yi, y in enumerate(YEARS):
        rows.add("coin_rank_summary", f, str(y), "", "rank_ic_mean", float(ic_y[yi]))
    ex_all = rates(rb_all)[0][:, 0]
    rank_top = bool(ex_all[nb - 1] >= ex_all[0])
    jr = nb - 1 if rank_top else 0
    ex_y = rates(rb_year)[:, jr, 0]
    rank_beats, _ = beat_years(np.nan_to_num(ex_y), np.ones(len(YEARS), dtype=bool))
    raw_stable = raw_beats >= 3
    rank_stable = rank_beats >= 3
    flags[f] = {
        "raw_best_end_top": float(raw_top),
        "raw_years_beat": float(raw_beats),
        "raw_stable": float(raw_stable),
        "rank_best_end_top": float(rank_top),
        "rank_years_beat": float(rank_beats),
        "rank_stable": float(rank_stable),
        "same_end": float(raw_top == rank_top),
        "stable": float(raw_stable and rank_stable),
    }
    for m, val in flags[f].items():
        rows.add("flags", f, "all", "", m, val)
    # interaction: excess in the three terciles of days by m_moon30
    for t_i in range(3):
        sel = np.where(terc == t_i, pre.bucket, -1).astype(np.int64)
        a = agg(sel, nb, pre.excess[:, :1], zeros, 1)
        r = rates(a)[0][:, 0]
        for end, jj in (("top", nb - 1), ("bottom", 0)):
            inter.add(
                "interaction", f, f"moon_t{t_i + 1}", end, "lift_hit10", float(r[jj]), a.n[0, jj]
            )


# --- shuffled-outcome null -------------------------------------------------------------------


def shuffle_null(fr: Frame, pre: dict[str, CoinPre]) -> tuple[pl.DataFrame, float, float, float]:
    """Run the within-day pipeline with hit10 permuted within each day, N_SHUFFLE times.

    Returns the per-run table and the pooled 95th percentiles of the top-decile excess and of the
    best-end excess, plus the mean number of coin factors that pass the within-day stability.
    """
    rng = np.random.default_rng(SEED + 1)
    h = fr.Y[:, 0]
    out: dict[str, list[object]] = {
        "shuffle": [],
        "factor": [],
        "top_excess": [],
        "bottom_excess": [],
        "best_excess": [],
        "rank_stable": [],
    }
    for s in range(N_SHUFFLE):
        order = np.lexsort((rng.random(len(h)), fr.day))
        hs = h[order]  # entries are sorted by day, so this permutes within each day
        for f, p in pre.items():
            ok = p.bucket >= 0
            ssum = np.bincount(fr.day[ok], weights=hs[ok], minlength=fr.n_days)
            scnt = np.bincount(fr.day[ok], minlength=fr.n_days)
            with np.errstate(invalid="ignore", divide="ignore"):
                dm = np.where(scnt > 0, ssum / scnt, 0.0)
            e = hs - dm[fr.day]
            a = agg(p.bucket, p.nb, e[:, None], np.zeros(len(h), dtype=np.int64), 1)
            ay = agg(p.bucket, p.nb, e[:, None], fr.year_i, len(YEARS))
            r = rates(a)[0][:, 0]
            top, bot = float(r[p.nb - 1]), float(r[0])
            jj = p.nb - 1 if top >= bot else 0
            beats = int((np.nan_to_num(rates(ay)[:, jj, 0]) > 0).sum())
            out["shuffle"].append(s)
            out["factor"].append(f)
            out["top_excess"].append(top)
            out["bottom_excess"].append(bot)
            out["best_excess"].append(max(top, bot))
            out["rank_stable"].append(beats >= 3)
    tab = pl.DataFrame(out)
    p95_top = float(np.percentile(tab["top_excess"].to_numpy(), 95))
    p95_best = float(np.percentile(tab["best_excess"].to_numpy(), 95))
    mean_stable = float(tab.group_by("shuffle").agg(k=pl.col("rank_stable").sum())["k"].mean())  # type: ignore[arg-type]
    return tab, p95_top, p95_best, mean_stable


# --- redundancy ------------------------------------------------------------------------------


def redundancy(fr: Frame) -> tuple[pl.DataFrame, list[list[str]], list[tuple[str, str, float]]]:
    names = fr.factors
    ranks = np.column_stack(
        [rank_avg(fr.df[f].to_numpy().astype(np.float64)) for f in names]
    ).astype(np.float64)
    k = len(names)
    m = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            ok = ~(np.isnan(ranks[:, i]) | np.isnan(ranks[:, j]))
            if ok.sum() < 100:
                m[i, j] = m[j, i] = np.nan
                continue
            x = ranks[ok, i]
            y = ranks[ok, j]
            if x.std() == 0 or y.std() == 0:
                m[i, j] = m[j, i] = np.nan
                continue
            m[i, j] = m[j, i] = float(np.corrcoef(x, y)[0, 1])
    tab = pl.DataFrame({"factor": names, **{n: m[:, i] for i, n in enumerate(names)}})
    parent = list(range(k))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs: list[tuple[str, str, float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            if not np.isnan(m[i, j]) and abs(m[i, j]) > CLUSTER_RHO:
                parent[find(i)] = find(j)
                pairs.append((names[i], names[j], float(m[i, j])))
    groups: dict[int, list[str]] = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(names[i])
    clusters = sorted((g for g in groups.values() if len(g) > 1), key=lambda g: -len(g))
    return tab, clusters, pairs


# --- base rates ------------------------------------------------------------------------------


def base_view(fr: Frame, rows: Rows) -> None:
    zeros = np.zeros(fr.df.height, dtype=np.int64)
    b = np.zeros(fr.df.height, dtype=np.int64)
    a_all = agg(b, 1, fr.Y, zeros, 1)
    a_year = agg(b, 1, fr.Y, fr.year_i, len(YEARS))
    emit_base(rows, "base", "", a_all, a_year)
    for gi, scope in enumerate(SCOPES):
        n = a_all.n[0, 0] if gi == 0 else a_year.n[gi - 1, 0]
        rows.add("base", "", scope, "base", "entries", float(n), n)
        rows.add(
            "base",
            "",
            scope,
            "base",
            "days",
            float(fr.n_days if gi == 0 else len(np.unique(fr.day[fr.year_i == gi - 1]))),
        )


# --- markdown --------------------------------------------------------------------------------


class Lookup:
    def __init__(self, df: pl.DataFrame) -> None:
        self.d: dict[tuple[str, str, str, str, str], float] = {}
        self.n: dict[tuple[str, str, str, str, str], int | None] = {}
        for r in df.iter_rows(named=True):
            key = (r["view"], r["factor"], r["scope"], r["bucket"], r["metric"])
            self.d[key] = r["value"]
            self.n[key] = r["n_entries"]

    def g(self, view: str, factor: str, scope: str, bucket: str, metric: str) -> float:
        return self.d.get((view, factor, scope, bucket, metric), float("nan"))


def pp(x: float, signed: bool = True) -> str:
    if math.isnan(x):
        return "n/a"
    return f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}"


def yes(x: float) -> str:
    return "yes" if x >= 0.5 else "no"


def table(header: list[str], body: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return out


def write_md(
    fr: Frame,
    out: pl.DataFrame,
    clusters: list[list[str]],
    shuffle: tuple[float, float, float],
    inter_factors: list[str],
) -> None:
    lk = Lookup(out)
    p95_top, p95_best, mean_stable_shuffle = shuffle
    L: list[str] = []
    L += [
        "# M3: single-factor results on discovery (2020-01-31 .. 2023-12-01)",
        "",
        "Generated by `m3_single.py` from `results/m3_single.csv`; every number below is read from",
        "that csv. Entries are discovery coin-days after the redenomination exclusions; 2024-26 is",
        "never read. Rates and lifts are in percent / percentage points (pp). *Lift* is a bucket's",
        "rate minus the base rate of the same sample (entries with a non-null factor). The 95%",
        "intervals are a bootstrap over calendar months (500 draws). Windows overlap for 30 days",
        "and coins move together, so even these intervals are optimistic about independence.",
        "",
        "## Base rates",
        "",
    ]
    body: list[list[str]] = []
    for sc in SCOPES:
        body.append(
            [
                "all" if sc == "all" else sc,
                f"{int(lk.g('base', '', sc, 'base', 'entries'))}",
                f"{int(lk.g('base', '', sc, 'base', 'days'))}",
                pp(lk.g("base", "", sc, "base", "hit10_rate"), False),
                pp(lk.g("base", "", sc, "base", "hit20_rate"), False),
                pp(lk.g("base", "", sc, "base", "hit_ns_rate"), False),
                pp(lk.g("base", "", sc, "base", "ret10_mean"), False),
            ]
        )
    # ret_20 / ret_ns means are stored in the base csv rows below
    L += table(
        ["scope", "entries", "days", "hit10 %", "hit20 %", "hit_ns %", "mean ret_10 %"], body
    )
    L += [""]
    body = []
    for sc in SCOPES:
        body.append(
            [
                sc,
                pp(lk.g("base_ret", "", sc, "base", "ret_10_mean"), False),
                pp(lk.g("base_ret", "", sc, "base", "ret_20_mean"), False),
                pp(lk.g("base_ret", "", sc, "base", "ret_ns_mean"), False),
            ]
        )
    L += ["Mean net trade return per entry by stop variant (fees and slippage in):", ""]
    L += table(["scope", "ret_10 %", "ret_20 %", "ret_ns %"], body)
    L += [""]

    # market ranked
    mk = []
    for f in fr.market:
        sp = lk.g("mkt_summary", f, "all", "spread", "spread_hit10")
        if not math.isnan(sp):
            mk.append((abs(sp), f))
    mk.sort(reverse=True)
    n_mkt_stable = sum(lk.g("flags", f, "all", "", "stable") >= 0.5 for f in fr.market)
    L += [
        '## Market factors (the "when" question): top 15 by quintile spread in hit10',
        "",
        "Quintiles over discovery days (each day once). Spread = top-quintile hit10 rate minus",
        "bottom-quintile rate. *Best end* is the quintile with the higher overall hit10 rate; its",
        "lift, 95% interval and per-year lifts are shown. *Stable*: best quintile beats the year's",
        "base rate in at least 3 of the 4 years in which that quintile has at least 20 days",
        "(*yrs* = years beating / years eligible).",
        "",
    ]
    body = []
    body2: list[list[str]] = []
    for _, f in mk[:15]:
        top = lk.g("flags", f, "all", "", "best_end_top") >= 0.5
        end = "top" if top else "bottom"
        yl = [pp(lk.g("mkt_summary", f, str(y), end, "lift_hit10")) for y in YEARS]
        body.append(
            [
                f"`{f}`",
                pp(lk.g("mkt_summary", f, "all", "spread", "spread_hit10")),
                f"[{pp(lk.g('mkt_summary', f, 'all', 'spread', 'ci_lo_spread_hit10'))}, "
                f"{pp(lk.g('mkt_summary', f, 'all', 'spread', 'ci_hi_spread_hit10'))}]",
                end,
                pp(lk.g("mkt_summary", f, "all", end, "lift_hit10")),
                f"[{pp(lk.g('mkt_summary', f, 'all', end, 'ci_lo_lift_hit10'))}, "
                f"{pp(lk.g('mkt_summary', f, 'all', end, 'ci_hi_lift_hit10'))}]",
                *yl,
                f"{int(lk.g('flags', f, 'all', '', 'years_beat'))}/"
                f"{int(lk.g('flags', f, 'all', '', 'years_eligible'))}",
                yes(lk.g("flags", f, "all", "", "stable")),
            ]
        )
        body2.append(
            [
                f"`{f}`",
                end,
                pp(lk.g("mkt_summary", f, "all", end, "lift_hit10")),
                pp(lk.g("mkt_summary", f, "all", end, "lift_hit20")),
                pp(lk.g("mkt_summary", f, "all", end, "lift_hit_ns")),
                pp(lk.g("mkt_summary", f, "all", end, "lift_ret10")),
                f"{lk.g('mkt_summary', f, 'all', '', 'spearman_day_hit10'):+.2f}",
            ]
        )
    L += table(
        [
            "factor",
            "spread pp",
            "spread 95%",
            "best end",
            "lift hit10 pp",
            "95% interval",
            "2020",
            "2021",
            "2022",
            "2023",
            "yrs",
            "stable",
        ],
        body,
    )
    L += ["", "Same factors and ends, the other labels (lift in pp over the base rate):", ""]
    L += table(
        ["factor", "end", "hit10", "hit20", "hit_ns", "mean ret_10", "Spearman, day hit10"], body2
    )
    L += [""]
    L += [
        f"Market factors flagged stable: {n_mkt_stable} of {len(fr.market)}.",
        "",
    ]

    # coin ranked
    cs = []
    for f in fr.coin:
        t = lk.g("coin_rank_summary", f, "all", "top", "lift_hit10")
        b = lk.g("coin_rank_summary", f, "all", "bottom", "lift_hit10")
        if not (math.isnan(t) or math.isnan(b)):
            cs.append((max(t, b), f))
    cs.sort(reverse=True)
    n_rank_stable = sum(lk.g("flags", f, "all", "", "rank_stable") >= 0.5 for f in fr.coin)
    n_raw_stable = sum(lk.g("flags", f, "all", "", "raw_stable") >= 0.5 for f in fr.coin)
    L += [
        '## Coin factors (the "which coin" question): top 20 by within-day excess hit10',
        "",
        "Within-day rank: each entry's percentile among the day's entries (days with at least 30),",
        "cut into deciles. *Excess* = bucket mean of (outcome minus that day's mean outcome), so",
        "the market is removed. Best end = the decile (top or bottom) with the larger excess.",
        "*Rank IC* = mean daily Spearman(factor, peak30), t-stat from monthly means. *Raw* columns",
        "are the same end's undifferentiated decile lift (deciles over all entries; per year with",
        "deciles recomputed in the year). *Stable* needs the raw best decile to beat the base in",
        "at least 3 of 4 years and the within-day best decile to have positive excess in at least",
        "3 of 4 years (*same end* tells whether both views pick the same end).",
        "",
    ]
    body = []
    body2 = []
    for _, f in cs[:20]:
        top = lk.g("flags", f, "all", "", "rank_best_end_top") >= 0.5
        end = "top" if top else "bottom"
        rtop = lk.g("flags", f, "all", "", "raw_best_end_top") >= 0.5
        rend = "top" if rtop else "bottom"
        ye = [pp(lk.g("coin_rank_summary", f, str(y), end, "lift_hit10")) for y in YEARS]
        yr = [pp(lk.g("coin_summary", f, str(y), end, "lift_hit10")) for y in YEARS]
        body.append(
            [
                f"`{f}`",
                end,
                pp(lk.g("coin_rank_summary", f, "all", end, "lift_hit10")),
                f"[{pp(lk.g('coin_rank_summary', f, 'all', end, 'ci_lo_lift_hit10'))}, "
                f"{pp(lk.g('coin_rank_summary', f, 'all', end, 'ci_hi_lift_hit10'))}]",
                *ye,
                f"{lk.g('coin_rank_summary', f, 'all', '', 'rank_ic_mean'):+.3f} "
                f"({lk.g('coin_rank_summary', f, 'all', '', 'rank_ic_t'):+.1f})",
                pp(lk.g("coin_summary", f, "all", end, "lift_hit10")),
                *yr,
                f"{int(lk.g('flags', f, 'all', '', 'raw_years_beat'))}"
                f"/{int(lk.g('flags', f, 'all', '', 'rank_years_beat'))}",
                yes(lk.g("flags", f, "all", "", "same_end")),
                yes(lk.g("flags", f, "all", "", "stable")),
            ]
        )
        body2.append(
            [
                f"`{f}`",
                end,
                pp(lk.g("coin_rank_summary", f, "all", end, "lift_hit10")),
                pp(lk.g("coin_rank_summary", f, "all", end, "lift_hit20")),
                pp(lk.g("coin_rank_summary", f, "all", end, "lift_hit_ns")),
                pp(lk.g("coin_rank_summary", f, "all", end, "lift_ret10")),
                f"[{pp(lk.g('coin_rank_summary', f, 'all', end, 'ci_lo_lift_hit_ns'))}, "
                f"{pp(lk.g('coin_rank_summary', f, 'all', end, 'ci_hi_lift_hit_ns'))}]",
                f"[{pp(lk.g('coin_rank_summary', f, 'all', end, 'ci_lo_lift_ret10'))}, "
                f"{pp(lk.g('coin_rank_summary', f, 'all', end, 'ci_hi_lift_ret10'))}]",
                f"{lk.g('coin_summary', f, 'all', '', 'spearman_peak30'):+.3f}",
                rend,
            ]
        )
    L += table(
        [
            "factor",
            "end",
            "excess hit10 pp",
            "95% interval",
            "2020",
            "2021",
            "2022",
            "2023",
            "rank IC (t)",
            "raw lift hit10 pp",
            "raw 2020",
            "raw 2021",
            "raw 2022",
            "raw 2023",
            "yrs raw/rank",
            "same end",
            "stable",
        ],
        body,
    )
    L += [
        "",
        "Same factors and ends, the other labels (within-day excess, pp; ret_10 excess in pp of "
        "trade return):",
        "",
    ]
    L += table(
        [
            "factor",
            "end",
            "hit10",
            "hit20",
            "hit_ns",
            "ret_10",
            "hit_ns 95%",
            "ret_10 95%",
            "Spearman vs peak30",
            "raw best end",
        ],
        body2,
    )
    L += [""]

    # stable list
    stable_m = [f for f in fr.market if lk.g("flags", f, "all", "", "stable") >= 0.5]
    stable_c = [f for f in fr.coin if lk.g("flags", f, "all", "", "stable") >= 0.5]
    L += ["## Stable factors", ""]
    L += [
        f"- Market ({len(stable_m)} of {len(fr.market)}): "
        + (", ".join(f"`{f}`" for f in stable_m) or "none"),
        f"- Coin ({len(stable_c)} of {len(fr.coin)}; raw alone passes {n_raw_stable}, "
        f"within-day alone passes {n_rank_stable}): "
        + (", ".join(f"`{f}`" for f in stable_c) or "none"),
        "",
    ]
    n_over = 0
    for _, f in cs[:20]:
        e_top = lk.g("flags", f, "all", "", "rank_best_end_top") >= 0.5
        e = "top" if e_top else "bottom"
        n_over += lk.g("coin_rank_summary", f, "all", e, "lift_hit10") > p95_best
    L += ["## How much of this is luck", ""]
    nfac = len(fr.coin)
    L += [
        f"hit10 was shuffled within each day {N_SHUFFLE} times and the within-day pipeline rerun "
        f"on every coin factor ({N_SHUFFLE * nfac} factor-runs; `results/m3_shuffle.csv`). Under",
        "the shuffle:",
        "",
        f"- 95th percentile of the top-decile hit10 excess: {pp(p95_top)} pp; of the best-end "
        f"excess (the ranking metric): {pp(p95_best)} pp.",
        f"- mean number of coin factors (of {nfac}) passing the within-day half of the stability "
        f"test: {mean_stable_shuffle:.1f}; in the real data {n_rank_stable} pass it "
        f"({n_raw_stable} pass the raw half, {len(stable_c)} pass both).",
        f"- of the top-20 coin factors above, {n_over} exceed the shuffled best-end 95th "
        "percentile.",
        "",
        "The shuffle destroys the coin-level persistence (the same coin on neighbouring days has "
        "overlapping windows and similar outcomes), so it is a floor for how large an excess can "
        "look by chance, not a ceiling: real noise is wider. The market factors are not "
        "shuffled (only 1,400 days, strongly autocorrelated); for them read the bootstrap "
        "intervals and the stable flag, remembering that 20 factors were looked at. The stable "
        "flag itself is weak: positive in 3 of 4 years is passed by many null factors "
        "(see the count above).",
        "",
    ]

    # interaction
    L += [
        "## Interaction: does the coin factor work in every market state?",
        "",
        "The 10 coin factors with the largest within-day top-decile hit10 excess; that excess "
        "(pp) in the three terciles of days by `m_moon30` (the share of coins that just made a "
        "big move). T1 = quiet, T3 = hot.",
        "",
    ]
    body = []
    for f in inter_factors:
        vals = [lk.g("interaction", f, f"moon_t{i}", "top", "lift_hit10") for i in (1, 2, 3)]
        bvals = [lk.g("interaction", f, f"moon_t{i}", "bottom", "lift_hit10") for i in (1, 2, 3)]
        body.append(
            [
                f"`{f}`",
                *[pp(v) for v in vals],
                *[pp(v) for v in bvals],
                "yes" if all(v > 0 for v in vals) else "no",
            ]
        )
    L += table(
        [
            "factor",
            "top T1",
            "top T2",
            "top T3",
            "bottom T1",
            "bottom T2",
            "bottom T3",
            "top > 0 in all",
        ],
        body,
    )
    L += [""]
    lo = [lk.g("interaction", "m_moon30", f"moon_t{i}", "", "day_min") for i in (1, 2, 3)]
    hi = [lk.g("interaction", "m_moon30", f"moon_t{i}", "", "day_max") for i in (1, 2, 3)]
    L += [
        "`m_moon30` ranges of the terciles: "
        + "; ".join(f"T{i + 1} {lo[i]:.3f}..{hi[i]:.3f}" for i in range(3))
        + ".",
        "",
    ]

    # redundancy
    L += [
        "## Redundancy",
        "",
        "Spearman matrix: `results/m3_redundancy.csv` (all entries, pairwise non-null; market "
        f"factors enter once per entry). Connected groups of factors with |rho| > {CLUSTER_RHO} "
        "(linked groups can chain through intermediate factors):",
        "",
    ]
    for g in clusters:
        L.append(f"- ({len(g)}) " + ", ".join(f"`{x}`" for x in g))
    if not clusters:
        L.append("- none")
    L += [""]

    # observations
    L += ["## Observations", ""]
    obs: list[str] = []
    base_h = [lk.g("base", "", str(y), "base", "hit10_rate") for y in YEARS]
    obs.append(
        "Base hit10 by year: "
        + ", ".join(f"{y} {pp(v, False)}%" for y, v in zip(YEARS, base_h, strict=True))
        + "; the year-to-year swing in the base rate is itself "
        f"{pp(max(base_h) - min(base_h), False)} pp, larger than most factor effects below."
    )
    if mk:
        f0 = mk[0][1]
        obs.append(
            f"Largest market spread: `{f0}` at "
            f"{pp(lk.g('mkt_summary', f0, 'all', 'spread', 'spread_hit10'))} pp "
            f"(95% {pp(lk.g('mkt_summary', f0, 'all', 'spread', 'ci_lo_spread_hit10'))} to "
            f"{pp(lk.g('mkt_summary', f0, 'all', 'spread', 'ci_hi_spread_hit10'))}); "
            f"{n_mkt_stable} of {len(fr.market)} market factors are stable."
        )
    n_mkt_ci = sum(
        (lk.g("mkt_summary", f, "all", "spread", "ci_lo_spread_hit10") > 0)
        or (lk.g("mkt_summary", f, "all", "spread", "ci_hi_spread_hit10") < 0)
        for f in fr.market
    )
    obs.append(
        f"{n_mkt_ci} of {len(fr.market)} market factors have a spread interval that excludes zero."
    )
    if cs:
        f0 = cs[0][1]
        e0 = "top" if lk.g("flags", f0, "all", "", "rank_best_end_top") >= 0.5 else "bottom"
        obs.append(
            f"Largest within-day coin excess: `{f0}` ({e0} decile) at "
            f"{pp(lk.g('coin_rank_summary', f0, 'all', e0, 'lift_hit10'))} pp hit10, "
            f"{pp(lk.g('coin_rank_summary', f0, 'all', e0, 'lift_hit_ns'))} pp hit_ns, "
            f"{pp(lk.g('coin_rank_summary', f0, 'all', e0, 'lift_ret10'))} pp mean ret_10."
        )
    both = 0
    only_stops = 0
    for _, f in cs[:20]:
        e = "top" if lk.g("flags", f, "all", "", "rank_best_end_top") >= 0.5 else "bottom"
        h10 = lk.g("coin_rank_summary", f, "all", e, "lift_hit10")
        hns = lk.g("coin_rank_summary", f, "all", e, "lift_hit_ns")
        if h10 > 0 and hns > 0:
            both += 1
        elif h10 > 0 and hns <= 0:
            only_stops += 1
    obs.append(
        f"Of the top-20 coin factors, {both} also raise hit_ns (they find rallies) and "
        f"{only_stops} raise hit10 only (they avoid stop-outs without finding more rallies)."
    )
    n_ci = sum(
        lk.g(
            "coin_rank_summary",
            f,
            "all",
            "top" if lk.g("flags", f, "all", "", "rank_best_end_top") >= 0.5 else "bottom",
            "ci_lo_lift_hit10",
        )
        > 0
        for _, f in cs[:20]
    )
    obs.append(f"{n_ci} of the top-20 have a bootstrap interval for the excess that excludes zero.")
    obs.append(
        f"{len(stable_c)} coin factors pass both stability halves; the shuffled null passes the "
        f"within-day half for {mean_stable_shuffle:.1f} of {nfac} on average, against "
        f"{n_rank_stable} in the real data."
    )
    pos_all = sum(
        all(lk.g("interaction", f, f"moon_t{i}", "top", "lift_hit10") > 0 for i in (1, 2, 3))
        for f in inter_factors
    )
    obs.append(
        f"Interaction: {pos_all} of the {len(inter_factors)} factors have a positive top-decile "
        "excess in all three market-state terciles."
    )
    if clusters:
        obs.append(
            f"Redundancy: {len(clusters)} linked groups at |rho| > {CLUSTER_RHO}, the largest "
            f"with {len(clusters[0])} factors, so the 65 factors are far fewer independent ideas."
        )
    for o in obs:
        L.append(f"- {o}")
    L += [""]
    (RESULTS / "m3_single.md").write_text("\n".join(L))


# --- main ------------------------------------------------------------------------------------


def base_ret_rows(fr: Frame, rows: Rows) -> None:
    for gi, scope in enumerate(SCOPES):
        sel = np.ones(fr.df.height, dtype=bool) if gi == 0 else fr.year_i == gi - 1
        for c in ("ret_10", "ret_20", "ret_ns"):
            v = fr.df[c].to_numpy().astype(np.float64)[sel]
            rows.add(
                "base_ret", "", scope, "base", f"{c}_mean", float(np.nanmean(v)), int(sel.sum())
            )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fr = load()
    w = boot_weights(fr.n_months)
    rows = Rows()
    inter = Rows()
    base_view(fr, rows)
    base_ret_rows(fr, rows)
    flags: dict[str, dict[str, float]] = {}
    for f in fr.market:
        market_view(fr, rows, w, f, flags)
    # terciles of days by m_moon30 for the interaction view
    dd = fr.df.group_by("day").agg(v=pl.col("m_moon30").drop_nulls().first()).sort("day")
    tp = (
        dd.select(r=pl.col("v").rank("average").cast(pl.Float64), n=pl.col("v").count())
        .with_columns(p=(pl.col("r") - 0.5) / pl.col("n"))["p"]
        .to_numpy()
        .astype(np.float64)
    )
    terc_d = bucket_of(tp, 3)
    terc = terc_d[fr.day]
    vday = dd["v"].to_numpy().astype(np.float64)
    for t_i in range(3):
        sel = terc_d == t_i
        inter.add(
            "interaction", "m_moon30", f"moon_t{t_i + 1}", "", "day_min", float(vday[sel].min())
        )
        inter.add(
            "interaction", "m_moon30", f"moon_t{t_i + 1}", "", "day_max", float(vday[sel].max())
        )
        inter.add("interaction", "m_moon30", f"moon_t{t_i + 1}", "", "days", float(sel.sum()))
    pre: dict[str, CoinPre] = {}
    for f in fr.coin:
        coin_view(fr, rows, w, f, flags, pre, inter, terc)
        print("coin factor done", f, flush=True)
    # which 10 coin factors for the interaction view: largest within-day top-decile excess
    base_df = rows.frame()
    top_ex = base_df.filter(
        (pl.col("view") == "coin_rank_summary")
        & (pl.col("scope") == "all")
        & (pl.col("bucket") == "top")
        & (pl.col("metric") == "lift_hit10")
    ).sort("value", descending=True)
    inter_factors = top_ex["factor"].to_list()[:10]
    inter_df = inter.frame().filter(
        (pl.col("factor") == "m_moon30") | pl.col("factor").is_in(inter_factors)
    )
    tab, p95_top, p95_best, mean_stable = shuffle_null(fr, pre)
    tab.write_csv(RESULTS / "m3_shuffle.csv", float_precision=6)
    rows_shuf = Rows()
    rows_shuf.add("shuffle", "", "all", "top", "p95_top_excess", p95_top)
    rows_shuf.add("shuffle", "", "all", "best", "p95_best_excess", p95_best)
    rows_shuf.add("shuffle", "", "all", "", "mean_rank_stable", mean_stable)
    red, clusters, _pairs = redundancy(fr)
    red.write_csv(RESULTS / "m3_redundancy.csv", float_precision=4)
    rows_red = Rows()
    for i, g in enumerate(clusters):
        for f in g:
            rows_red.add("cluster", f, "all", str(i + 1), "cluster_size", float(len(g)))
    out = pl.concat([base_df, inter_df, rows_shuf.frame(), rows_red.frame()])
    out.write_csv(RESULTS / "m3_single.csv", float_precision=6)
    write_md(fr, out, clusters, (p95_top, p95_best, mean_stable), inter_factors)
    print("wrote results")


if __name__ == "__main__":
    main()
