"""w12: why crowded-short entry (low funding / low ls_acc) helps candidate trades but not the 10-slot book."""  # noqa: E501

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from common_ls import OWN, meta, scores
from diag import annotate, scorecard, table
from sim import Rule, fmt_years, simulate, simulate_yearly

KIND = {0: "time", 1: "stop", 2: "tp", 3: "liq", 4: "stop"}
m = meta()
pct = scores("day5")
yr = m["year"].to_numpy()
idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.995))
F = (
    pl.read_parquet(OWN / "feats.parquet")
    .select("row", "rate")
    .join(
        pl.read_parquet(OWN / "metric_feats.parquet").select("row", "ls_acc", "oi_ch4h"),
        on="row",
        how="full",
        coalesce=True,
    )
)
C = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]}).join(F, on="row", how="left")
C = C.with_columns(pl.col("rate", "ls_acc", "oi_ch4h").fill_nan(None))
print("cands", C.height, "null rate", C["rate"].null_count(), "null ls", C["ls_acc"].null_count())
R10 = Rule(hold_min=4320, tp=0.2, slots=10, cost=0.004)
RU = Rule(hold_min=4320, tp=0.2, slots=100_000, cost=0.004, liq_frac=None)
U = simulate(C, RU, start_equity=1e12).trades.with_columns(
    exit_kind=pl.col("kind").replace_strict(KIND, default="time"),
    year=pl.col("open_time").dt.year(),
)
U = (
    U.join(C.select("row", "rate", "ls_acc", "oi_ch4h"), on="row", how="left", suffix="_c")
    if "rate" not in U.columns
    else U
)
print("uncon", U.height, f"{U['net'].mean() * 1e4:+.0f}bp")
U = U.with_columns(day=pl.col("open_time").dt.date(), o=pl.col("net"))
out = ["# Why crowded-short entry does not lift the 10-slot capitulation-long book", ""]


def book(c, name, key=None):
    b = simulate_yearly(c, R10)
    return b


def line(name, b):
    return f"| {name} | {b.summary['trades']} | {b.summary['bp']:+.0f} | {fmt_years(b)} |"


# A. variance decomposition
rows = [
    "| measure | n | days | share of variance between days | within-day std / total std | mean #cands/day | within-day Spearman IC vs net (day-demeaned ranks) | IC raw |",  # noqa: E501
    "|---|---|---|---|---|---|---|---|",
]


def dm_ic(d, f):
    d = d.select("day", f, "net").drop_nulls().drop_nans()
    d = d.with_columns(
        r=pl.col(f).rank().over("day"), q=pl.col("net").rank().over("day"), n=pl.len().over("day")
    ).filter(pl.col("n") >= 5)
    d = d.with_columns(
        r=pl.col("r") - pl.col("r").mean().over("day"),
        q=pl.col("q") - pl.col("q").mean().over("day"),
    )
    ic = float((d["r"] * d["q"]).sum() / np.sqrt((d["r"] ** 2).sum() * (d["q"] ** 2).sum()))
    # per-day IC mean
    return ic, d["day"].n_unique()


for f in ("rate", "ls_acc", "oi_ch4h"):
    d = U.select("day", f, "net").drop_nulls().drop_nans()
    tot = d[f].var()
    mu = d[f].mean()
    dayv = d.group_by("day").agg(n=pl.len(), mu=pl.col(f).mean())
    between = float((dayv["n"] * (dayv["mu"] - mu) ** 2).sum() / (d.height - 1))
    within = tot - between
    ic, nd = dm_ic(U, f)
    raw = float(d.select(pl.corr(pl.col(f).rank(), pl.col("net").rank())).item())
    rows.append(
        f"| {f} | {d.height} | {dayv.height} | {between / tot:.0%} | {np.sqrt(within / tot):.2f} | {d.height / dayv.height:.0f} | {ic:+.3f} | {raw:+.3f} |"  # noqa: E501
    )
out += ["## A. Day vs coin", "", *rows, ""]
# day-level: outcome by day-mean crowding tercile, and within-day terciles
dm = (
    U.group_by("day")
    .agg(n=pl.len(), r=pl.col("rate").mean(), a=pl.col("ls_acc").mean(), net=pl.col("net").mean())
    .filter(pl.col("n") >= 5)
)
out.append(
    f"Day-level (days with >=5 cands: {dm.height}): Spearman(day mean rate, day mean net) = {float(dm.select(pl.corr(pl.col('r').rank(), pl.col('net').rank())).item()):+.2f}; "  # noqa: E501
    f"Spearman(day mean ls_acc, day mean net) = {float(dm.drop_nulls().select(pl.corr(pl.col('a').rank(), pl.col('net').rank())).item()):+.2f}."  # noqa: E501
)
U = U.with_columns(
    rd=pl.col("rate") - pl.col("rate").mean().over("day"),
    ad=pl.col("ls_acc") - pl.col("ls_acc").mean().over("day"),
)
out += [
    "",
    "Within-day terciles of the day-demeaned rate (low = more crowded shorts than same-day peers):",
    "",
    "| tercile | n | net bp | TP share |",
    "|---|---|---|---|",
]
V = U.filter(pl.col("rd").is_not_null()).with_columns(
    t=((pl.col("rd").rank("ordinal") - 1) * 3 / pl.len()).floor()
)
for t, nm in ((0, "low"), (1, "mid"), (2, "high")):
    s = V.filter(pl.col("t") == t)
    out.append(
        f"| {nm} | {s.height} | {s['net'].mean() * 1e4:+.0f} | {(s['kind'] == 2).mean():.0%} |"
    )
# B. which trades does the book take
bb = simulate_yearly(C, R10)
Tk = bb.trades
tk = set(Tk["row"].to_list())
U = U.with_columns(taken=pl.col("row").is_in(list(tk)))
out += [
    "",
    "## B. Taken vs untaken candidates (base book)",
    "",
    "| group | n | net bp | rate bp/8h mean | rate median | ls_acc mean | oi_ch4h mean | TP share | mean min after episode start |",  # noqa: E501
    "|---|---|---|---|---|---|---|---|---|",
]
tt = U.sort("t")
t_ = tt["t"].to_numpy()
ep = np.concatenate([[0], np.cumsum(np.diff(t_) >= 7200)])
st = t_[np.r_[0, np.flatnonzero(np.diff(t_) >= 7200) + 1]][ep]
U = tt.with_columns(off=pl.Series((t_ - st) / 60), ep=pl.Series(ep))
for nm, s in (("taken", U.filter(pl.col("taken"))), ("untaken", U.filter(~pl.col("taken")))):
    out.append(
        f"| {nm} | {s.height} | {s['net'].mean() * 1e4:+.0f} | {s['rate'].mean() * 1e4:+.2f} | {s['rate'].median() * 1e4:+.2f} | {s['ls_acc'].mean():.2f} | {s['oi_ch4h'].mean():+.3f} | {(s['kind'] == 2).mean():.0%} | {s['off'].mean():.0f} |"  # noqa: E501
    )
# within-day: taken minus same-day untaken, crowding
dd = (
    U.group_by("day")
    .agg(
        r_t=pl.col("rate").filter(pl.col("taken")).mean(),
        r_u=pl.col("rate").filter(~pl.col("taken")).mean(),
        a_t=pl.col("ls_acc").filter(pl.col("taken")).mean(),
        a_u=pl.col("ls_acc").filter(~pl.col("taken")).mean(),
        n_t=pl.col("taken").sum(),
        n_u=(~pl.col("taken")).sum(),
    )
    .filter((pl.col("n_t") > 0) & (pl.col("n_u") > 0))
)
out.append(
    f"\nSame-day pairs ({dd.height} days with both taken and untaken): mean rate taken {dd['r_t'].mean() * 1e4:+.2f} vs untaken {dd['r_u'].mean() * 1e4:+.2f} bp; ls_acc {dd['a_t'].mean():.2f} vs {dd['a_u'].mean():.2f}."  # noqa: E501
)
# candidates per day / flush days
fd = U.group_by("day").agg(n=pl.len(), nt=pl.col("taken").sum())
out.append(
    f"Candidate days {fd.height}; days with >10 candidates {fd.filter(pl.col('n') > 10).height} (hold {fd.filter(pl.col('n') > 10)['n'].sum()} of {fd['n'].sum()} candidates); days with <=10 candidates {fd.filter(pl.col('n') <= 10).height} (all taken unless slots busy: taken {fd.filter(pl.col('n') <= 10)['nt'].sum()} of {fd.filter(pl.col('n') <= 10)['n'].sum()})."  # noqa: E501
)
# crowding vs entry time within episode
e = U.with_columns(early=pl.col("off") < 15)
out += [
    "",
    "Crowding by entry time inside the episode (all candidates):",
    "",
    "| window | n | rate bp/8h | ls_acc | net bp |",
    "|---|---|---|---|---|",
]
for nm, s in (("first 15 min", e.filter(pl.col("early"))), ("later", e.filter(~pl.col("early")))):
    out.append(
        f"| {nm} | {s.height} | {s['rate'].mean() * 1e4:+.2f} | {s['ls_acc'].mean():.2f} | {s['net'].mean() * 1e4:+.0f} |"  # noqa: E501
    )
pd = (
    U.group_by("day")
    .agg(
        n=pl.len(),
        taken=pl.col("taken").sum(),
        rate_bp=pl.col("rate").mean() * 1e4,
        rate_sd_bp=pl.col("rate").std() * 1e4,
        ls=pl.col("ls_acc").mean(),
        net=pl.col("net").mean() * 1e4,
    )
    .sort("day")
)
out += [
    "",
    "Per flush day (all candidates; rate_sd = within-day std):",
    "",
    "| day | cands | taken | mean rate bp | within-day sd bp | ls_acc | net bp |",
    "|---|---|---|---|---|---|---|",
]
for r in pd.iter_rows(named=True):
    out.append(
        f"| {r['day']} | {r['n']} | {r['taken']} | {r['rate_bp'] if r['rate_bp'] is not None else float('nan'):+.2f} | {r['rate_sd_bp'] or 0:.2f} | {r['ls'] or float('nan'):.2f} | {r['net']:+.0f} |"  # noqa: E501
    )
# C. variants
out += [
    "",
    "## C. Book variants (10 slots, fresh $10k per year, simulate_yearly)",
    "",
    "| variant | trades | bp | years |",
    "|---|---|---|---|",
]
V = {}
V["base (prio = day5 pct)"] = C
V["prio low rate"] = C.with_columns(prio=-pl.col("rate").fill_null(1.0))
V["prio low ls_acc"] = C.with_columns(prio=-pl.col("ls_acc").fill_null(9.0))
V["filter rate<1e-4"] = C.filter(pl.col("rate").is_null() | (pl.col("rate") < 1e-4))
V["filter ls_acc<1.5"] = C.filter(pl.col("ls_acc").is_null() | (pl.col("ls_acc") < 1.5))
books = {}
for k, c in V.items():
    b = simulate_yearly(c, R10)
    books[k] = b
    out.append(line(k, b))
# why priority lowers: entry-time and coin shift
out += [
    "",
    "## D. What a priority rule changes",
    "",
    "| book | n | mean off (min) | share first 15 min | mean rate bp | mean ls_acc | uncon. net bp of taken | n distinct days | mean stake $ |",  # noqa: E501
    "|---|---|---|---|---|---|---|---|---|",
]
Un = U.select("row", "off", "ep", unc=pl.col("net"))
for k in ("base (prio = day5 pct)", "prio low rate", "prio low ls_acc", "filter rate<1e-4"):
    t = (
        books[k]
        .trades.select("row", "net", "stake", "open_time")
        .join(Un, on="row")
        .join(C.select("row", "rate", "ls_acc"), on="row", how="left")
    )
    out.append(
        f"| {k} | {t.height} | {t['off'].mean():.1f} | {(t['off'] < 15).mean():.0%} | {t['rate'].mean() * 1e4:+.2f} | {t['ls_acc'].mean():.2f} | {t['unc'].mean() * 1e4:+.0f} | {t['open_time'].dt.date().n_unique()} | {t['stake'].mean():.0f} |"  # noqa: E501
    )
a = set(books["base (prio = day5 pct)"].trades["row"].to_list())
b2 = set(books["prio low rate"].trades["row"].to_list())
out.append(
    f"\nPrio-low-rate vs base: {len(a & b2)} common trades, {len(a - b2)} dropped, {len(b2 - a)} added. Dropped (unconstrained) {U.filter(pl.col('row').is_in(list(a - b2)))['net'].mean() * 1e4:+.0f} bp, added {U.filter(pl.col('row').is_in(list(b2 - a)))['net'].mean() * 1e4:+.0f} bp; "  # noqa: E501
    f"same-coin overlap: {len(set(books['base (prio = day5 pct)'].trades['symbol'].to_list()) & set(books['prio low rate'].trades['symbol'].to_list()))} coins in both."  # noqa: E501
)
dr = U.filter(pl.col("row").is_in(list(a - b2)))
ad = U.filter(pl.col("row").is_in(list(b2 - a)))
out.append(
    f"Dropped: mean off {dr['off'].mean():.1f} min, day5 pct {pct[dr['row'].to_numpy()].mean():.4f}; added: mean off {ad['off'].mean():.1f} min, pct {pct[ad['row'].to_numpy()].mean():.4f}."  # noqa: E501
)
# how much of the book is time-priority: share of ties; do 'prio' sorts matter? count simultaneous entries  # noqa: E501
sim_n = U.group_by("t").agg(n=pl.len())
out.append(
    f"Entry timestamps: {sim_n.height} distinct for {U.height} candidates; timestamps with >1 candidate: {sim_n.filter(pl.col('n') > 1).height} (covering {sim_n.filter(pl.col('n') > 1)['n'].sum()} candidates). Priority only orders candidates sharing an entry bar."  # noqa: E501
)
# E. scorecards
cards = {}
for k in ("base (prio = day5 pct)", "filter rate<1e-4", "prio low rate"):
    t = books[k].trades.with_columns(
        exit_kind=pl.col("kind").replace_strict(KIND, default="time"),
        year=pl.col("open_time").dt.year(),
    )
    sg = dict(zip(C["row"].to_list(), C["prio"].to_list(), strict=True))
    fr = t.select(
        signal=pl.col("row").replace_strict({r: float(pct[r]) for r in t["row"].to_list()}),
        threshold=pl.lit(0.995),
        net="net",
        mfe="mfe",
        mae="adv",
        tp=pl.lit(0.2),
        stop=pl.lit(None, dtype=pl.Float64),
        exit_kind="exit_kind",
        year="year",
    )
    cards["book: " + k] = scorecard(annotate(fr))
fr = U.select(
    signal=pl.col("row").map_elements(lambda r: float(pct[r]), return_dtype=pl.Float64),
    threshold=pl.lit(0.995),
    net="net",
    mfe="mfe",
    mae="adv",
    tp=pl.lit(0.2),
    stop=pl.lit(None, dtype=pl.Float64),
    exit_kind="exit_kind",
    year="year",
)
cards["all candidates (unconstrained)"] = scorecard(annotate(fr))
fl = U.filter(pl.col("rate").is_null() | (pl.col("rate") < 1e-4))
fr = fl.select(
    signal=pl.col("row").map_elements(lambda r: float(pct[r]), return_dtype=pl.Float64),
    threshold=pl.lit(0.995),
    net="net",
    mfe="mfe",
    mae="adv",
    tp=pl.lit(0.2),
    stop=pl.lit(None, dtype=pl.Float64),
    exit_kind="exit_kind",
    year="year",
)
cards["candidates rate<1e-4 (unconstrained)"] = scorecard(annotate(fr))
out += ["", "## E. diag scorecard rows", "", table(cards)]
Path("results/why/_crowding_raw.md").write_text("\n".join(out))
print("\n".join(out))
