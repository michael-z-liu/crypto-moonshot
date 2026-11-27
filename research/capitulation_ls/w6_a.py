"""w6-A: episodes of q>=0.995 signals, wait-W rule replication, per-episode early vs late outcomes."""  # noqa: E501

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import Rule, simulate_yearly

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(220)
m = meta()
pct = scores("day5")
yr = m["year"].to_numpy()
t_all = m["open_time"].dt.epoch("s").to_numpy()
idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.995))
idx = idx[np.argsort(t_all[idx], kind="stable")]
t = t_all[idx]
ep = np.concatenate([[0], np.cumsum(np.diff(t) >= 7200)])
tmin_in_ep = t - pd_first if (pd_first := None) else 0
first = {}
for e, tt in zip(ep, t, strict=False):
    first.setdefault(e, tt)
off = np.array([(tt - first[e]) / 60 for e, tt in zip(ep, t, strict=False)])
sig = pl.DataFrame(
    {"row": idx, "ep": ep, "off": off, "t": t, "year": yr[idx], "sym": m["symbol"].to_numpy()[idx]}
)
sig.write_parquet("/tmp/w6_sig.parquet")
print("signals", sig.height, "episodes", sig["ep"].n_unique())
e = sig.group_by("ep").agg(
    n=pl.len(),
    dur=pl.col("off").max(),
    year=pl.col("year").first(),
    nsym=pl.col("sym").n_unique(),
    day=pl.col("t").first(),
)
e = e.with_columns(day=pl.from_epoch("day", "s").dt.date())
print(
    e.group_by("year")
    .agg(
        eps=pl.len(),
        sigs=pl.col("n").sum(),
        med_dur=pl.col("dur").median(),
        max_dur=pl.col("dur").max(),
        single=(pl.col("n") == 1).sum(),
        dur_ge20=(pl.col("dur") >= 20).sum(),
        dur_ge60=(pl.col("dur") >= 60).sum(),
    )
    .sort("year")
)
print(
    "dur quantiles",
    e["dur"].quantile(0.25),
    e["dur"].median(),
    e["dur"].quantile(0.75),
    e["dur"].quantile(0.9),
    "n/ep median",
    e["n"].median(),
)
for W in (0, 10, 15, 20, 30, 60):
    print(
        f"W={W}: episodes with any signal at off>=W: {(e['dur'] >= W).sum()} / {e.height}; signals kept {(sig['off'] >= W).sum()}"  # noqa: E501
    )
# replication
Y = (2022, 2023, 2024, 2025, 2026)
cand = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
res = {}
for W in (0, 10, 15, 20, 30, 60):
    keep = off >= W
    c = cand.filter(pl.Series(keep))
    b = simulate_yearly(c, Rule(hold_min=4320, tp=0.2, slots=10, cost=0.004, lev=1), years=Y)
    print(f"W={W} cand={c.height} n={b.summary['trades']} bp={b.summary['bp']:.0f}")
    print(
        "  ",
        " | ".join(
            f"{r['year']}:{r['trades']}t {r['bp']:+.0f}bp {r['ret'] * 100:+.0f}%"
            for r in b.years.iter_rows(named=True)
        ),
    )
    tr = b.trades.join(sig.select("row", "ep", "off"), on="row")
    tr.write_parquet(f"/tmp/w6_trades_W{W}.parquet")
# all candidates unconstrained outcome (1x, TP20/72h, cost .4%) for early vs late
from sim import exits, paths  # noqa: E402

p = paths(idx)
side = np.ones(len(idx), dtype=int)
ret, tx, kind = exits(
    p, side, Rule(hold_min=4320, tp=0.2, cost=0.004), __import__("sim").STEP_END_MIN, None
)
net = ret - 0.004
adv = np.maximum.accumulate(np.nan_to_num(-p[:, 1], nan=0.0), axis=1)[np.arange(len(idx)), tx]
sig = sig.with_columns(net=pl.Series(net), mae=pl.Series(adv))
sig.write_parquet("/tmp/w6_sig2.parquet")
