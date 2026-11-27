"""w6-B: per-episode anatomy: duration, equal-weight 1m index low time, early vs late outcomes."""

import contextlib
import datetime as dt

import numpy as np
import polars as pl

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(250)
K = "/Users/mike/code/crypto-moonshot/data/klines/spot/interval=1m/symbol={}/{}.parquet"
sig = pl.read_parquet("/tmp/w6_sig2.parquet")
HOR = 720  # minutes after episode start
rows = []
sig_rows = []
for ep, g in sig.group_by("ep", maintain_order=True):
    ep = ep[0]
    t0 = int(g["t"].min())  # epoch s of first signal bar open; entry = t0+300
    start = t0 // 60 * 60
    n = HOR + 10
    grid0 = start - 10 * 60
    mats = []
    syms = []
    for s in g.sort("t")["sym"].unique(maintain_order=True).to_list()[:80]:
        d0 = dt.datetime.fromtimestamp(grid0, dt.UTC)
        d1 = dt.datetime.fromtimestamp(grid0 + n * 60, dt.UTC)
        ms = sorted({f"{d0:%Y-%m}", f"{d1:%Y-%m}"})
        fr = []
        for mo in ms:
            with contextlib.suppress(Exception):
                fr.append(pl.read_parquet(K.format(s, mo), columns=["open_time", "close", "low"]))
        if not fr:
            continue
        d = pl.concat(fr).with_columns(ts=pl.col("open_time").dt.epoch("s"))
        d = d.filter((pl.col("ts") >= grid0) & (pl.col("ts") < grid0 + n * 60))
        if d.height < 30:
            continue
        arr = np.full(n, np.nan)
        lo = np.full(n, np.nan)
        k = ((d["ts"].to_numpy() - grid0) // 60).astype(int)
        arr[k] = d["close"].to_numpy()
        lo[k] = d["low"].to_numpy()
        # forward-fill close
        s_ = pl.Series(arr).fill_nan(None).forward_fill().to_numpy()
        mats.append((s, s_, lo))
    ref_i = 10  # index of episode start minute
    rel = []
    for s, a, _lo in mats:
        if np.isnan(a[ref_i - 1]):
            continue
        rel.append((s, a / a[ref_i - 1]))
    if not rel:
        continue
    M = np.vstack([r for _, r in rel])
    ew = np.nanmean(M, axis=0)
    ew_min = int(np.nanargmin(ew[ref_i:]))
    low_val = ew[ref_i:][ew_min] - 1
    last = g["off"].max()
    # signal timing vs index low
    gg = g.with_columns(before_low=(pl.col("off") + 5 <= ew_min))  # entry = off+5 min
    # time at which index is within 2pts of its low, rebound at +60/+120 after low
    rows.append(
        dict(
            ep=ep,
            day=dt.datetime.fromtimestamp(t0, dt.UTC).strftime("%Y-%m-%d %H:%M"),
            year=int(g["year"][0]),
            sigs=g.height,
            coins=g["sym"].n_unique(),
            idx_coins=len(rel),
            dur=last,
            low_min=ew_min,
            low_pct=low_val * 100,
            ew_at_20=(ew[ref_i + 20] - 1) * 100,
            ew_at_60=(ew[ref_i + 60] - 1) * 100,
            ew_end=(ew[ref_i + HOR - 1] - 1) * 100,
            early_net=gg.filter(pl.col("off") < 20)["net"].mean(),
            late_net=gg.filter(pl.col("off") >= 20)["net"].mean(),
            n_early=gg.filter(pl.col("off") < 20).height,
            n_late=gg.filter(pl.col("off") >= 20).height,
            share_before_low=gg["before_low"].mean(),
            net_before=gg.filter(pl.col("before_low"))["net"].mean(),
            net_after=gg.filter(~pl.col("before_low"))["net"].mean(),
            last_vs_low=last + 5 - ew_min,
        )
    )
E = pl.DataFrame(rows)
E.write_parquet("/tmp/w6_ep.parquet")
print(
    E.with_columns(
        (pl.col(c) * 1e4).round(0) for c in ("early_net", "late_net", "net_before", "net_after")
    ).drop("ep")
)
print(
    "low_min quantiles",
    E["low_min"].quantile(0.25),
    E["low_min"].median(),
    E["low_min"].quantile(0.75),
)
print(
    "share eps with low within 30/60/120 min of start",
    [(E["low_min"] <= x).mean() for x in (30, 60, 120)],
)
print("last signal vs low (min) median", E["last_vs_low"].median())
