"""X2: books, signal-bin table, trade files for the picked and baseline configs."""

from __future__ import annotations

import numpy as np
import polars as pl
import x2_quiet_eval as ev  # re-runs grid prints (cheap)
from x2_quiet_build import CONFIGS, OWN

df, outs, sig, yr, side = ev.df, ev.outs, ev.sig, ev.yr, ev.side
KIND = {0: "time", 1: "tp", 2: "stop", 3: "liq"}
CI = {c: i for i, c in enumerate(CONFIGS)}
picks = {
    "Q_long_30d_tp30_stop25_thr0.5": (1, CI[(336, 0.3, 0.25)], 0.5),
    "BASE_long_30d_tp30_stop25_nofilter": (1, CI[(336, 0.3, 0.25)], 0.0),
    "Q_short_7d_tp30_thr0.9": (-1, CI[(168, 0.3, None)], 0.9),
    "BASE_short_7d_tp30_nofilter": (-1, CI[(168, 0.3, None)], 0.0),
}
print("\n== books (10 slots, $10k per year, stake<=0.5% of 24h vol) ==")
tr = []
for name, (sd, ci, th) in picks.items():
    sel = (side == sd) & (sig >= th) & (yr >= 2021)
    bk = [ev.book(sel, ci, y) for y in range(2021, 2027)]
    net = outs[:, ci, 0]
    print(
        name,
        "n",
        int(sel.sum()),
        "avg",
        f"{net[sel].mean():+.3f}",
        "median",
        f"{np.median(net[sel]):+.3f}",
        "book",
        [f"{r:+.0%}/{n}" for r, n in bk],
    )
    H, tp, st = CONFIGS[ci]
    idx = np.where(sel)[0]
    sub = df[idx]
    tr.append(
        pl.DataFrame(
            {
                "strategy": "quiet_breakout",
                "config": name,
                "symbol": sub["symbol"],
                "side": sub["side"],
                "t_in": sub["t_in"],
                "t_out": outs[idx, ci, 2].astype(np.int64),
                "signal": sub["signal"],
                "threshold": th,
                "net": outs[idx, ci, 0],
                "mfe": outs[idx, ci, 3],
                "mae": outs[idx, ci, 4],
                "tp": tp,
                "stop": (st if st else (1.0 if sd == -1 else None)),
                "exit_kind": [KIND[int(k)] for k in outs[idx, ci, 1]],
                "year": sub["year"],
            },
            schema_overrides={"tp": pl.Float64, "stop": pl.Float64},
        )
    )
pl.concat(tr).write_parquet(OWN + "quiet_trades.parquet")
# non-entries: below-threshold candidates of the picked long config
sd, ci, th = picks["Q_long_30d_tp30_stop25_thr0.5"]
idx = np.where((side == sd) & (sig < th) & (yr >= 2021))[0]
idx = np.random.default_rng(0).choice(idx, min(20000, len(idx)), replace=False)
H, tp, st = CONFIGS[ci]
sub = df[idx]
pl.DataFrame(
    {
        "strategy": "quiet_breakout",
        "config": "Q_long_30d_tp30_stop25_thr0.5",
        "symbol": sub["symbol"],
        "side": sub["side"],
        "t_in": sub["t_in"],
        "t_out": outs[idx, ci, 2].astype(np.int64),
        "signal": sub["signal"],
        "threshold": th,
        "net": outs[idx, ci, 0],
        "mfe": outs[idx, ci, 3],
        "mae": outs[idx, ci, 4],
        "tp": tp,
        "stop": st,
        "exit_kind": [KIND[int(k)] for k in outs[idx, ci, 1]],
        "year": sub["year"],
    }
).write_parquet(OWN + "quiet_nonentries.parquet")
print("\n== signal bins, long, TP30/stop25/30d: n, mean, median, TP-hit share ==")
net = outs[:, ci, 0]
kind = outs[:, ci, 1]
for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
    s = (side == 1) & (sig >= lo) & (sig < hi) & (yr >= 2021)
    print(
        lo,
        hi,
        int(s.sum()),
        f"{net[s].mean():+.3f}",
        f"{np.median(net[s]):+.3f}",
        f"{(kind[s] == 1).mean():.2f}",
    )
# quiet-only effect: quiet component alone
q = (side == 1) & (yr >= 2021)
for nm, col in (("rv<0.7", "rv"), ("rq<0.7", "rq")):
    v = df[col].to_numpy()
    for lab, s in (("quiet", q & (v < 0.7)), ("loud", q & (v >= 1.0))):
        print(nm, lab, int(s.sum()), f"{net[s].mean():+.3f}", f"{np.median(net[s]):+.3f}")
# 2021 outliers
print(
    "2021 baseline long top trades:",
    pl.DataFrame({"s": df["symbol"][np.where(q & (yr == 2021))[0]], "n": net[q & (yr == 2021)]})
    .sort("n", descending=True)
    .head(5)
    .rows(),
)
