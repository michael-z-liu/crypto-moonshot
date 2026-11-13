import sys

sys.path.insert(0, ".")
import m12_cnn_wf as m
import numpy as np
import polars as pl
from evalx import load_events
from portfolio import Exit, simulate

rows = []
ev, fwd = load_events()
f2 = (
    pl.col("is_spike")
    & (pl.col("ret_4h") <= -0.10)
    & (pl.col("qv_base") >= 20_000)
    & (pl.col("btc_vol_mult") >= 6)
)
idx = np.flatnonzero(ev.select(f2.fill_null(False).alias("m"))["m"].to_numpy())
yr = ev["open_time"].dt.year().to_numpy()
for y in range(2020, 2027):
    k = idx[yr[idx] == y]
    res = simulate(ev[k], np.asarray(fwd[k]), Exit(bars=12), slots=5, priority="ret_4h")
    st = res.stats()
    rows.append(
        dict(
            strategy="F2 rule",
            year=y,
            trades=st["trades"],
            bp=st["mean_bp"],
            ret=float((res.trades["stake"] * res.trades["ret"]).sum()),
            dd=st["max_dd"],
        )
    )
d = m.Data()
years = d.meta["year"].to_numpy()
for name, label in (("base", "Base CNN"), ("path_all", "Path forecaster CNN")):
    pct = np.nan_to_num(np.load(m.WF / f"{name}.npz")["pct"], nan=-1)
    for y in m.YEARS:
        k = np.flatnonzero((years == y) & (pct >= m.Q_MAIN))
        sig = d.meta[k].with_columns(neg=pl.Series(-pct[k]))
        res = simulate(
            sig, np.asarray(d.fwd[k]), Exit(bars=12), slots=5, cost=m.COST, priority="neg"
        )
        st = res.stats()
        rows.append(
            dict(
                strategy=label,
                year=y,
                trades=st["trades"],
                bp=st["mean_bp"],
                ret=float((res.trades["stake"] * res.trades["ret"]).sum()),
                dd=st["max_dd"],
            )
        )
df = pl.DataFrame(rows)
df.write_csv("results/replication/yearly_summary.csv")
with pl.Config(tbl_rows=40, float_precision=3):
    print(df)
