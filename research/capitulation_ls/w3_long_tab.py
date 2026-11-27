"""W3: per-row table (scored rows 2022+) with realistic TP20/72h outcome, path stats and scores."""

import numpy as np
import polars as pl
from common_ls import meta, scores
from sim import STEP_END_MIN, Rule, exits, paths

m = meta()
names = ["day5", "tp20day", "day10b", "day_transformer", "day_tcn", "day_gru"]
S = {n: scores(n) for n in names}
ok = np.isfinite(S["day5"]) & (m["year"].to_numpy() >= 2022)
rows = np.flatnonzero(ok)
out = {k: [] for k in ("o72", "peak", "trough", "c72", "t_tp", "c8", "c1")}
for a in range(0, len(rows), 100_000):
    r = rows[a : a + 100_000]
    p = paths(r)
    ret, te, kind = exits(p, np.ones(len(r)), Rule(hold_min=4320, tp=0.2))
    out["o72"].append(ret - 0.004)
    hi = np.nan_to_num(p[:, 0], nan=-1)
    lo = np.nan_to_num(p[:, 1], nan=0)
    out["peak"].append(hi.max(1))
    out["trough"].append(lo.min(1))
    out["c72"].append(np.nan_to_num(p[:, 2, -1]))
    out["t_tp"].append(np.where(kind == 2, STEP_END_MIN[np.minimum(te, 159)], -1))
    out["c8"].append(np.nan_to_num(p[:, 2, 95]))
    out["c1"].append(np.nan_to_num(p[:, 2, 11]))
d = pl.DataFrame({"row": rows, **{k: np.concatenate(v) for k, v in out.items()}})
mm = m.select(
    pl.int_range(pl.len()).alias("row"),
    "symbol",
    "day",
    "year",
    "ret_4h",
    "btc_ret_4h",
    "y1h",
    "y4h",
    "qv_base",
)
d = d.join(mm, on="row").with_columns(**{n: pl.Series(S[n][rows]) for n in names})
d.write_parquet("/tmp/w3_tab.parquet")
print(d.shape)
