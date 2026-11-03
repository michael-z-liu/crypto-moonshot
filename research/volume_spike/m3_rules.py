"""M3: grid of simple capitulation rules, one open trade per symbol at a time, time exits."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common import RESULTS
from evalx import COST, load_events

HOLDS = {12: "f1h", 48: "f4h", 144: "fwd_12h", 288: "fwd_24h"}


def non_overlapping(sym: np.ndarray, t: np.ndarray, hold_bars: int) -> np.ndarray:
    """Keep an event only if the same symbol has no trade still open (entered < hold ago)."""
    keep = np.zeros(len(t), dtype=bool)
    last: dict[str, np.datetime64] = {}
    hold = np.timedelta64(hold_bars * 5, "m")
    for i in np.argsort(t, kind="stable"):
        s = sym[i]
        if s not in last or t[i] >= last[s] + hold:
            keep[i] = True
            last[s] = t[i]
    return keep


def main() -> None:
    events, fwd = load_events()
    base = events.filter(pl.col("is_spike") & pl.col("fwd_24h").is_not_null()).with_row_index("k")
    rows_idx = np.flatnonzero((events["is_spike"] & events["fwd_24h"].is_not_null()).to_numpy())
    base = base.with_columns(
        f1h=pl.Series(np.asarray(fwd[rows_idx, 2, 11])),
        f4h=pl.Series(np.asarray(fwd[rows_idx, 2, 47])),
    )
    out = []
    grid = itertools.product(
        ["ret_4h", "ret_24h"], [-0.06, -0.08, -0.10, -0.12, -0.15, -0.20],
        [0, 5_000, 20_000, 50_000], HOLDS.items(),
    )  # fmt: skip
    for feat, th, liq, (hold, col) in grid:
        sel = base.filter((pl.col(feat) <= th) & (pl.col("qv_base") >= liq))
        if sel.height == 0:
            continue
        keep = non_overlapping(sel["symbol"].to_numpy(), sel["open_time"].to_numpy(), hold)
        sel = sel.filter(pl.Series(keep)).with_columns(r=pl.col(col) - COST)
        for split in ("train", "valid"):
            s = sel.filter(pl.col("split") == split)
            if s.height < 20:
                continue
            yearly = s.group_by(pl.col("open_time").dt.year()).agg(pl.col("r").mean())["r"]
            r = s["r"].to_numpy()
            out.append(
                {
                    "feat": feat, "th": th, "liq": liq, "hold": hold, "split": split,
                    "n": len(r), "days": s["open_time"].dt.date().n_unique(),
                    "mean_bp": r.mean() * 1e4, "median_bp": float(np.median(r)) * 1e4,
                    "t": r.mean() / r.std() * np.sqrt(len(r)), "win": (r > 0).mean(),
                    "worst_year_bp": yearly.min() * 1e4,
                }
            )  # fmt: skip
    df = pl.DataFrame(out)
    df.write_csv(RESULTS / "m3_rules.csv")
    wide = df.pivot(
        on="split", index=["feat", "th", "liq", "hold"], values=["n", "mean_bp", "worst_year_bp"]
    ).sort("mean_bp_train", descending=True)
    with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=200):
        print(wide.filter(pl.col("n_train") >= 200).head(50))


if __name__ == "__main__":
    main()
