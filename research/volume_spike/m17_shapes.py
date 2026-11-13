"""Post-hoc (B1): which bracket shape suits the CNN's picks? Existing scores, no retraining.

Grid: take profit 3/5/8%/none x stop 2/3/5%/none x time stop 1h/2h, top 2%, 5 slots, 0.3% cost,
for the PR 8 base CNN and the bracket ensemble. Writes results/m17_shapes.csv.

Run: uv run python research/volume_spike/m17_shapes.py [check]
"""

from __future__ import annotations

import numpy as np
import polars as pl
from common import RESULTS
from m12_cnn_wf import WF, YEARS, Data
from portfolio import Exit, simulate


def main() -> None:
    data = Data()
    years = data.meta["year"].to_numpy()
    br = ["br1h", "br1h_w2h", "br1h_mkt", "br1h_bce", "br2h", "br30m"]
    pcts = {
        "base": np.nan_to_num(np.load(WF / "base.npz")["pct"], nan=-1),
        "br_ens": np.mean([np.nan_to_num(np.load(WF / f"{n}.npz")["pct"], nan=-1) for n in br], 0),
    }
    rows = []
    for name, pct in pcts.items():
        for year in YEARS:
            k = np.flatnonzero((years == year) & (pct >= 0.98))
            f = np.nan_to_num(np.asarray(data.fwd[k]))
            sig = data.meta[k].with_columns(neg=pl.Series(-pct[k]))
            for tp in (0.03, 0.05, 0.08, None):
                for sl in (0.02, 0.03, 0.05, None):
                    for bars in (12, 24):
                        st = simulate(sig, f, Exit(tp=tp, sl=sl, bars=bars), priority="neg").stats()
                        rows.append({"model": name, "tp": tp or 0.0, "sl": sl or 0.0,
                                     "hold_h": bars / 12, "year": year, "trades": st["trades"],
                                     "bp": st["mean_bp"], "ret": st["total_x"] - 1})  # fmt: skip
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m17_shapes.csv")
    w = df.pivot(on="year", index=["model", "tp", "sl", "hold_h"], values="bp")
    w = w.with_columns(
        sel=pl.mean_horizontal("2022", "2023", "2024"), late=pl.mean_horizontal("2025", "2026")
    ).sort("model", "sel", descending=True)
    w = w.with_columns(pl.col("tp", "sl") * 100, pl.exclude("model", "tp", "sl", "hold_h").round())
    with pl.Config(tbl_rows=80, tbl_cols=12, tbl_width_chars=200):
        print(w)


def check() -> None:
    """Post-hoc: the +5% target with no stop, 1h time exit, for the bracket ensemble.

    Volume (threshold x slots), risk (worst trade, drawdown) and stress (cost, one candle late,
    placebo). Writes results/m17_tp_only.csv.
    """
    from m17_bracket import delayed

    data = Data()
    years = data.meta["year"].to_numpy()
    br = ["br1h", "br1h_w2h", "br1h_mkt", "br1h_bce", "br2h", "br30m"]
    pct = np.mean([np.nan_to_num(np.load(WF / f"{n}.npz")["pct"], nan=-1) for n in br], 0)
    day = data.meta["day"].to_physical().to_numpy()
    rng = np.random.default_rng(0)
    placebo = pct.copy()
    for d in np.unique(day):
        k = np.flatnonzero(day == d)
        placebo[k] = rng.permutation(pct[k])
    late = delayed(data.fwd)
    ex = Exit(tp=0.05, sl=None, bars=12)
    rows = []
    cases = [
        ("default", pct, data.fwd, 0.003, 0.98, 5),
        ("cost 0.6%", pct, data.fwd, 0.006, 0.98, 5),
        ("one candle late", pct, late, 0.003, 0.98, 5),
        ("placebo", placebo, data.fwd, 0.003, 0.98, 5),
    ]
    cases += [(f"top {(1 - q) * 100:g}%, {s} slots", pct, data.fwd, 0.003, q, s)
              for q in (0.995, 0.99, 0.98, 0.95, 0.9) for s in (5, 10, 20)]  # fmt: skip
    for case, p, fw, cost, q, slots in cases:
        for year in YEARS:
            k = np.flatnonzero((years == year) & (p >= q))
            sig = data.meta[k].with_columns(neg=pl.Series(-p[k]))
            res = simulate(sig, np.nan_to_num(np.asarray(fw[k])), ex, slots=slots, cost=cost,
                           priority="neg")  # fmt: skip
            st, tr = res.stats(), res.trades
            if st["trades"] == 0:
                continue
            rows.append(
                {
                    "case": case,
                    "year": year,
                    "trades": st["trades"],
                    "bp": st["mean_bp"],
                    "ret": float((tr["stake"] * tr["ret"]).sum()),  # simple sum, from day one
                    "dd": st["max_dd"],
                    "win": st["win"],
                    "tp_rate": float((tr["ret"] > 0.05 - cost - 1e-9).mean()),
                    "worst": tr["ret"].min(),
                    "p5": tr["ret"].quantile(0.05),
                }
            )
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m17_tp_only.csv")
    with pl.Config(tbl_rows=120, tbl_cols=12, tbl_width_chars=200, float_precision=3):
        print(df)


if __name__ == "__main__":
    check() if __import__("sys").argv[1:] == ["check"] else main()
