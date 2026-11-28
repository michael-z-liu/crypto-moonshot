"""X11: limit entries below the signal price for the long sleeve L (in-sample exploration).

Instead of buying at the next 5m open, rest a buy limit X% below it for up to W hours. It fills
only if a 5m low trades at least 0.1% through the limit (queue safety). From the fill, the take
profit is +20% of the fill price and the time exit stays 72h after the signal. The fill bar is
not used for exits (its high may precede its low). Unfilled candidates free their slot.
Per-trade results over every candidate, then the 10-slot LMC book with L replaced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from common_ls import OWN, meta, scores
from diag import annotate, scorecard
from e25_combined import book
from sim import STEP_END_MIN, paths

OUT = Path(__file__).resolve().parent / "results" / "explore"
COST, TP, COLS = 0.004, 0.20, ["symbol", "adv", "t_in", "t_out", "net", "qv", "sleeve", "prio"]


def limit_trades(x: float, win_h: float, rows: np.ndarray, pct: np.ndarray) -> pl.DataFrame:
    m = meta()
    p = paths(rows)  # [n, 3, 160] high, low, close relative to the market entry
    hi, lo, cl = p[:, 0], p[:, 1], p[:, 2]
    n_win = int(win_h * 12)  # 5m steps (the first 96 steps are 5m bars)
    lvl = -x
    out = []
    t0 = m["open_time"].dt.epoch("s").to_numpy()[rows] + 300
    sym = m["symbol"].to_numpy()[rows]
    qv = m["qv_base"].to_numpy()[rows] * 288
    for i in range(len(rows)):
        if x > 0:
            hit = np.flatnonzero(lo[i, :n_win] <= lvl - 0.001)
            if not len(hit):
                continue
            k = int(hit[0])
            fill = 1 + lvl
        else:
            k, fill = -1, 1.0
        rel_hi = (1 + hi[i, k + 1 :]) / fill - 1
        rel_lo = (1 + lo[i, k + 1 :]) / fill - 1
        rel_cl = (1 + cl[i, k + 1 :]) / fill - 1
        j = np.flatnonzero(rel_hi >= TP)
        if len(j):
            e, ret, kind = k + 1 + int(j[0]), TP, "tp"
        else:
            e, ret, kind = p.shape[2] - 1, float(rel_cl[-1]), "time"
        seg = slice(0, e - k)
        adv = float(max(0.0, -np.nanmin(rel_lo[seg]))) if e > k else 0.0
        mfe = float(max(0.0, np.nanmax(rel_hi[seg]))) if e > k else 0.0
        t_fill = t0[i] + (int(STEP_END_MIN[k]) * 60 if k >= 0 else 0)
        out.append({"symbol": sym[i], "adv": adv, "mfe": mfe, "t_in": int(t_fill),
                    "t_out": int(t0[i] + STEP_END_MIN[e] * 60), "net": ret - COST,
                    "qv": float(qv[i]), "sleeve": "L", "prio": 3.0 + float(pct[i]),
                    "kind": kind, "year": int(m["year"][int(rows[i])])})  # fmt: skip
    return pl.DataFrame(out)


if __name__ == "__main__":
    m = meta()
    pct_all = scores("day5")
    yr = m["year"].to_numpy()
    rows = np.flatnonzero((yr >= 2022) & np.isfinite(pct_all) & (pct_all >= 0.995))
    pct = pct_all[rows]
    sl = pl.read_parquet(OWN / "explore" / "x8_sleeves.parquet")
    mc = sl.filter(pl.col("sleeve").is_in(["M", "C"])).select(COLS)
    lines = [
        f"# X11: limit entries below the signal price for L (candidates n={len(rows)}, 2022-26)",
        "",
        "| limit | window h | filled | fill % | mean net bp | 22-24 bp | 25-26 bp | TP % | E_o | "
        "LMC trades | LMC bp | LMC min yr % | LMC years 22..26 % |",
        "|" + "---|" * 13,
    ]
    for x, w in [(0.0, 0.0), (0.02, 1), (0.02, 4), (0.04, 1), (0.04, 4), (0.06, 4), (0.08, 4),
                 (0.10, 8), (0.04, 8), (0.06, 8)]:  # fmt: skip
        t = limit_trades(x, w, rows, pct)
        f = t.select(signal=pl.lit(1.0), threshold=pl.lit(0.0), net="net", mfe="mfe", mae="adv",
                     tp=pl.lit(TP), stop=pl.lit(None, dtype=pl.Float64), exit_kind="kind",
                     year="year")  # fmt: skip
        sc = scorecard(annotate(f))
        a = t.filter(pl.col("year") <= 2024)["net"].mean() or 0.0
        b_ = t.filter(pl.col("year") >= 2025)["net"].mean() or 0.0
        b = book(pl.concat([t.select(COLS), mc], how="vertical_relaxed"), 10)
        n = int(b["trades"].sum())
        bp = float((b["bp"] * b["trades"]).sum() / n)
        yrs = "/".join(f"{r['ret'] * 100:+.0f}" for r in b.iter_rows(named=True))
        row = (f"| {x:.0%} | {w} | {t.height} | {t.height / len(rows):.0%} | {t['net'].mean() * 1e4:+.0f} | "  # noqa: E501
               f"{a * 1e4:+.0f} | {b_ * 1e4:+.0f} | {(t['kind'] == 'tp').mean():.0%} | {sc['E_o']:+.3f} | "  # noqa: E501
               f"{n} | {bp:+.0f} | {b['ret'].min() * 100:+.0f} | {yrs} |")  # fmt: skip
        lines.append(row)
        print(row, flush=True)
    (OUT / "limit_entry.md").write_text("\n".join(lines) + "\n")
