"""X2 quiet breakouts: signal, grid, per-year book, trade files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from x2_quiet_build import CONFIGS, OWN

df = pl.read_parquet(OWN + "x2_cand.parquet")
outs = np.load(OWN + "x2_outs.npy")
df = df.filter(pl.col("year") >= 0)
yr = df["year"].to_numpy()
side = df["side"].to_numpy()
sig = np.zeros(df.height)


def pct(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    ref = np.sort(ref)
    return np.searchsorted(ref, x) / len(ref)


for sd in (1, -1):
    m = side == sd
    ref = m & (yr >= 2020) & (yr <= 2023)  # in-sample reference
    qraw = 0.5 * np.log(df["rv"].to_numpy()) + 0.5 * np.log(df["rq"].to_numpy())
    quiet = 1 - pct(qraw[ref], qraw[m])
    surge, mag = df["surge"].to_numpy(), df["mag"].to_numpy()
    b = 0.5 * pct(surge[ref], surge[m]) + 0.5 * pct(mag[ref], mag[m])
    sig[m] = np.sqrt(np.clip(quiet, 0, 1) * np.clip(b, 0, 1))
df = df.with_columns(signal=pl.Series(sig))
t_in = df["t_in"].to_numpy()
q24 = df["qv24"].to_numpy()
THRS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]


def book(sel: np.ndarray, ci: int, y: int) -> tuple[float, int]:
    idx = np.where(sel & (yr == y))[0]
    idx = idx[np.argsort(t_in[idx])]
    eq, open_ = 10_000.0, []  # (t_out, stake, ret)
    n = 0
    for i in idx:
        t = t_in[i]
        for o in [o for o in open_ if o[0] <= t]:
            eq += o[1] * o[2]
            open_.remove(o)
        if len(open_) >= 10:
            continue
        stake = min(eq / 10, 0.005 * q24[i])
        if stake < 10:
            continue
        net, _, tout, _, _ = outs[i, ci]
        open_.append((tout, stake, net))
        n += 1
    for o in open_:
        eq += o[1] * o[2]
    return eq / 10_000 - 1, n


lines = [
    "side,hold_h,tp,stop,thr,n,mean_is,mean_21_23,mean_24_26,y2021,y2022,y2023,y2024,y2025,y2026,book_min_year"
]
res = []
for sd in (1, -1):
    for ci, (H, tp, st) in enumerate(CONFIGS):
        net = outs[:, ci, 0]
        for th in THRS:
            sel = (side == sd) & (sig >= th) & (yr >= 2021)
            if sel.sum() < 150:
                continue
            ys = [
                net[sel & (yr == y)].mean() if (sel & (yr == y)).sum() >= 20 else np.nan
                for y in range(2021, 2027)
            ]
            ins = sel & (yr <= 2023)
            oos = sel & (yr >= 2024)
            res.append(
                (
                    sd,
                    ci,
                    th,
                    int(sel.sum()),
                    net[ins].mean() if ins.sum() else np.nan,
                    net[oos].mean() if oos.sum() else np.nan,
                    ys,
                )
            )
            lines.append(
                f"{sd},{H},{tp},{st},{th},{sel.sum()},{net[ins].mean():.4f},{net[ins].mean():.4f},"
                f"{net[oos].mean() if oos.sum() else float('nan'):.4f},"
                + ",".join(f"{v:.4f}" for v in ys)
                + ","
            )
Path("results/explore/x2_grid.csv").write_text("\n".join(lines))
print(len(res), "configs x thresholds tried per side (n>=150)")
# in-sample pick (2021-23 mean, n_is>=150, at least thr 0.5 for the "quiet" family)
for sd in (1, -1):
    cand = [r for r in res if r[0] == sd and r[2] >= 0.5 and not np.isnan(r[4])]
    cand.sort(key=lambda r: -r[4])
    print("side", sd, "top in-sample")
    for r in cand[:8]:
        print(
            CONFIGS[r[1]],
            r[2],
            r[3],
            f"IS {r[4]:+.3f} OOS {r[5]:+.3f}",
            [f"{v:+.2f}" for v in r[6]],
        )
    base = [r for r in res if r[0] == sd and r[2] == 0.0]
    base.sort(key=lambda r: -r[4])
    print("side", sd, "baseline (no quiet filter) best")
    for r in base[:3]:
        print(CONFIGS[r[1]], r[3], f"IS {r[4]:+.3f} OOS {r[5]:+.3f}", [f"{v:+.2f}" for v in r[6]])
np.save(OWN + "x2_sig.npy", sig)
