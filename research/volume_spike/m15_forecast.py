"""How good are the CNN's 8h path forecasts (the user's forecasting-engine idea)?

For a path variant's walk-forward forecasts (10/50/90% quantiles of the net return at 15m..8h):
- calibration: how often the realised return falls below p10, inside p10..p90, above p90;
- ranking: correlation of the median forecast with the realised return, per horizon;
- shape: realised average path for each decile of the forecast peak;
- examples: forecast fans vs. what happened, for the top-forecast spikes of each year.

    uv run python research/volume_spike/m15_forecast.py path_all
"""

from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl
from common import CACHE, RESULTS
from m12_cnn_wf import PATH_STEPS, WF

COST = 0.003


def rank(v: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(v))


def main(variant: str) -> None:
    meta = pl.read_parquet(CACHE / "seq2_meta.parquet").with_columns(
        year=pl.col("open_time").dt.year()
    )
    fwd = np.load(CACHE / "fwd2.npy", mmap_mode="r")
    z = np.load(WF / f"{variant}.npz")
    path = z["path"].astype(np.float32)  # [n, steps, 3] net %, NaN where not scored
    ok = ~np.isnan(path[:, 0, 1]) & ~np.isnan(np.asarray(fwd[:, 2, -1]))  # scored, 8h outcome known
    k = np.flatnonzero(ok)
    real = (np.asarray(fwd[k, 2, :], dtype=np.float32)[:, PATH_STEPS] - COST) * 100
    p = path[k]
    years = meta["year"].to_numpy()[k]
    rows = []
    for j, step in enumerate(PATH_STEPS):
        r, lo, mid, hi = real[:, j], p[:, j, 0], p[:, j, 1], p[:, j, 2]
        rows.append({
            "minutes": int((step + 1) * 5),
            "below_p10": float((r < lo).mean()), "inside": float(((r >= lo) & (r <= hi)).mean()),
            "above_p90": float((r > hi).mean()),
            "corr": float(np.corrcoef(mid, np.clip(r, -30, 30))[0, 1]),
            "rank_corr": float(np.corrcoef(rank(mid), rank(r))[0, 1]),
            "mean_forecast": float(mid.mean()), "mean_real": float(r.mean()),
        })  # fmt: skip
    calib = pl.DataFrame(rows)
    peak = p[:, :, 1].max(1)
    dec = np.digitize(peak, np.quantile(peak, np.linspace(0, 1, 11)[1:-1]))
    deciles = []
    for d in range(10):
        m = dec == d
        deciles.append({
            "decile": d + 1, "n": int(m.sum()), "forecast_peak": float(peak[m].mean()),
            "real_peak": float(real[m].max(1).mean()),
            "real_path": [0.0, *np.clip(real[m], -30, 30).mean(0).round(1).tolist()],
            "forecast_path": [0.0, *p[m, :, 1].mean(0).round(1).tolist()],
        })  # fmt: skip
    examples = []
    for y in (2022, 2023, 2024, 2025, 2026):
        m = np.flatnonzero(years == y)
        top = m[np.argsort(-peak[m])[:2]]
        mid = m[np.argsort(np.abs(peak[m] - np.median(peak[m])))[:1]]
        for i in [*top, *mid]:
            row = meta[int(k[i])]
            examples.append({
                "year": y, "symbol": row["symbol"][0], "time": row["open_time"][0].isoformat(),
                "kind": "top forecast" if i in top else "typical forecast",
                "p10": p[i, :, 0].round(2).tolist(), "p50": p[i, :, 1].round(2).tolist(),
                "p90": p[i, :, 2].round(2).tolist(), "real": real[i].round(2).tolist(),
            })  # fmt: skip
    calib.write_csv(RESULTS / f"m15_forecast_calibration_{variant}.csv")
    (RESULTS / f"m15_forecast_{variant}.json").write_text(
        json.dumps({"calibration": calib.to_dicts(), "deciles": deciles, "examples": examples})
        + "\n"
    )
    with pl.Config(tbl_rows=40, float_precision=3):
        print(calib)
    for d in deciles:
        rp = d["real_path"]
        print(d["decile"], d["n"], round(d["forecast_peak"], 2), round(d["real_peak"], 2),
              "real 1h", rp[4], "4h", rp[16], "8h", rp[32])  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1])
