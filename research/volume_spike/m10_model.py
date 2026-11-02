"""M10: a gradient-boosted model over all spike features, predicting the 1h net return.

Trained on 2020-2023 spikes, threshold chosen on 2024, then evaluated on 2025-2026 (a period
already inspected for the hand-made rules, but never used to fit or tune this model).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from common import RESULTS
from evalx import COST, load_events
from m3_rules import non_overlapping
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = [
    "vol_mult", "vol_mult_3h", "tbr", "tbr_base", "tbr_excess", "ret_bar", "range_bar",
    "range_mult",
    "clv", "upper_wick", "runup_1h", "ret_4h", "ret_24h", "ret_7d", "dist_hi_24h", "dist_hi_7d",
    "dist_lo_24h", "vola_24h", "qv_base", "qv_usdt", "trades_mult", "avg_trade_mult",
    "prev_spikes_1h", "prev_spikes_24h", "tick_rel", "flat_share", "listed_days", "btc_ret_bar",
    "btc_runup_1h", "btc_ret_4h", "btc_ret_24h", "btc_ret_7d", "btc_vol_mult", "btc_vola_24h",
    "hour", "dow",
]  # fmt: skip
HOLD = 12


def main() -> None:
    ev, fwd = load_events()
    idx = np.flatnonzero(ev["is_spike"].to_numpy())
    sp = ev[idx].with_columns(y=pl.Series(np.asarray(fwd[idx, 2, HOLD - 1])) - COST)
    X = sp.select(FEATURES).cast(pl.Float64).fill_nan(None).to_numpy()
    y = np.clip(sp["y"].to_numpy(), -0.2, 0.2)
    split = sp["split"].to_numpy()
    tr = np.flatnonzero(split == "train")
    rng = np.random.default_rng(0)
    sub = rng.choice(tr, size=min(400_000, len(tr)), replace=False)
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
        l2_regularization=1.0, random_state=0,
    )  # fmt: skip
    model.fit(X[sub], y[sub])
    sp = sp.with_columns(pred=pl.Series(model.predict(X)))
    rows = []
    for q in (0.99, 0.995, 0.998, 0.999):
        th = float(np.quantile(sp.filter(pl.col("split") == "train")["pred"].to_numpy(), q))
        for s in ("train", "valid", "test"):
            d = sp.filter((pl.col("split") == s) & (pl.col("pred") >= th))
            keep = non_overlapping(d["symbol"].to_numpy(), d["open_time"].to_numpy(), HOLD)
            d = d.filter(pl.Series(keep))
            days = d.group_by(pl.col("open_time").dt.date()).agg(pl.col("y").mean())["y"]
            weeks = (d["open_time"].max() - d["open_time"].min()).days / 7 if d.height > 1 else 1
            rows.append({"quantile": q, "threshold_bp": th * 1e4, "split": s, "trades": d.height,
                         "per_week": d.height / max(weeks, 1), "mean_bp": d["y"].mean() * 1e4,
                         "day_bp": days.mean() * 1e4, "win_days": (days > 0).mean(),
                         "days": days.len()})  # fmt: skip
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m10_model.csv")
    with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=200, float_precision=2):
        print(df)
    sp.filter(
        pl.col("pred") >= np.quantile(sp.filter(pl.col("split") == "train")["pred"], 0.998)
    ).select(
        "symbol", "open_time", "split", "pred", "y", "ret_4h", "btc_vol_mult", "qv_base"
    ).write_parquet(RESULTS.parent / "cache" / "model_top.parquet")


if __name__ == "__main__":
    main()
