"""E08: walk-forward gradient-boosted ranking of big-move events, long (crashes) or short (pumps).

For each test year Y (2022..2026): fit on events before Y (from 2020-06), predict Y. Target: the
trade's price return at the horizon in the trade's direction, clipped to [-0.5, 1]. Evaluation:
yearly rank IC, top-quantile mean returns, and a book (perp-gated with funding for shorts; spot
costs for longs) trading events whose prediction beats the training predictions' q-quantile.

Run: uv run --group research python e08_gbm.py short|long [horizon=72h]
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from common_ls import OWN
from sklearn.ensemble import HistGradientBoostingRegressor
from sim import Rule, fmt_years, simulate_yearly

SIDE = sys.argv[1] if len(sys.argv) > 1 else "short"
H = sys.argv[2] if len(sys.argv) > 2 else "72h"
S = -1 if SIDE == "short" else 1
FEATS = [
    "ret_4h", "ret_24h", "vol_mult", "btc_vol_mult", "btc_ret_4h", "listed_days", "hour", "qv_base",
    "red", "off_high4h", "off_low4h", "off_high24h", "off_low24h", "r5m", "r15m", "r1h", "r2h", "r8h",
    "wick_up", "wick_dn", "taker", "taker1h", "taker4h", "trades_x", "vol1h", "vol4h", "vol_prev20h",
    "n_hot_bars4h", "btc_r1h", "btc_r24h", "btc_vol1h", "eth_r4h", "eth_r24h", "mkt_r4h", "mkt_r24h",
    "spike_share", "spike_share1h", "r7d", "r30d", "r90d", "dhi30d", "dlo30d", "dhi90d", "dlo90d",
    "vol30d", "qv7_30", "n_prior_pumps30", "rate", "rate24",
]  # fmt: skip


def load() -> pl.DataFrame:
    d = pl.read_parquet(OWN / "feats.parquet")
    d = d.filter(pl.col("ret_4h") >= 0.15) if S < 0 else d.filter(pl.col("ret_4h") <= -0.10)
    return d.filter(pl.col("qv_base") >= 20_000).with_columns(
        target=(S * pl.col(f"y{H}")).clip(-0.5, 1.0), year=pl.col("open_time").dt.year()
    )


def fit_predict(d: pl.DataFrame, seed: int = 0) -> np.ndarray:
    pred = np.full(d.height, np.nan)
    ref = np.full(d.height, np.nan)  # the q-quantile reference: training predictions
    X = d.select(FEATS).to_numpy().astype(np.float32)
    y = d["target"].to_numpy()
    yr = d["year"].to_numpy()
    for Y in (2022, 2023, 2024, 2025, 2026):
        tr, te = yr < Y, yr == Y
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=31,
                                            min_samples_leaf=200, l2_regularization=1.0, random_state=seed)  # fmt: skip
        mdl.fit(X[tr], y[tr])
        p_tr = mdl.predict(X[tr])
        p_te = mdl.predict(X[te])
        pred[te] = p_te
        # percentile of each test prediction among the training predictions (no test info)
        ref[te] = np.searchsorted(np.sort(p_tr), p_te) / len(p_tr)
    return np.stack([pred, ref])


def main() -> None:
    d = load()
    pr = fit_predict(d)
    d = d.with_columns(pred=pl.Series(pr[0]), pq=pl.Series(pr[1]))
    t = d.filter(pl.col("year") >= 2022)
    print(f"{SIDE} {H}: {t.height} test rows")
    ic = t.group_by("year").agg(ic=pl.corr(pl.col("pred").rank(), pl.col("target").rank()), n=pl.len()).sort("year")
    print(ic)
    print(t.with_columns(b=pl.col("pq").cut([0.5, 0.8, 0.9, 0.95])).group_by("b").agg(
        n=pl.len(), mean=pl.col("target").mean() * 1e4, med=pl.col("target").median() * 1e4).sort("b"))  # fmt: skip
    d.select("row", "pred", "pq").write_parquet(OWN / f"e08_{SIDE}_{H}.parquet")
    hold = {"4h": 240, "24h": 1440, "72h": 4320}[H]
    base = t.filter(pl.col("perp")) if S < 0 else t
    for q in (0.0, 0.5, 0.8, 0.9):
        c = base.filter(pl.col("pq") >= q).select("row", side=pl.lit(S), prio="pred")
        for slots in (3, 5):
            b = simulate_yearly(c, Rule(hold_min=hold, slots=slots, cost=0.003 if S < 0 else 0.004), perp_funding=S < 0)
            print(f"q>={q:.2f} slots {slots}: {b.summary['bp']:+.0f}bp  {fmt_years(b, (2022, 2023, 2024, 2025, 2026))}")
    # Rule baselines on the same rows.
    for thr in (0.3, 0.4, 0.6):
        c = base.filter((pl.col("ret_4h") >= thr) if S < 0 else (pl.col("ret_4h") <= -0.15)).select("row", side=pl.lit(S), prio=(S * pl.col("ret_4h")))
        b = simulate_yearly(c, Rule(hold_min=hold, slots=5, cost=0.003 if S < 0 else 0.004), perp_funding=S < 0)
        print(f"rule thr {thr}: {b.summary['bp']:+.0f}bp  {fmt_years(b, (2022, 2023, 2024, 2025, 2026))}")


if __name__ == "__main__":
    main()
