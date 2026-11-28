"""E17: can squeezes be predicted? Walk-forward GBM on pump shorts (perp paths), with and without
the futures-metric features. Then books that skip the riskiest predicted pumps, under realistic
tails (isolated 1x: liquidation at +100%; cross margin: losses beyond the stake)."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN
from e08_gbm import FEATS
from sim import Rule, fmt_years, perp_paths, simulate_yearly
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

MF = [
    "oi_ch1h",
    "oi_ch4h",
    "oi_ch24h",
    "oi_to_vol",
    "top_ls_acc",
    "top_ls_pos",
    "ls_acc",
    "taker_ls",
    "ls_acc_ch4h",
    "top_ls_pos_ch4h",
    "taker_ls_1h",
]


def main() -> None:
    d = pl.read_parquet(OWN / "feats.parquet").filter(
        pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("ret_4h") >= 0.25)
    )
    d = d.join(pl.read_parquet(OWN / "metric_feats.parquet"), on="row", how="left").sort("row")
    pp, ok = perp_paths(d["row"].to_numpy())
    d = d.filter(pl.Series(ok))
    H72 = 96 + 63
    adv = np.nan_to_num(pp[:, 0, : H72 + 1], nan=0).max(1)
    d = d.with_columns(
        adv72=pl.Series(adv),
        sq50=pl.Series(adv >= 0.5),
        sq100=pl.Series(adv >= 1.0),
        short72=pl.Series(-np.nan_to_num(pp[:, 2, H72], nan=0)),
        year=pl.col("open_time").dt.year(),
    )
    print(
        d.height,
        "pump rows; squeeze>=50%:",
        round(float(d["sq50"].mean()) * 100, 1),
        "%, >=100%:",
        round(float(d["sq100"].mean()) * 100, 1),
        "%",
    )
    print(
        d.group_by(pl.col("oi_ch4h").qcut(5, allow_duplicates=True))
        .agg(n=pl.len(), sq50=pl.col("sq50").mean(), s72=pl.col("short72").mean())
        .sort("oi_ch4h")
    )
    print(
        d.group_by(pl.col("ls_acc").qcut(5, allow_duplicates=True))
        .agg(n=pl.len(), sq50=pl.col("sq50").mean(), s72=pl.col("short72").mean())
        .sort("ls_acc")
    )
    print(
        d.group_by(pl.col("rate").qcut(5, allow_duplicates=True))
        .agg(n=pl.len(), sq50=pl.col("sq50").mean(), s72=pl.col("short72").mean())
        .sort("rate")
    )

    risk = {}
    for name, feats in (("base", FEATS), ("metrics", FEATS + MF)):
        X = d.select(feats).to_numpy().astype(np.float32)
        yr = d["year"].to_numpy()
        out = np.full(d.height, np.nan)
        ret = np.full(d.height, np.nan)
        aucs = []
        for Y in (2022, 2023, 2024, 2025, 2026):
            tr, te = yr < Y, yr == Y
            clf = HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.04,
                min_samples_leaf=100,
                l2_regularization=1.0,
                random_state=0,
            )
            clf.fit(X[tr], d["sq50"].to_numpy()[tr])
            out[te] = clf.predict_proba(X[te])[:, 1]
            reg = HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.04,
                min_samples_leaf=100,
                l2_regularization=1.0,
                random_state=0,
            )
            reg.fit(X[tr], np.clip(d["short72"].to_numpy()[tr], -1, 1))
            ret[te] = reg.predict(X[te])
            aucs.append(round(roc_auc_score(d["sq50"].to_numpy()[te], out[te]), 3))
        print(name, "AUC squeeze>=50% by year 2022-26:", aucs)
        risk[name] = (out, ret)

    Y = (2022, 2023, 2024, 2025, 2026)
    for name, (r, er) in risk.items():
        x = d.with_columns(risk=pl.Series(r), er=pl.Series(er)).filter(pl.col("year") >= 2022)
        for cap in (1.0, 0.3, 0.2, 0.1):
            c = x.filter((pl.col("ret_4h") >= 0.4) & (pl.col("risk") <= cap)).select(
                "row", side=pl.lit(-1), prio="ret_4h"
            )
            for lab, rule in (
                ("iso1x", Rule(hold_min=4320, slots=10, cost=0.003, sl=1.0, stop_slip=0.0)),
                ("cross", Rule(hold_min=4320, slots=10, cost=0.003, loss_floor=-20.0)),
            ):
                b = simulate_yearly(c, rule, perp_funding=True, source="perp", years=Y)
                print(
                    f"{name:7s} r4>=0.4 risk<={cap:.1f} {lab}: {b.summary['bp']:+5.0f}bp {fmt_years(b, Y)}"  # noqa: E501
                )
        for q in (0.0, 0.05, 0.10):
            c = x.filter(pl.col("er") >= q).select("row", side=pl.lit(-1), prio="er")
            b = simulate_yearly(
                c,
                Rule(hold_min=4320, slots=10, cost=0.003, sl=1.0),
                perp_funding=True,
                source="perp",
                years=Y,
            )
            print(f"{name:7s} exp.ret>={q:.2f} iso1x: {b.summary['bp']:+5.0f}bp {fmt_years(b, Y)}")


if __name__ == "__main__":
    main()
