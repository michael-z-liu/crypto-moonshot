"""S3 of the CNN robustness study (docs/plans/cnn-robustness.md): training and ensembles.

    cd research/volume_spike
    export MOONSHOT_UNIVERSE=all
    uv run --group research python s3_train.py check                  # base 2022 reproduces S0
    uv run --group research python s3_train.py train allch_y1h ...    # one GPU training each
    uv run --group research python s3_train.py stack                  # stack_gbm, gbm_scalars
    uv run --group research python s3_train.py score allch_y1h ... stack_gbm ens_s3 ens_top
    uv run --group research python s3_train.py table                  # S3 table (2022-24)

Variants (fixed before any result, see the plan's S3):

- `allch_y1h`: all 13 channels, y1h target, MSE (the control for `allch_xs`).
- `allch_xs`: as `allch_y1h`, loss `xs` (target minus the same day's mean). The day's mean
  only enters the training target; scoring sees the inputs only, so it is causal.
- `allch_cost`: as `allch_y1h`, loss `asym`: target y1h - 0.3% (the round trip), squared error
  with over-forecasts weighted 2x.
- `path_all_h6`: path_all retrained every 6 months (`m12_cnn_wf.folds_of("half")`).
- `path_all5`: path_all with 5 seeds (m12's existing config).
- `stack_gbm`: a gradient-boosted stacker on the CNNs' walk-forward pct scores + causal scalars,
  target y1h; for test year Y it trains on 2022..Y-1 (so it starts in 2023). `gbm_scalars`: the
  same without the CNN scores.
- `ens_s3`, `ens_top`: mean pct ensembles, threshold matched to path_all's 2022-24 candidate
  count at q 0.98 (no outcomes used).

Scores go to `cache_all/wf/<name>.npz` (key `pct`); ledger variants are `S3:<name>`.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import m12_cnn_wf as m12
import numpy as np
import polars as pl
import scorecard as sc
from common import CACHE

S3_CNN = {
    "allch_y1h": m12.Cfg(channels=m12.VARIANTS["all_ch"].channels, note="all 13 channels, y1h"),
    "allch_xs": m12.Cfg(
        channels=m12.VARIANTS["all_ch"].channels,
        loss="xs",
        note="all 13 channels, y1h minus day mean",
    ),
    "allch_cost": m12.Cfg(
        channels=m12.VARIANTS["all_ch"].channels,
        loss="asym",
        note="all 13 channels, y1h - 0.3%, over-forecasts weighted 2x",
    ),
    "path_all_h6": replace(m12.VARIANTS["path_all"], note="path_all, retrained every 6 months"),
}
SCHEDULE = {"path_all_h6": "half"}
m12.VARIANTS |= S3_CNN
CNN_ORDER = ("allch_y1h", "allch_xs", "allch_cost", "path_all5", "path_all_h6")


def train(names: list[str]) -> None:
    data = m12.Data()
    for name in names:
        t0 = time.time()
        m12.run(name, data, SCHEDULE.get(name, "yearly"))
        print(f"{name} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def check() -> None:
    """Retrain base's 2022 fold with the current m12 and compare it with S0's base.npz."""
    data = m12.Data()
    t0 = time.time()
    m12.run("base", data, years=(2022,), out_name="s3_check_base2022")
    new = np.load(m12.WF / "s3_check_base2022.npz")
    old = np.load(m12.WF / "base.npz")
    rows = data.meta["year"].to_numpy() == 2022
    for key in ("raw", "pct"):
        a, b = new[key][rows], old[key][rows]
        print(
            f"base 2022 {key}: identical={np.array_equal(a, b)}"
            f" max abs diff={np.nanmax(np.abs(a - b)):.3g}",
            flush=True,
        )
    print(f"check done in {(time.time() - t0) / 60:.1f} min", flush=True)


def source_of(name: str) -> str:
    cfg = asdict(m12.VARIANTS[name])
    cfg.pop("note")
    sched = SCHEDULE.get(name, "yearly")
    return f"m12_cnn_wf {name} {cfg} folds {sched} (cache_all/wf/{name}.npz)"


# ----------------------------------------------------------------------------- stacker

S0_CNN = ("base", "market", "path_all")
GBM_SCALARS = (
    "qv_base", "listed_days", "hour", "ret_4h", "ret_24h", "btc_vol_mult", "btc_ret_4h",
    "vol_mult",
)  # fmt: skip
GBM_PARAMS: dict[str, object] = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 1000,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}  # fixed before any result; no tuning
GBM_YEARS = (2023, 2024, 2025, 2026)
PURGE = timedelta(hours=2)  # drop training rows whose 1h outcome reaches into the test year


def stack_members() -> list[str]:
    """CNN scores available to the stacker: S0's three and every finished S3 CNN."""
    return [n for n in (*S0_CNN, *CNN_ORDER) if (m12.WF / f"{n}.npz").exists()]


def stack(with_cnn: bool) -> None:
    """Walk-forward GBM: for test year Y, fit on [2022-01-01, Y-01-01 - PURGE), score Y.

    Scores become percentiles of the fold's own (in-sample) training predictions, as m12's."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    name = "stack_gbm" if with_cnn else "gbm_scalars"
    meta = pl.read_parquet(CACHE / "seq2_meta.parquet")
    cols = [
        np.log(meta["qv_base"].to_numpy().clip(1, None)) if c == "qv_base" else meta[c].to_numpy()
        for c in GBM_SCALARS
    ]
    members = stack_members() if with_cnn else []
    cols += [sc.wf_scores(n) for n in members]
    x = np.stack([np.asarray(c, dtype=np.float64) for c in cols], 1)
    y = np.clip(meta["y1h"].to_numpy(), -0.2, 0.2)
    t = meta["open_time"]
    year = t.dt.year().to_numpy()
    pct = np.full(len(y), np.nan, dtype=np.float32)
    raw = np.full(len(y), np.nan, dtype=np.float32)
    t0 = time.time()
    for Y in GBM_YEARS:
        end = datetime(Y, 1, 1, tzinfo=UTC) - PURGE
        tr = np.flatnonzero(((t >= datetime(2022, 1, 1, tzinfo=UTC)) & (t < end)).to_numpy())
        te = np.flatnonzero(year == Y)
        model = HistGradientBoostingRegressor(**GBM_PARAMS)  # type: ignore[arg-type]
        model.fit(x[tr], y[tr])
        ref = np.sort(model.predict(x[tr[:: max(1, len(tr) // 50_000)]]))
        s = model.predict(x[te])
        raw[te] = s
        pct[te] = np.searchsorted(ref, s) / len(ref)
        print(f"{name} {Y}: {len(tr)} train rows, {len(te)} scored, {time.time() - t0:.0f}s")
    np.savez(m12.WF / f"{name}.npz", pct=pct, raw=raw, members=np.array(members))
    print(f"{name}: features {[*GBM_SCALARS, *members]}", flush=True)


def gbm_source(name: str) -> str:
    with np.load(m12.WF / f"{name}.npz") as f:
        members = [str(m) for m in f["members"]]
    return (
        f"s3_train {name}: HistGradientBoostingRegressor {GBM_PARAMS} on y1h, features"
        f" {[*GBM_SCALARS, *members]} (m12 pct), fit 2022..Y-1 (2h purge), score Y in 2023-26;"
        f" pct of in-sample training predictions; 2023-24 only (2022 unscored)"
        f" (cache_all/wf/{name}.npz)"
    )


# ----------------------------------------------------------------------------- ensembles

ENS_S3 = (*CNN_ORDER, *S0_CNN)
PATH_ALL_Q = 0.98


def match_q(score: np.ndarray, target: int) -> float:
    """The threshold giving `target` candidate rows in 2022-24 (no outcomes used)."""
    m = sc.meta()
    v = score[sc.in_range(m, sc.SELECT_END) & np.isfinite(score)]
    return float(np.sort(v)[::-1][target - 1])


def path_all_count() -> int:
    m = sc.meta()
    p = sc.wf_scores("path_all")
    with np.errstate(invalid="ignore"):
        return int((sc.in_range(m, sc.SELECT_END) & (p >= PATH_ALL_Q)).sum())


def cnn_ranking() -> pl.DataFrame:
    """Qualified CNN variants of S0 and S3, by the plan's ranking key (2022-24 only)."""
    names = [*S0_CNN, *(f"S3:{n}" for n in CNN_ORDER)]
    led = pl.read_csv(sc.LEDGER, infer_schema_length=None)
    rule = sc.Rule().describe()
    return (
        led.filter(pl.col("variant").is_in(names) & (pl.col("rule") == rule))
        .filter(pl.col("stage").is_in(["S0", "S3"]))
        .sort("timestamp")
        .unique("variant", keep="last")
        .filter(pl.col("qualifies"))
        .sort(["mean_bp", "worst_bp"], descending=True)
    )


def ensemble(name: str) -> tuple[np.ndarray, sc.Rule, str]:
    if name == "ens_s3":
        members = list(ENS_S3)
    else:
        top = cnn_ranking().head(3)["variant"].to_list()
        members = [v.removeprefix("S3:") for v in top]
    ens = np.mean(np.stack([sc.wf_scores(n) for n in members]), 0)  # NaN unless all present
    target = path_all_count()
    q = match_q(ens, target)
    src = (
        f"mean pct of m12 {', '.join(members)} (cache_all/wf); threshold {q:.6f} matches"
        f" path_all's {target} candidates 2022-24 at q {PATH_ALL_Q}"
    )
    return ens, sc.Rule(q=q), src


# ----------------------------------------------------------------------------- scoring


def card_path(name: str) -> Path:
    return sc.OUT / "cards" / f"S3_{name}.txt"


def score(names: list[str]) -> None:
    for name in names:
        if name.startswith("ens_"):
            s, rule, src = ensemble(name)
        elif name in ("stack_gbm", "gbm_scalars"):
            s, rule, src = sc.wf_scores(name), sc.Rule(), gbm_source(name)
        else:
            s, rule, src = sc.wf_scores(name), sc.Rule(), source_of(name)
        card = sc.scorecard(f"S3:{name}", s, rule, stage="S3", source=src)
        text = sc.show(card) + f"\n  source: {src}"
        print(text, flush=True)
        card_path(name).parent.mkdir(parents=True, exist_ok=True)
        card_path(name).write_text(text + "\n")


# ----------------------------------------------------------------------------- table

TABLE = (*CNN_ORDER, "stack_gbm", "gbm_scalars", "ens_s3", "ens_top")


def fragile(score: np.ndarray, rule: sc.Rule) -> tuple[bool, dict[int, float]]:
    """2022-24 bp per year without that year's single best trade; fragile if any is < 0.

    The scorecard's own book (to the data end); only 2022-24 entries are read."""
    trades = sc.run_book(sc.signals(score, rule), rule).trades
    yr = trades.with_columns(year=pl.col("entry_time").dt.year()).filter(
        pl.col("year").is_in(sc.SELECT_YEARS)
    )
    out = {}
    for y in sc.SELECT_YEARS:
        r = np.sort(yr.filter(pl.col("year") == y)["net_return"].to_numpy())[:-1]
        out[y] = float(r.mean()) * 1e4 if len(r) else float("nan")
    return any(v < 0 for v in out.values()), out


def scores_rule(name: str) -> tuple[np.ndarray, sc.Rule]:
    if name.startswith("ens_"):
        s, rule, _ = ensemble(name)
        return s, rule
    return sc.wf_scores(name), sc.Rule()


def table() -> None:
    """The S3 table (2022-24 only) from the ledger, plus the fragile flag."""
    led = pl.read_csv(sc.LEDGER, infer_schema_length=None)
    rows = []
    for name in (*S0_CNN, "ens3", *TABLE):
        variant = name if name in (*S0_CNN, "ens3") else f"S3:{name}"
        r = led.filter(pl.col("variant") == variant).sort("timestamp").tail(1)
        if r.height == 0:
            continue
        d = r.row(0, named=True)
        if name == "ens3":
            s = np.mean(np.stack([sc.wf_scores(n) for n in S0_CNN]), 0)
            fr, ex = fragile(s, sc.Rule())
        else:
            fr, ex = fragile(*scores_rule(name))
        bp = [d[f"bp_{y}"] for y in sc.SELECT_YEARS]
        rows.append(
            {
                "variant": variant,
                **{f"bp_{y}": d[f"bp_{y}"] for y in sc.SELECT_YEARS},
                "mean_bp": d["mean_bp"],
                "mean_2324": float(np.mean(bp[1:])),
                "cost06_bp": d["cost06_bp"],
                "late_bp": d["late_bp"],
                "ex5_bp": d["ex5_bp"],
                "ci_lo": d["ci_lo_bp"],
                "ci_hi": d["ci_hi_bp"],
                **{f"trades_{y}": d[f"trades_{y}"] for y in sc.SELECT_YEARS},
                "n_signals": d["n_signals_2224"],
                "qualifies": d["qualifies"],
                "fragile": fr,
                **{f"ex1_{y}": round(v, 1) for y, v in ex.items()},
            }
        )
    df = pl.DataFrame(rows)
    df.write_csv(sc.OUT / "S3_table.csv")
    with pl.Config(tbl_rows=40, tbl_cols=30, tbl_width_chars=300, float_precision=1):
        print(df)


def timings() -> None:
    """Training minutes and epochs kept (best validation epoch + 1) per (fold, seed)."""
    rows = []
    for name in ("base", "market", "path_all", *CNN_ORDER):
        f = m12.WF / f"{name}_log.csv"
        if not f.exists():
            continue
        log = pl.read_csv(f)
        fold = "fold" if "fold" in log.columns else "year"
        best = log.sort("valid_corr", descending=True).group_by(fold, "seed").first()
        run = log.group_by(fold, "seed").agg(pl.len())
        kept = (best["epoch"] + 1).to_numpy()
        rows.append(
            {
                "variant": name,
                "fits": best.height,
                "epochs_run_mean": round(float(run["len"].mean()), 2),  # type: ignore[arg-type]
                "epochs_kept_mean": round(float(kept.mean()), 2),
                "epochs_kept_min": int(kept.min()),
                "epochs_kept_max": int(kept.max()),
                "epoch_min": round(float(log["secs"].sum()) / 60, 1),
            }
        )
    df = pl.DataFrame(rows)
    df.write_csv(sc.OUT / "S3_timings.csv")
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=200):
        print(df)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "train":
        train(sys.argv[2:])
    elif cmd == "check":
        check()
    elif cmd == "stack":
        stack(with_cnn=True)
        stack(with_cnn=False)
    elif cmd == "score":
        score(sys.argv[2:])
    elif cmd == "table":
        table()
    elif cmd == "timings":
        timings()
    else:
        raise SystemExit(__doc__)
