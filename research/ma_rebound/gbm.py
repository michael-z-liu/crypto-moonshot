"""M4: the gradient-boosted benchmark (docs/plans/ma-rebound.md, "M4 spec").

    uv run --group research python gbm.py run                 # 5 folds, seed 0
    uv run --group research python gbm.py seeds               # seed check on fold 2022

scikit-learn `HistGradientBoostingRegressor` on hand-made features of the base CNN's inputs
(`features`): the scalars; returns over several lookbacks, ranges and the drawdown from the
window high on the fine (5m) and coarse (MA timeframe) windows, all in units of u; the volume
relative to the window median over the last candles / bars; the taker-buy share; the level's
slope; BTC's returns on both windows. Same rows, target (`ret`, the E1 gross return / u clipped
to [-5, 5]), folds and percentile rule as `train.py`'s base. The model has no Huber loss, so it
uses squared error on the clipped target. Early stopping: the number of boosting iterations
(at most 1000) with the best Spearman correlation on the fold's validation rows, checked every
10 iterations.

Output in `cache/m4/scores/gbm/`: `seed0.parquet`, `mean.parquet` (the same table: one seed),
`folds.npz`, `run.json`, `log.csv` (folds 2022-2025). Fold 2026 validates on 2025 H2: nothing
of it is printed; its log goes to `results/sealed/m4/`.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
import polars as pl
import train as tr
from common import SEALED
from sklearn.ensemble import HistGradientBoostingRegressor

type F32 = npt.NDArray[np.float32]
type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]

MAX_ITER = 1000
EVAL_EVERY = 10
FINE_K = (3, 12, 48, 95)
COARSE_K = (2, 8, 32, 127)


def _ret(c: F32, valid: F32, k: int) -> F32:
    """c[-1] - c[-1-k] (NaN where position -1-k is padding)."""
    return np.where(valid[:, -1 - k] > 0, c[:, -1] - c[:, -1 - k], np.nan).astype(np.float32)


def _masked(x: F32, valid: F32, fill: float) -> F32:
    return np.where(valid > 0, x, fill).astype(np.float32)


def features(data: tr.Data) -> tuple[F32, list[str]]:
    """Hand-made features of every family row (data: the base variant's `Data`)."""
    fc, cc = data.fine_ch, data.coarse_ch
    n = data.meta.height
    out: list[F32] = []
    names: list[str] = list(data.scal_names)
    for s in range(0, n, 50_000):
        idx = np.arange(s, min(n, s + 50_000))
        xf16, xc16 = data.raw(idx)
        xf = np.asarray(xf16, dtype=np.float32)
        xc = np.asarray(xc16, dtype=np.float32)
        f = {c: xf[:, i] for i, c in enumerate(fc)}
        c_ = {c: xc[:, i] for i, c in enumerate(cc)}
        cols: list[F32] = [data.scal[idx]]
        new: list[str] = []

        def add(name: str, x: F32, cols: list[F32] = cols, new: list[str] = new) -> None:
            cols.append(x[:, None].astype(np.float32))
            new.append(name)

        fm, cm, lm, bm = f["mask"], c_["mask"], c_["level_mask"], c_["btc_mask"]
        ones = np.ones_like(fm)
        for k in FINE_K:
            add(f"f_ret{k}", _ret(f["close"], fm, k))
            add(f"f_btc_ret{k}", _ret(f["btc_close"], ones, k))
        for k in (12, 48, 96):
            hi = _masked(f["high"][:, -k:], fm[:, -k:], -np.inf).max(1)
            lo = _masked(f["low"][:, -k:], fm[:, -k:], np.inf).min(1)
            add(f"f_range{k}", hi - lo)
        add("f_drawdown", f["close"][:, -1] - _masked(f["high"], fm, -np.inf).max(1))
        for k in (3, 12):
            add(f"f_vol{k}", f["logqv"][:, -k:].mean(1))
            add(f"f_btc_vol{k}", f["btc_logqv"][:, -k:].mean(1))
        for k in (12, 96):
            w = fm[:, -k:]
            add(f"f_taker{k}", (f["taker"][:, -k:] * w).sum(1) / np.maximum(w.sum(1), 1))
        for k in COARSE_K:
            add(f"c_ret{k}", _ret(c_["close"], cm, k))
            add(f"c_btc_ret{k}", _ret(c_["btc_close"], bm, k))
        for k in (8, 32, 128):
            hi = _masked(c_["high"][:, -k:], cm[:, -k:], -np.inf).max(1)
            lo = _masked(c_["low"][:, -k:], cm[:, -k:], np.inf).min(1)
            add(f"c_range{k}", hi - lo)
        add("c_drawdown", c_["close"][:, -1] - _masked(c_["high"], cm, -np.inf).max(1))
        for k in (8, 32):
            add(f"c_level_slope{k}", _ret(c_["level"], lm, k))
        for k in (1, 4):
            add(f"c_vol{k}", c_["logqv"][:, -k:].mean(1))
        add("c_bars", cm.sum(1))
        out.append(np.concatenate(cols, 1))
        if s == 0:
            names += new
    x = np.concatenate(out)
    x[~np.isfinite(x)] = np.nan
    return x, names


def fit_fold(
    x: F32, y: F32, fold: tr.Fold, seed: int
) -> tuple[HistGradientBoostingRegressor, int, list[dict[str, object]]]:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=MAX_ITER,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(x[fold.train], y[fold.train])
    log: list[dict[str, object]] = []
    best, best_it = -np.inf, 1
    for it, pred in enumerate(model.staged_predict(x[fold.valid]), 1):
        if it % EVAL_EVERY:
            continue
        rho = tr.spearman(pred.astype(np.float32), y[fold.valid])
        log.append({"year": fold.year, "seed": seed, "iteration": it, "valid_spearman": rho})
        if rho > best:
            best, best_it = rho, it
    return model, best_it, log


def predict_at(model: HistGradientBoostingRegressor, x: F32, it: int) -> F32:
    for i, pred in enumerate(model.staged_predict(x), 1):
        if i == it:
            return pred.astype(np.float32)
    raise ValueError(it)


def run(out: str = "gbm", years: tuple[int, ...] = tr.YEARS, seed: int = 0) -> str:
    t_start = time.perf_counter()
    run_id = f"{out}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    dest = tr.SCORES / out
    dest.mkdir(parents=True, exist_ok=True)
    data = tr.Data(tr.BASE)
    x, names = features(data)
    y = data.t.ret
    print(f"features: {x.shape[1]} ({time.perf_counter() - t_start:.0f}s)", flush=True)
    n = data.meta.height
    raw = np.full(n, np.nan, dtype=np.float32)
    pct = np.full(n, np.nan)
    fold_of = np.full(n, -1, dtype=np.int32)
    logs: list[dict[str, object]] = []
    sealed_logs: list[dict[str, object]] = []
    fold_sets: dict[str, I64] = {}
    summary: list[dict[str, object]] = []
    for fold in tr.folds(data, years):
        t0 = time.perf_counter()
        sealed = fold.year >= tr.SEALED_YEAR
        key = f"{fold.year}_{fold.group}"
        for part in ("train", "valid", "test"):
            fold_sets[f"{key}_{part}"] = getattr(fold, part) + data.offset
        model, it, log = fit_fold(x, y, fold, seed)
        (sealed_logs if sealed else logs).extend(log)
        r_tr = predict_at(model, x[fold.train], it)
        r_te = predict_at(model, x[fold.test], it)
        raw[fold.test] = r_te
        pct[fold.test] = tr.percentile(r_tr, r_te)
        fold_of[fold.test] = fold.year
        mins = (time.perf_counter() - t0) / 60
        if sealed:
            print(f"fold {fold.year} done", flush=True)
        else:
            best = max(float(str(r["valid_spearman"])) for r in log)
            summary.append({"year": fold.year, "best_iteration": it, "valid_spearman": best})
            print(
                f"fold {fold.year}: train {len(fold.train)}, valid {len(fold.valid)},"
                f" test {len(fold.test)}; best iteration {it}, valid spearman {best:+.4f},"
                f" {mins:.1f} min",
                flush=True,
            )
    ok = np.isfinite(pct)
    table = pl.DataFrame(
        {
            "cid": data.meta["cid"].filter(ok),
            "score": pct[ok],
            "raw": raw[ok],
            "fold": fold_of[ok],
        }
    )
    table.write_parquet(dest / f"seed{seed}.parquet")
    table.write_parquet(dest / "mean.parquet")
    np.savez(dest / "folds.npz", **fold_sets)  # type: ignore[arg-type]
    if logs:
        pl.DataFrame(logs).write_csv(dest / "log.csv")
    if sealed_logs:
        (SEALED / "m4").mkdir(parents=True, exist_ok=True)
        pl.DataFrame(sealed_logs).write_csv(SEALED / "m4" / f"{run_id}-fold2026.csv")
    meta = {
        "run_id": run_id,
        "variant": "gbm",
        "config": json.dumps(
            {"family": "limit", "target": "ret", "pooled": True, "seed": seed, "model": "hgb"}
        ),
        "years": list(years),
        "max_train": None,
        "features": names,
        "folds": summary,
        "minutes": (time.perf_counter() - t_start) / 60,
    }
    (dest / "run.json").write_text(json.dumps(meta, indent=1))
    print(f"{out}: run {run_id}, {meta['minutes']:.1f} min", flush=True)
    return run_id


def seed_check() -> None:
    """Fold 2022 with seeds 0, 1, 2: how much the test predictions differ."""
    data = tr.Data(tr.BASE)
    x, _ = features(data)
    y = data.t.ret
    fold = tr.folds(data, (2022,))[0]
    preds = []
    for seed in (0, 1, 2):
        model, it, _ = fit_fold(x, y, fold, seed)
        preds.append(predict_at(model, x[fold.test], it))
        print(f"seed {seed}: best iteration {it}", flush=True)
    for s in (1, 2):
        d = np.abs(preds[s] - preds[0])
        print(f"seed {s} vs 0: max |diff| {d.max():.3g}, rows differing {(d > 0).sum()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["run", "seeds"])
    a = p.parse_args()
    if a.cmd == "run":
        run()
    else:
        seed_check()


if __name__ == "__main__":
    main()
