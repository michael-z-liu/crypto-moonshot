"""M7: a 1D convolutional network on the raw candles before each spike, predicting 1h net return.

Protocol (docs/plans/volume-spike-research.md, M7): fit on 2020-2023, choose on 2024, then look
at 2025-2026 once. Two stages so the test period stays closed while choices are made:

    uv run python research/volume_spike/m11_cnn.py train    # fits, prints train/valid only
    uv run python research/volume_spike/m11_cnn.py report   # the full table, including test

Input sequences come from build_seq.py. Variant "coin" sees the coin's own candles (OHLC, volume,
taker-buy share) and its liquidity; "coin_btc" also sees BTC's close and volume.
"""

from __future__ import annotations

import sys
import time
from datetime import date

import numpy as np
import polars as pl
import torch
from common import CACHE, RESULTS
from evalx import load_events
from m3_rules import non_overlapping
from m10_model import FEATURES
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn

VARIANTS = {"coin": [0, 1, 2, 3, 4, 5], "coin_btc": [0, 1, 2, 3, 4, 5, 6, 7]}
SEEDS = (0, 1, 2)
EPOCHS = 8
BATCH = 2048
HOLD = 12
QUANTILES = (0.99, 0.995, 0.998, 0.999)
CASCADE = date(2025, 10, 10)  # the liquidation cascade that dominates the GBM's test result
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        def block(c_in: int, c_out: int, k: int) -> list[nn.Module]:
            return [nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.GELU()]

        self.conv = nn.Sequential(
            *block(channels, 32, 5), *block(32, 32, 5), nn.MaxPool1d(2),  # 48 steps
            *block(32, 64, 5), nn.MaxPool1d(2),  # 24 steps
            *block(64, 64, 3), *block(64, 64, 3),
        )  # fmt: skip
        # average and max over time, the last step (the spike candle), and the liquidity scalar
        self.head = nn.Sequential(
            nn.Linear(64 * 3 + 1, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        z = torch.cat([h.mean(2), h.amax(2), h[:, :, -1], s], 1)
        return self.head(z).squeeze(1)


class Data:
    """All sequences in memory (float16), standardised per channel on the device."""

    def __init__(self) -> None:
        self.meta = pl.read_parquet(CACHE / "seq_meta.parquet")
        self.seq = np.load(CACHE / "seq.npy")
        self.split = self.meta["split"].to_numpy()
        self.y = np.clip(self.meta["y"].to_numpy(), -0.2, 0.2) * 100  # percent
        q = np.log(self.meta["qv_base"].fill_null(1.0).to_numpy().clip(1.0, None))
        tr = self.split == "train"
        self.s = ((q - q[tr].mean()) / q[tr].std()).astype(np.float32)[:, None]
        sample = self.seq[np.flatnonzero(tr)[::20]].astype(np.float32)
        self.mu = torch.tensor(sample.mean((0, 2)), device=DEVICE)[None, :, None]
        self.sd = torch.tensor(sample.std((0, 2)), device=DEVICE)[None, :, None]

    def batch(self, idx: np.ndarray, ch: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.seq[idx]).to(DEVICE).float()
        x = ((x - self.mu) / self.sd).clamp(-10, 10)[:, ch]
        return x, torch.from_numpy(self.s[idx]).to(DEVICE)


def predict(net: Net, data: Data, rows: np.ndarray, ch: list[int]) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(rows), 16384):
            x, s = data.batch(rows[k : k + 16384], ch)
            out.append(net(x, s).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def top_mean(pred: np.ndarray, y: np.ndarray, q: float) -> float:
    """Mean target (percent) of the rows above the q quantile of pred: a quick validation score."""
    return float(y[pred >= np.quantile(pred, q)].mean())


def fit(data: Data, ch: list[int], seed: int) -> tuple[Net, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(len(ch)).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tr = np.flatnonzero(data.split == "train")
    va = np.flatnonzero(data.split == "valid")
    y = torch.from_numpy(data.y.astype(np.float32))
    best, best_state, log = -np.inf, None, []
    for epoch in range(EPOCHS):
        net.train()
        t0 = time.time()
        perm = rng.permutation(tr)
        loss_sum = 0.0
        for k in range(0, len(perm), BATCH):
            b = np.sort(perm[k : k + BATCH])
            x, s = data.batch(b, ch)
            loss = nn.functional.mse_loss(net(x, s), y[b].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(b)
        pv = predict(net, data, va, ch)
        corr = float(np.corrcoef(pv, data.y[va])[0, 1])
        row = {"epoch": epoch, "train_mse": loss_sum / len(tr), "valid_corr": corr,
               "valid_top1pct": top_mean(pv, data.y[va], 0.99),
               "valid_top02pct": top_mean(pv, data.y[va], 0.998),
               "secs": time.time() - t0}  # fmt: skip
        log.append(row)
        print(seed, {k: round(v, 4) for k, v in row.items()}, flush=True)
        if corr > best:  # epochs are chosen on validation correlation
            best = corr
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    assert best_state is not None
    net.load_state_dict(best_state)
    return net, log


def train() -> None:
    data = Data()
    rows = np.arange(len(data.y))
    logs = []
    for name, ch in VARIANTS.items():
        preds = []
        for seed in SEEDS:
            net, log = fit(data, ch, seed)
            logs += [{"variant": name, "seed": seed, **r} for r in log]
            p = predict(net, data, rows, ch)
            preds.append((p - p[data.split == "train"].mean()) / p[data.split == "train"].std())
        np.save(CACHE / f"cnn_pred_{name}.npy", np.mean(preds, 0))
    pl.DataFrame(logs).write_csv(RESULTS / "m11_cnn_training.csv")
    report(("train", "valid"))


def gbm_pred() -> np.ndarray:
    """The M3 gradient-boosting model (m10_model.py), refit identically, on the same rows."""
    path = CACHE / "gbm_pred.npy"
    if path.exists():
        return np.load(path)
    ev, _ = load_events()
    sp = ev.filter(pl.col("is_spike"))
    meta = pl.read_parquet(CACHE / "seq_meta.parquet")
    X = sp.select(FEATURES).cast(pl.Float64).fill_nan(None).to_numpy()
    y = np.clip(meta["y"].to_numpy(), -0.2, 0.2)
    tr = np.flatnonzero(meta["split"].to_numpy() == "train")
    sub = np.random.default_rng(0).choice(tr, size=min(400_000, len(tr)), replace=False)
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
        l2_regularization=1.0, random_state=0,
    )  # fmt: skip
    model.fit(X[sub], y[sub])
    p = model.predict(X)
    np.save(path, p)
    return p


def trade_stats(d: pl.DataFrame) -> dict[str, float]:
    days = d.group_by(pl.col("open_time").dt.date().alias("day")).agg(pl.col("y").mean())
    ex = d.filter(pl.col("open_time").dt.date() != CASCADE)
    ex_days = days.filter(pl.col("day") != CASCADE)
    return {
        "trades": d.height,
        "mean_bp": d["y"].mean() * 1e4 if d.height else np.nan,
        "day_bp": days["y"].mean() * 1e4 if days.height else np.nan,
        "win_days": (days["y"] > 0).mean() if days.height else np.nan,
        "days": days.height,
        "mean_bp_ex1010": ex["y"].mean() * 1e4 if ex.height else np.nan,
        "day_bp_ex1010": ex_days["y"].mean() * 1e4 if ex_days.height else np.nan,
    }


def selection_table(
    meta: pl.DataFrame, models: dict[str, np.ndarray], splits: tuple[str, ...]
) -> pl.DataFrame:
    rows = []
    for name, pred in models.items():
        sp = meta.with_columns(pred=pl.Series(pred))
        train_pred = pred[meta["split"].to_numpy() == "train"]
        for q in QUANTILES:
            th = float(np.quantile(train_pred, q))
            for s in splits:
                d = sp.filter((pl.col("split") == s) & (pl.col("pred") >= th))
                d = d.filter(
                    pl.Series(
                        non_overlapping(d["symbol"].to_numpy(), d["open_time"].to_numpy(), HOLD)
                    )
                )
                rows.append({"model": name, "quantile": q, "split": s, **trade_stats(d)})
    return pl.DataFrame(rows)


def f1_filter_table(
    meta: pl.DataFrame, models: dict[str, np.ndarray], splits: tuple[str, ...]
) -> pl.DataFrame:
    """F1 trades (the pre-registered flush rule) split by the model score: enter or skip?"""
    f1 = (
        (meta["ret_4h"] <= -0.08) & (meta["qv_base"] >= 5_000) & (meta["btc_vol_mult"] >= 6)
    ).fill_null(False)
    keep = f1.to_numpy()
    keep[keep] = non_overlapping(
        meta.filter(f1)["symbol"].to_numpy(), meta.filter(f1)["open_time"].to_numpy(), HOLD
    )
    rows = []
    for name, pred in models.items():
        d = meta.with_columns(pred=pl.Series(pred)).filter(pl.Series(keep))
        cut = float(d.filter(pl.col("split") == "train")["pred"].median())
        for s in splits:
            for half, cond in (
                ("all F1", pl.lit(True)),
                ("enter (score above)", pl.col("pred") >= cut),
                ("skip (score below)", pl.col("pred") < cut),
            ):
                rows.append({"model": name, "split": s, "group": half,
                             **trade_stats(d.filter((pl.col("split") == s) & cond))})  # fmt: skip
    return pl.DataFrame(rows)


def report(splits: tuple[str, ...] = ("train", "valid", "test")) -> None:
    meta = pl.read_parquet(CACHE / "seq_meta.parquet")
    models = {f"cnn_{v}": np.load(CACHE / f"cnn_pred_{v}.npy") for v in VARIANTS}
    models["gbm"] = gbm_pred()
    sel = selection_table(meta, models, splits)
    f1 = f1_filter_table(meta, models, splits)
    tag = "" if "test" in splits else "_trainvalid"
    sel.write_csv(RESULTS / f"m11_cnn_selection{tag}.csv")
    f1.write_csv(RESULTS / f"m11_cnn_f1_filter{tag}.csv")
    with pl.Config(tbl_rows=100, tbl_cols=12, tbl_width_chars=220, float_precision=1):
        print(sel)
        print(f1)


def portfolio() -> None:
    """Post-hoc (after the test): the chosen CNN, the GBM and F2 in the 5-slot portfolio sim."""
    from portfolio import Exit, simulate

    ev, fwd = load_events()
    meta = pl.read_parquet(CACHE / "seq_meta.parquet")
    rows_ = meta["event"].to_numpy()
    tr = meta["split"].to_numpy() == "train"
    out = []
    picks: dict[str, np.ndarray] = {}
    for name, file in (("cnn_coin_btc", "cnn_pred_coin_btc.npy"), ("gbm", "gbm_pred.npy")):
        p = np.load(CACHE / file)
        keep = p >= np.quantile(p[tr], 0.995)
        picks[name] = rows_[keep]
        ev_p = np.full(ev.height, np.nan)
        ev_p[rows_] = p
        ev = ev.with_columns(pl.Series(f"neg_{name}", -ev_p))
    f2 = (
        pl.col("is_spike") & (pl.col("ret_4h") <= -0.10) & (pl.col("qv_base") >= 20_000)
        & (pl.col("btc_vol_mult") >= 6)
    )  # fmt: skip
    picks["F2 rule"] = np.flatnonzero(ev.select(f2.fill_null(False).alias("m"))["m"].to_numpy())
    for name, idx in picks.items():
        idx = np.sort(idx)
        prio = f"neg_{name}" if name != "F2 rule" else "ret_4h"
        for s in ("train", "valid", "test"):
            k = idx[(ev["split"].to_numpy()[idx] == s)]
            res = simulate(ev[k], np.asarray(fwd[k]), Exit(bars=HOLD), slots=5, priority=prio)
            st = res.stats()
            yearly = res.trades.group_by(pl.col("entry_time").dt.year().alias("y")).agg(
                (pl.col("ret").mean() * 1e4).alias("bp")
            )  # fmt: skip
            by_year = "; ".join(
                f"{r['y']}: {r['bp']:+.0f}" for r in yearly.sort("y").iter_rows(named=True)
            )
            out.append({"model": name, "split": s, **st, "by_year": by_year})
    df = pl.DataFrame(out)
    df.write_csv(RESULTS / "m11_cnn_portfolio.csv")
    with pl.Config(
        tbl_rows=20, tbl_cols=14, tbl_width_chars=260, float_precision=2, fmt_str_lengths=60
    ):
        print(df.drop("sharpe", "days_in_market"))


def stress() -> None:
    """Post-hoc stress tests of the chosen CNN portfolio on the test period."""
    from portfolio import Exit, simulate

    ev, fwd = load_events()
    meta = pl.read_parquet(CACHE / "seq_meta.parquet")
    rows_ = meta["event"].to_numpy()
    tr = meta["split"].to_numpy() == "train"
    out = []
    for variant in ("coin_btc", "coin"):
        p = np.load(CACHE / f"cnn_pred_{variant}.npy")
        ev_p = np.full(ev.height, np.nan)
        ev_p[rows_] = p
        e2 = ev.with_columns(neg=pl.Series(-ev_p))
        idx = np.sort(rows_[p >= np.quantile(p[tr], 0.995)])
        test = idx[e2["split"].to_numpy()[idx] == "test"]
        day = e2["open_time"].dt.date().to_numpy()[test]
        f = np.asarray(fwd[test])
        late = (1 + f[:, :, 1:]) / (1 + f[:, 2:3, :1]) - 1  # enter at the next candle's close
        calm = day != CASCADE
        cases = {
            "base (5 slots, 0.3%)": (test, f, 5, 0.003),
            "without 2025-10-10": (test[calm], f[calm], 5, 0.003),
            "3 slots": (test, f, 3, 0.003),
            "10 slots": (test, f, 10, 0.003),
            "cost 0.6%": (test, f, 5, 0.006),
            "cost 1.0%": (test, f, 5, 0.010),
            "enter one candle late": (test, late, 5, 0.003),
        }  # fmt: skip
        if variant == "coin":
            cases = {"base (5 slots, 0.3%)": cases["base (5 slots, 0.3%)"]}
        for case, (k, fw, slots, cost) in cases.items():
            res = simulate(e2[k], fw, Exit(bars=HOLD), slots=slots, cost=cost, priority="neg")
            st = res.stats()
            out.append({"variant": variant, "case": case, **st})
    df = pl.DataFrame(out).drop("sharpe", "days_in_market")
    df.write_csv(RESULTS / "m11_cnn_stress.csv")
    with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=220, float_precision=2):
        print(df)


if __name__ == "__main__":
    {"train": train, "report": report, "portfolio": portfolio, "stress": stress}[sys.argv[1]]()
