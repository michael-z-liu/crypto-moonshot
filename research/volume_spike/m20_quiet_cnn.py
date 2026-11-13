"""Quiet-coin CNN (B3 step 3): which quiet coins jump +20% within 3 days?

Candidates: quiet snapshots (Q7 deciles 1-3, build_quiet_seq.py). Inputs: 7 days of 1h candles
(10 channels), the last 24h of 5m candles (6 channels) and 23 snapshot features. Target: the high
reaches +20% within 3 days of entry. Walk-forward as M9: for test year Y the model trains before
Y-1 July, stops early on Y-1 July-December (top-2% hit rate) and scores Y; 2 seeds averaged;
scores become percentiles of the fold's training scores. Comparison (not a candidate): boosted
trees on the snapshot features only, same folds. Trading: the $50 no-stop book of m18_longshot.

Run: MOONSHOT_UNIVERSE=all uv run python research/volume_spike/m20_quiet_cnn.py run|gbm|eval
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime

import numpy as np
import polars as pl
import torch
from common import BARS_PER_DAY, CACHE, RESULTS
from m12_cnn_wf import BATCH, DEVICE, WF, YEARS, Head, conv_block
from torch import nn

TARGET_BARS = 3 * BARS_PER_DAY
EPOCHS, SEEDS = 6, 2
LOGS = [
    "q1",
    "q7",
    "rv1_rel",
    "rv7_rel",
    "qv1_rel",
    "qv7_rel",
    "rng1_rel",
    "rng7_rel",
    "trades_rel",
    "avgtrade_rel",
    "tick_rel",
    "qv_med",
    "btc_rv7_rel",
]
PLAIN = ["flat1", "ret1", "ret7", "ret30", "dist_hi30", "tbr_diff", "btc_ret1", "btc_ret7"]
LOG1P = ["days_since_spike", "age_days"]


def scalar_matrix(m: pl.DataFrame, early: np.ndarray) -> np.ndarray:
    cols = []
    for c in LOGS:
        cols.append(np.log(np.clip(m[c].to_numpy().astype(np.float64), 1e-9, None)))
    for c in PLAIN:
        cols.append(m[c].to_numpy().astype(np.float64))
    for c in LOG1P:
        cols.append(np.log1p(np.clip(m[c].to_numpy().astype(np.float64), 0, None)))
    x = np.stack(cols, 1)
    x = np.where(np.isfinite(x), x, np.nan)
    mu, sd = np.nanmean(x[early], 0), np.nanstd(x[early], 0) + 1e-9
    return np.clip(np.nan_to_num((x - mu) / sd, nan=0.0), -5, 5).astype(np.float32)


class QData:
    def __init__(self) -> None:
        self.meta = pl.read_parquet(CACHE / "qseq_meta.parquet")
        self.h = np.load(CACHE / "qseq_h.npy", mmap_mode="r")
        self.m = np.load(CACHE / "qseq_m.npy", mmap_mode="r")
        t = self.meta["open_time"]
        self.t = t
        self.year = self.meta["year"].to_numpy()
        up = self.meta["up_20"].to_numpy()
        self.y = ((up >= 0) & (up < TARGET_BARS)).astype(np.float32)
        early = (t < datetime(2021, 7, 1, tzinfo=UTC)).to_numpy()
        k = np.flatnonzero(early)[::20]
        sh = np.asarray(self.h[np.sort(k)], dtype=np.float32)
        sm = np.asarray(self.m[np.sort(k)], dtype=np.float32)
        self.mu_h, self.sd_h = sh.mean((0, 2)), sh.std((0, 2)) + 1e-6
        self.mu_m, self.sd_m = sm.mean((0, 2)), sm.std((0, 2)) + 1e-6
        self.s = scalar_matrix(self.meta, early)

    def batch(self, idx: np.ndarray) -> tuple[torch.Tensor, ...]:
        h = (np.asarray(self.h[idx], dtype=np.float32) - self.mu_h[None, :, None]) / self.sd_h[
            None, :, None
        ]
        m = (np.asarray(self.m[idx], dtype=np.float32) - self.mu_m[None, :, None]) / self.sd_m[
            None, :, None
        ]
        return (
            torch.from_numpy(np.clip(h, -10, 10)).to(DEVICE),
            torch.from_numpy(np.clip(m, -10, 10)).to(DEVICE),
            torch.from_numpy(self.s[idx]).to(DEVICE),
        )


def trunk(c_in: int) -> nn.Sequential:
    return nn.Sequential(
        *conv_block(c_in, 32, 5),
        *conv_block(32, 32, 5),
        nn.MaxPool1d(2),
        *conv_block(32, 64, 5),
        nn.MaxPool1d(2),
        *conv_block(64, 64, 3),
    )


class TwoBranch(nn.Module):
    def __init__(self, n_s: int) -> None:
        super().__init__()
        self.th, self.tm = trunk(10), trunk(6)
        self.head = Head(6 * 64 + n_s, 1)

    def forward(self, h: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a, b = self.th(h), self.tm(m)
        z = [a.mean(2), a.amax(2), a[:, :, -1], b.mean(2), b.amax(2), b[:, :, -1], s]
        return self.head(torch.cat(z, 1)).squeeze(1)


def predict(net: nn.Module, d: QData, rows: np.ndarray) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(rows), 8192):
            out.append(net(*d.batch(rows[k : k + 8192])).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def folds(d: QData, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cut = datetime(year - 1, 7, 1, tzinfo=UTC)
    tr = np.flatnonzero((d.t < cut).to_numpy())
    va = np.flatnonzero(((d.t >= cut) & (d.t < datetime(year, 1, 1, tzinfo=UTC))).to_numpy())
    return tr, va, np.flatnonzero(d.year == year)


def top_hit(score: np.ndarray, y: np.ndarray, q: float = 0.98) -> float:
    return float(y[score >= np.quantile(score, q)].mean())


def run() -> None:
    d = QData()
    print("rows", len(d.y), "base rate", d.y.mean(), flush=True)
    pct = np.full(len(d.y), np.nan, dtype=np.float32)
    logs = []
    for year in YEARS:
        tr, va, te = folds(d, year)
        ref = np.sort(tr[:: max(1, len(tr) // 50_000)])
        o_te, o_ref = [], []
        for seed in range(SEEDS):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = TwoBranch(d.s.shape[1]).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            yt = torch.from_numpy(d.y)
            best, state, bad = -np.inf, None, 0
            for epoch in range(EPOCHS):
                net.train()
                t0 = time.time()
                perm = rng.permutation(tr)
                for k in range(0, len(perm), BATCH):
                    b = np.sort(perm[k : k + BATCH])
                    if len(b) < 32:
                        continue
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        net(*d.batch(b)), yt[b].to(DEVICE)
                    )
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                score = top_hit(predict(net, d, va), d.y[va])
                logs.append(
                    {
                        "year": year,
                        "seed": seed,
                        "epoch": epoch,
                        "valid_top2_hit": score,
                        "valid_base": float(d.y[va].mean()),
                        "secs": time.time() - t0,
                    }
                )
                print(logs[-1], flush=True)
                if score > best:
                    best, bad = score, 0
                    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 2:
                        break
            assert state is not None
            net.load_state_dict(state)
            o_te.append(predict(net, d, te))
            o_ref.append(predict(net, d, ref))
        s_te, s_ref = np.mean(o_te, 0), np.mean(o_ref, 0)
        pct[te] = np.searchsorted(np.sort(s_ref), s_te) / len(s_ref)
    WF.mkdir(exist_ok=True)
    np.savez(WF / "quiet_cnn.npz", pct=pct)
    pl.DataFrame(logs).write_csv(WF / "quiet_cnn_log.csv")


def gbm() -> None:
    """Comparison: boosted trees on the 23 snapshot features, same folds."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    d = QData()
    pct = np.full(len(d.y), np.nan, dtype=np.float32)
    for year in YEARS:
        tr, _, te = folds(d, year)
        sub = np.random.default_rng(0).choice(tr, size=min(600_000, len(tr)), replace=False)
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=500,
            l2_regularization=1.0,
            random_state=0,
        )
        model.fit(d.s[sub], d.y[sub])
        ref = np.sort(tr[:: max(1, len(tr) // 50_000)])
        s_ref, s_te = model.predict_proba(d.s[ref])[:, 1], model.predict_proba(d.s[te])[:, 1]
        pct[te] = np.searchsorted(np.sort(s_ref), s_te) / len(s_ref)
        print(year, "done", flush=True)
    np.savez(WF / "quiet_gbm.npz", pct=pct)


def evaluate() -> None:
    from m18_longshot import run_book, show
    from m19_quiet import EX_CASCADE, as_book
    from sklearn.metrics import roc_auc_score

    d = QData()
    meta = as_book(d.meta)
    rows, cls = [], []
    rng = np.random.default_rng(0)
    for name in ("quiet_cnn", "quiet_gbm"):
        f = WF / f"{name}.npz"
        if not f.exists():
            continue
        pct = np.load(f)["pct"]
        # placebo: the same scores shuffled within each 4-hour snapshot time
        tkey = meta["open_time"].to_physical().to_numpy()
        plc = pct.copy()
        order = np.argsort(tkey, kind="stable")
        bounds = np.flatnonzero(np.diff(tkey[order])) + 1
        for g in np.split(order, bounds):
            plc[g] = rng.permutation(pct[g])
        for year in YEARS:
            k = d.year == year
            s, y = pct[k], d.y[k]
            cls.append(
                {
                    "model": name,
                    "year": year,
                    "base": float(y.mean()),
                    "auc": float(roc_auc_score(y, s)),
                    **{f"hit_top{t}": top_hit(s, y, 1 - t / 100) for t in (1, 2, 5)},
                }
            )
        for tag, p in ((name, pct), (name + " placebo", plc)):
            m = meta.with_columns(pct=pl.Series(p), neg=pl.Series(-p)).filter(
                pl.col("pct").is_not_nan()
            )
            for q in (0.95, 0.98, 0.99):
                sel = m.filter(pl.col("pct") >= q)
                label = f"{tag} top {(1 - q) * 100:g}%"
                rows += run_book(sel, label, "neg", range(2022, 2027))
                dx = sel.filter(~pl.col("open_time").is_between(*EX_CASCADE, closed="left"))
                rows += run_book(dx, label + " ex-cascade", "neg", range(2025, 2026))
    base = meta.filter(pl.col("year") >= 2022).with_columns(neg=-pl.col("q7"))
    rows += run_book(base, "all quiet snapshots (Q7 d1-3)", "neg", range(2022, 2027))
    pl.DataFrame(cls).write_csv(RESULTS / "m20_quiet_cls.csv")
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m20_quiet_book.csv")
    with pl.Config(tbl_rows=40, float_precision=3):
        print(pl.DataFrame(cls))
    show(df.filter(~pl.col("entry").str.ends_with("ex-cascade")))


if __name__ == "__main__":
    {"run": run, "gbm": gbm, "eval": evaluate}[sys.argv[1]]()
