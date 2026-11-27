"""Walk-forward neural rankers for big-move events (M3): transformer, TCN, tabular MLP.

Inputs: the 24h, 13-channel 5m sequence of seq2 (coin OHLC, volume, taker share, trades, BTC, ETH,
market) and the tabular features of feats.parquet (standardised on the training years, NaN -> 0
plus a missing flag). Target: the trade's price return at the horizon in the trade's direction,
clipped to [-0.5, 1], Huber loss. For each test year Y: train on events before Y-1 July, early-stop
on Y-1 July..December, predict Y. Seeds are averaged.

Run: uv run --group research python nn_seq.py <short|long> <transformer|tcn|mlp> [horizon] [seeds]
Writes OWN/nn_<side>_<model>_<h>.parquet (row, pred, pq).
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime

import numpy as np
import polars as pl
import torch
from common_ls import CACHE_ALL, OWN
from e08_gbm import FEATS
from torch import nn

SIDE = sys.argv[1]
MODEL = sys.argv[2]
H = sys.argv[3] if len(sys.argv) > 3 else "72h"
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
S = -1 if SIDE == "short" else 1
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def data() -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    d = pl.read_parquet(OWN / "feats.parquet")
    d = d.filter(pl.col("ret_4h") >= 0.15) if S < 0 else d.filter(pl.col("ret_4h") <= -0.10)
    d = d.filter(pl.col("qv_base") >= 20_000).sort("row").with_columns(
        target=(S * pl.col(f"y{H}")).clip(-0.5, 1.0), year=pl.col("open_time").dt.year()
    )
    seq = np.load(CACHE_ALL / "seq2.npy", mmap_mode="r")
    x = np.asarray(seq[d["row"].to_numpy()], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    tab = d.select(FEATS).to_numpy().astype(np.float32)
    return d, x, tab


class Transformer(nn.Module):
    def __init__(self, ch: int, ntab: int, d: int = 64, patch: int = 12) -> None:
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(ch * patch, d)
        self.pos = nn.Parameter(torch.randn(1, 288 // patch + 1, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 3)
        self.tab = nn.Sequential(nn.Linear(ntab, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d + 64, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        p = x.reshape(b, c, n // self.patch, self.patch).permute(0, 2, 1, 3).reshape(b, n // self.patch, c * self.patch)
        z = torch.cat([self.cls.expand(b, -1, -1), self.embed(p)], 1) + self.pos
        z = self.enc(z)[:, 0]
        return self.head(torch.cat([z, self.tab(t)], 1)).squeeze(1)


class TCN(nn.Module):
    def __init__(self, ch: int, ntab: int, w: int = 48) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        cin = ch
        for dil in (1, 2, 4, 8, 16, 32):
            layers += [nn.Conv1d(cin, w, 3, padding=dil, dilation=dil), nn.GELU(), nn.BatchNorm1d(w)]
            cin = w
        self.conv = nn.Sequential(*layers)
        self.tab = nn.Sequential(nn.Linear(ntab, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * w + 64, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        z = torch.cat([z[:, :, -1], z.mean(2), self.tab(t)], 1)
        return self.head(z).squeeze(1)


class MLP(nn.Module):
    def __init__(self, ch: int, ntab: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ntab, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(t).squeeze(1)


def standardise(tab: np.ndarray, tr: np.ndarray) -> np.ndarray:
    miss = np.isnan(tab)
    mu = np.nanmean(tab[tr], 0)
    sd = np.nanstd(tab[tr], 0) + 1e-6
    z = np.clip((tab - mu) / sd, -6, 6)
    z[miss] = 0
    return np.concatenate([z, miss.astype(np.float32)], 1).astype(np.float32)


def train_one(x: np.ndarray, tab: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = {"transformer": Transformer, "tcn": TCN, "mlp": MLP}[MODEL](x.shape[1], tab.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    loss_fn = nn.HuberLoss(delta=0.1)

    def to(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a).to(DEV)

    def predict(idx: np.ndarray) -> np.ndarray:
        net.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(idx), 4096):
                j = idx[k : k + 4096]
                out.append(net(to(x[j]), to(tab[j])).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    tri, vai, tei = np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te)
    best, best_state, bad = -np.inf, None, 0
    for _ep in range(30):
        net.train()
        perm = np.random.permutation(tri)
        for k in range(0, len(perm), 256):
            j = perm[k : k + 256]
            opt.zero_grad()
            loss = loss_fn(net(to(x[j]), to(tab[j])), to(y[j]))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        pv = predict(vai)
        ic = float(pl.DataFrame({"p": pv, "y": y[vai]}).select(pl.corr(pl.col("p").rank(), pl.col("y").rank())).item())
        if ic > best:
            best, bad = ic, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    return predict(tei), predict(tri)


def main() -> None:
    d, x, tab_raw = data()
    y = d["target"].to_numpy().astype(np.float32)
    t = d["open_time"]
    pred = np.full(d.height, np.nan)
    pq = np.full(d.height, np.nan)
    t0 = time.time()
    for Y in (2022, 2023, 2024, 2025, 2026):
        cut = datetime(Y - 1, 7, 1, tzinfo=UTC)
        tr = (t < cut).to_numpy()
        va = ((t >= cut) & (t < datetime(Y, 1, 1, tzinfo=UTC))).to_numpy()
        te = (d["year"] == Y).to_numpy()
        tab = standardise(tab_raw, tr)
        pt, ptr = [], []
        for s in range(SEEDS):
            a, b = train_one(x, tab, y, tr, va, te, s)
            pt.append(a)
            ptr.append(b)
        p_te, p_tr = np.mean(pt, 0), np.mean(ptr, 0)
        pred[te] = p_te
        pq[te] = np.searchsorted(np.sort(p_tr), p_te) / len(p_tr)
        ic = np.corrcoef(np.argsort(np.argsort(p_te)), np.argsort(np.argsort(y[te])))[0, 1]
        print(f"{Y}: n={te.sum()} IC {ic:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
    out = d.select("row").with_columns(pred=pl.Series(pred), pq=pl.Series(pq))
    out.write_parquet(OWN / f"nn_{SIDE}_{MODEL}_{H}.parquet")


if __name__ == "__main__":
    main()
