"""M9: walk-forward CNN study over many variants (see docs/plans/volume-spike-research.md, M9).

For each test year Y in 2022..2026 a model is trained on liquid spikes before Y-1 July, early-
stopped on Y-1 July..December, and scores year Y. Scores become percentiles of that fold's own
training scores, so a threshold never uses the future.

    uv run python research/volume_spike/m12_cnn_wf.py run base win24h ...   # train variants
    uv run python research/volume_spike/m12_cnn_wf.py eval                   # table of all runs

Inputs come from build_seq2.py. Predictions go to cache/wf/<variant>.npz.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime

import numpy as np
import polars as pl
import torch
from common import CACHE, RESULTS
from torch import nn

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
WF = CACHE / "wf"
YEARS = (2022, 2023, 2024, 2025, 2026)
SELECT_YEARS = (2022, 2023, 2024)  # the only years used to choose between variants
BATCH = 1024
PATH_STEPS = np.arange(3, 97, 3) - 1  # 15m, 30m, ... 8h (bar index of the close)
QUANTILES = (0.1, 0.5, 0.9)
COST = 0.003

COIN = (0, 1, 2, 3, 4, 5)
BTC = (7, 8)
ETH = (9, 10)
MARKET = (11, 12)
TRADES = (6,)


@dataclass(frozen=True)
class Cfg:
    channels: tuple[int, ...] = COIN + BTC
    seq: int = 96  # candles, ending at the spike candle
    target: str = "y1h"  # y15m..y4h, or "path" (quantile forecast of the 8h close path)
    loss: str = "mse"  # mse | huber | bce | xs (minus the same day's mean) | quantile (path)
    arch: str = "cnn"  # cnn | tcn | gru | mlp
    scalars: tuple[str, ...] = ("liq",)
    universe: str = "all"  # all | crash (only spikes in a crash context are trained and traded)
    seeds: int = 2
    epochs: int = 6
    width: int = 1  # channel multiplier for the conv layers
    note: str = field(default="", compare=False)


VARIANTS: dict[str, Cfg] = {
    "base": Cfg(note="M7 setup, walk-forward: coin + BTC, 8h, 1h target"),
    "coin_only": Cfg(channels=COIN, note="no BTC channels"),
    "win2h": Cfg(seq=24, note="2h window"),
    "win24h": Cfg(seq=288, note="24h window"),
    "eth": Cfg(channels=COIN + BTC + ETH, note="+ ETH"),
    "market": Cfg(channels=COIN + BTC + MARKET, note="+ market breadth and spike share"),
    "all_ch": Cfg(channels=COIN + TRADES + BTC + ETH + MARKET, note="all 13 channels"),
    "more_scalars": Cfg(scalars=("liq", "age", "hour"), note="+ coin age and hour of day"),
    "t30m": Cfg(target="y30m", note="30m target and hold"),
    "t4h": Cfg(target="y4h", note="4h target and hold"),
    "huber": Cfg(loss="huber", note="Huber loss"),
    "bce": Cfg(loss="bce", note="classify: net 1h return > 0"),
    "xs": Cfg(loss="xs", note="target minus the same day's mean: rank within the day"),
    "tcn": Cfg(arch="tcn", note="dilated temporal conv net"),
    "gru": Cfg(arch="gru", note="conv stem + GRU"),
    "mlp": Cfg(arch="mlp", note="MLP on the raw grid (no convolution)"),
    "wide": Cfg(width=2, note="twice the filters"),
    "crash": Cfg(universe="crash", note="train and trade only in a crash context"),
    "path": Cfg(target="path", loss="quantile", note="forecast the 8h path (10/50/90%)"),
    "path_all": Cfg(
        target="path",
        loss="quantile",
        channels=COIN + TRADES + BTC + ETH + MARKET,
        note="path forecast, all channels",
    ),
}
# M9 finalists (top 4 by the selection rule on 2022-24) and combinations of the per-dimension
# winners, all with 5 seeds. Defined after the screen; see the plan's research log.
VARIANTS |= {
    "crash5": replace(VARIANTS["crash"], seeds=5),
    "t4h5": replace(VARIANTS["t4h"], seeds=5),
    "huber5": replace(VARIANTS["huber"], seeds=5),
    "path_all5": replace(VARIANTS["path_all"], seeds=5),
    "combo_a": Cfg(universe="crash", target="y4h", loss="huber", seeds=5,
                   note="crash context + 4h target + Huber"),
    "combo_b": Cfg(seq=288, target="y4h", loss="huber", seeds=5,
                   note="4h target + Huber + 24h window"),
    "combo_c": Cfg(target="path", loss="quantile", channels=COIN + TRADES + BTC + ETH + MARKET,
                   universe="crash", seeds=5, note="path forecast, all channels, crash context"),
}  # fmt: skip


# ----------------------------------------------------------------------------- models
def conv_block(c_in: int, c_out: int, k: int, dil: int = 1) -> list[nn.Module]:
    pad = dil * (k // 2)
    return [nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dil), nn.BatchNorm1d(c_out), nn.GELU()]


class Head(nn.Module):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, n_out)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CNN(nn.Module):
    """The M7 network: conv blocks with pooling, then average / max / last step + scalars."""

    def __init__(self, ch: int, n_s: int, n_out: int, w: int = 1) -> None:
        super().__init__()
        a, b = 32 * w, 64 * w
        self.conv = nn.Sequential(
            *conv_block(ch, a, 5), *conv_block(a, a, 5), nn.MaxPool1d(2),
            *conv_block(a, b, 5), nn.MaxPool1d(2), *conv_block(b, b, 3), *conv_block(b, b, 3),
        )  # fmt: skip
        self.head = Head(3 * b + n_s, n_out)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return self.head(torch.cat([h.mean(2), h.amax(2), h[:, :, -1], s], 1))


class TCN(nn.Module):
    """Dilated residual convs (receptive field about 250 candles), no pooling."""

    def __init__(self, ch: int, n_s: int, n_out: int, w: int = 1) -> None:
        super().__init__()
        c = 48 * w
        self.stem = nn.Sequential(*conv_block(ch, c, 3))
        self.blocks = nn.ModuleList(
            nn.Sequential(*conv_block(c, c, 3, d)) for d in (1, 2, 4, 8, 16, 32, 64)
        )
        self.head = Head(3 * c + n_s, n_out)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(torch.cat([h.mean(2), h.amax(2), h[:, :, -1], s], 1))


class GRU(nn.Module):
    def __init__(self, ch: int, n_s: int, n_out: int, w: int = 1) -> None:
        super().__init__()
        c = 32 * w
        self.stem = nn.Sequential(*conv_block(ch, c, 5), nn.MaxPool1d(2))
        self.rnn = nn.GRU(c, 64 * w, batch_first=True)
        self.head = Head(2 * 64 * w + n_s, n_out)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.stem(x).transpose(1, 2))
        return self.head(torch.cat([h[:, -1], h.mean(1), s], 1))


class MLP(nn.Module):
    def __init__(self, ch: int, n_s: int, n_out: int, seq: int, w: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(ch * seq, 256 * w), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256 * w, 64), nn.GELU(),
        )  # fmt: skip
        self.head = Head(64 + n_s, n_out)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.net(x), s], 1))


def make(cfg: Cfg, n_s: int, n_out: int) -> nn.Module:
    ch = len(cfg.channels)
    if cfg.arch == "tcn":
        return TCN(ch, n_s, n_out, cfg.width)
    if cfg.arch == "gru":
        return GRU(ch, n_s, n_out, cfg.width)
    if cfg.arch == "mlp":
        return MLP(ch, n_s, n_out, cfg.seq, cfg.width)
    return CNN(ch, n_s, n_out, cfg.width)


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self) -> None:
        t0 = time.time()
        self.meta = pl.read_parquet(CACHE / "seq2_meta.parquet").with_columns(
            year=pl.col("open_time").dt.year(), day=pl.col("open_time").dt.date()
        )
        self.seq = np.load(CACHE / "seq2.npy")
        self.fwd = np.load(CACHE / "fwd2.npy", mmap_mode="r")
        self.t = self.meta["open_time"].to_numpy()
        m = self.meta
        early = (m["open_time"] < datetime(2021, 7, 1, tzinfo=UTC)).to_numpy()
        sample = self.seq[np.flatnonzero(early)[::10]].astype(np.float32)
        self.mu = sample.mean((0, 2)).astype(np.float32)
        self.sd = (sample.std((0, 2)) + 1e-6).astype(np.float32)
        liq = np.log(m["qv_base"].to_numpy().clip(1, None))
        age = np.log1p(m["listed_days"].to_numpy())
        hour = m["hour"].to_numpy() / 24 * 2 * np.pi
        z = lambda v: ((v - v[early].mean()) / v[early].std()).astype(np.float32)  # noqa: E731
        hour_sc = np.stack([np.sin(hour), np.cos(hour)], 1).astype(np.float32)
        self.scal = {"liq": z(liq)[:, None], "age": z(age)[:, None], "hour": hour_sc}
        self.crash = (
            (m["ret_4h"] <= -0.03) | (m["btc_vol_mult"] >= 4) | (m["btc_ret_4h"] <= -0.01)
        ).fill_null(False).to_numpy()  # fmt: skip
        fw = np.asarray(self.fwd[:, 2, :], dtype=np.float32)
        self.path = np.clip(fw[:, PATH_STEPS] - COST, -0.3, 0.3) * 100  # percent, net of cost
        print(f"data loaded in {time.time() - t0:.0f}s", flush=True)

    def target(self, cfg: Cfg) -> np.ndarray:
        if cfg.target == "path":
            return self.path
        y = np.clip(self.meta[cfg.target].to_numpy(), -0.2, 0.2) * 100
        if cfg.loss == "xs":
            day = self.meta["day"].to_physical().to_numpy()
            _, inv = np.unique(day, return_inverse=True)
            mean = np.bincount(inv, y) / np.bincount(inv)
            y = y - mean[inv]
        return y.astype(np.float32)

    def batch(self, idx: np.ndarray, cfg: Cfg) -> tuple[torch.Tensor, torch.Tensor]:
        ch = list(cfg.channels)
        x = self.seq[idx][:, ch, -cfg.seq :].astype(np.float32)
        x = np.clip((x - self.mu[ch][None, :, None]) / self.sd[ch][None, :, None], -10, 10)
        s = np.concatenate([self.scal[k][idx] for k in cfg.scalars], 1)
        return torch.from_numpy(x).to(DEVICE), torch.from_numpy(s).to(DEVICE)


def score_of(cfg: Cfg, out: torch.Tensor) -> torch.Tensor:
    """One number per spike for ranking: the 1h median forecast for path models."""
    if cfg.target == "path":
        return out.view(len(out), len(PATH_STEPS), len(QUANTILES))[:, 3, 1]
    return out.squeeze(1)


def loss_fn(cfg: Cfg, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if cfg.loss == "quantile":
        o = out.view(len(out), len(PATH_STEPS), len(QUANTILES))
        q = torch.tensor(QUANTILES, device=out.device)[None, None, :]
        e = y[:, :, None] - o
        return torch.maximum(q * e, (q - 1) * e).mean()
    o = out.squeeze(1)
    if cfg.loss == "huber":
        return nn.functional.huber_loss(o, y, delta=2.0)
    if cfg.loss == "bce":
        return nn.functional.binary_cross_entropy_with_logits(o, (y > 0).float())
    return nn.functional.mse_loss(o, y)


def predict(net: nn.Module, data: Data, cfg: Cfg, rows: np.ndarray) -> np.ndarray:
    net.eval()
    outs = []
    with torch.no_grad():
        for k in range(0, len(rows), 8192):
            x, s = data.batch(rows[k : k + 8192], cfg)
            outs.append(net(x, s).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 1))


def fit_fold(
    data: Data, cfg: Cfg, y: np.ndarray, year: int, seed: int
) -> tuple[nn.Module, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t = data.meta["open_time"]
    cut_tr = datetime(year - 1, 7, 1, tzinfo=UTC)
    cut_va = datetime(year, 1, 1, tzinfo=UTC)
    uni = data.crash if cfg.universe == "crash" else np.ones(len(y), dtype=bool)
    tr = np.flatnonzero((t < cut_tr).to_numpy() & uni)
    va = np.flatnonzero(((t >= cut_tr) & (t < cut_va)).to_numpy() & uni)
    n_out = len(PATH_STEPS) * len(QUANTILES) if cfg.target == "path" else 1
    net = make(cfg, sum(data.scal[k].shape[1] for k in cfg.scalars), n_out).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    yt = torch.from_numpy(y)
    y1h = np.clip(data.meta["y1h"].to_numpy(), -0.2, 0.2)
    best, best_state, log, bad = -np.inf, None, [], 0
    for epoch in range(cfg.epochs):
        net.train()
        t0 = time.time()
        perm = rng.permutation(tr)
        for k in range(0, len(perm), BATCH):
            b = np.sort(perm[k : k + BATCH])
            if len(b) < 32:
                continue
            x, s = data.batch(b, cfg)
            loss = loss_fn(cfg, net(x, s), yt[b].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = predict(net, data, cfg, va)
        sv = (
            pv.reshape(len(va), len(PATH_STEPS), len(QUANTILES))[:, 3, 1]
            if cfg.target == "path"
            else pv[:, 0]
        )
        corr = float(np.corrcoef(sv, y1h[va])[0, 1]) if len(va) > 10 else 0.0
        log.append(
            {
                "year": year,
                "seed": seed,
                "epoch": epoch,
                "valid_corr": corr,
                "secs": time.time() - t0,
            }
        )
        if corr > best:
            best, bad = corr, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    return net, log


def run(name: str, data: Data) -> None:
    cfg = VARIANTS[name]
    WF.mkdir(exist_ok=True)
    y = data.target(cfg)
    n = len(y)
    years = data.meta["year"].to_numpy()
    t = data.meta["open_time"]
    uni = data.crash if cfg.universe == "crash" else np.ones(n, dtype=bool)
    pct = np.full(n, np.nan, dtype=np.float32)
    raw = np.full(n, np.nan, dtype=np.float32)
    pct_seed = np.full((cfg.seeds, n), np.nan, dtype=np.float32)
    path = (
        np.full((n, len(PATH_STEPS), len(QUANTILES)), np.nan, dtype=np.float16)
        if cfg.target == "path"
        else None
    )
    logs = []
    t_start = time.time()
    for year in YEARS:
        test = np.flatnonzero((years == year) & uni)
        train_rows = np.flatnonzero((t < datetime(year - 1, 7, 1, tzinfo=UTC)).to_numpy() & uni)
        ref = np.sort(train_rows[:: max(1, len(train_rows) // 50_000)])
        outs_test, outs_ref = [], []
        for seed in range(cfg.seeds):
            net, log = fit_fold(data, cfg, y, year, seed)
            logs += log
            outs_test.append(predict(net, data, cfg, test))
            outs_ref.append(predict(net, data, cfg, ref))
        o_test, o_ref = np.mean(outs_test, 0), np.mean(outs_ref, 0)
        if cfg.target == "path":
            o3 = o_test.reshape(len(test), len(PATH_STEPS), len(QUANTILES))
            assert path is not None
            path[test] = o3.astype(np.float16)
            s_test, s_ref = (
                o3[:, 3, 1],
                o_ref.reshape(len(ref), len(PATH_STEPS), len(QUANTILES))[:, 3, 1],
            )
        else:
            s_test, s_ref = o_test[:, 0], o_ref[:, 0]
        raw[test] = s_test
        pct[test] = np.searchsorted(np.sort(s_ref), s_test) / len(s_ref)
        for j, (ot, orf) in enumerate(zip(outs_test, outs_ref, strict=True)):
            if cfg.target == "path":
                ot, orf = (
                    ot.reshape(len(test), -1, 3)[:, 3, 1:2],
                    orf.reshape(len(ref), -1, 3)[:, 3, 1:2],
                )
            pct_seed[j, test] = np.searchsorted(np.sort(orf[:, 0]), ot[:, 0]) / len(orf)
        print(f"{name} {year}: {len(test)} scored, {time.time() - t_start:.0f}s", flush=True)
    out = {"pct": pct, "raw": raw, "pct_seed": pct_seed}
    if path is not None:
        out["path"] = path
    np.savez(WF / f"{name}.npz", **out)
    pl.DataFrame(logs).with_columns(variant=pl.lit(name)).write_csv(WF / f"{name}_log.csv")


# ----------------------------------------------------------------------------- evaluation
def portfolio_years(data: Data, pct: np.ndarray, hold: int, q: float = 0.995, slots: int = 5,
                    cost: float = COST) -> list[dict[str, object]]:  # fmt: skip
    from portfolio import Exit, simulate

    out = []
    years = data.meta["year"].to_numpy()
    for year in YEARS:
        k = np.flatnonzero((years == year) & (pct >= q))
        sig = data.meta[k].with_columns(neg=pl.Series(-pct[k]))
        res = simulate(
            sig, np.asarray(data.fwd[k]), Exit(bars=hold), slots=slots, cost=cost, priority="neg"
        )
        st = res.stats()
        out.append(
            {
                "year": year,
                **{key: st.get(key) for key in ("trades", "mean_bp", "win", "cagr", "max_dd")},
            }
        )
    return out


HOLD = {"y15m": 3, "y30m": 6, "y1h": 12, "y2h": 24, "y4h": 48, "path": 12}


Q_MAIN = 0.98  # top 2% of the fold's training scores (see the plan's M9 amendment)


def evaluate() -> None:
    data = Data()
    rows = []
    for name, cfg in VARIANTS.items():
        f = WF / f"{name}.npz"
        if not f.exists():
            continue
        pct = np.nan_to_num(np.load(f)["pct"], nan=-1)
        row: dict[str, object] = {"variant": name, "note": cfg.note}
        for q in (Q_MAIN, 0.97, 0.99):
            by = {r["year"]: r for r in portfolio_years(data, pct, HOLD[cfg.target], q=q)}
            sel = [float(by[y]["mean_bp"] or 0.0) for y in SELECT_YEARS]
            tag = "" if q == Q_MAIN else f"_q{round(q * 100)}"
            late = [float(by[y]["mean_bp"] or 0.0) for y in (2025, 2026)]
            row |= {
                f"select_mean_bp{tag}": float(np.mean(sel)),
                f"select_worst_bp{tag}": min(sel),
                f"late_mean_bp{tag}": float(np.mean(late)),
            }
            if q == Q_MAIN:
                row |= {f"bp_{y}": by[y]["mean_bp"] for y in YEARS}
                row |= {f"n_{y}": by[y]["trades"] for y in YEARS}
                row |= {f"dd_{y}": by[y]["max_dd"] for y in YEARS}
        row["cfg"] = str(asdict(cfg))
        rows.append(row)
    df = pl.DataFrame(rows).sort("select_mean_bp", descending=True)
    df.write_csv(RESULTS / "m12_variants.csv")
    cols = ["variant", "select_mean_bp", "select_worst_bp", *[f"bp_{y}" for y in YEARS]]
    cols += ["late_mean_bp", "select_mean_bp_q97", "select_mean_bp_q99"]
    cols += ["late_mean_bp_q97", "late_mean_bp_q99"]
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=260, float_precision=0):
        print(df.select(cols))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        d = Data()
        for v in sys.argv[2:]:
            run(v, d)
    else:
        evaluate()
