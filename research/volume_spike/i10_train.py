"""Issue #10 training harness (docs/plans/cnn-improvements.md, M1).

Walk-forward training of m12's path models with the options issue #10 asks to test. It reuses
`m12_cnn_wf` (`Data`, the networks via `make`, `predict`, `loss_fn`, `folds_of`, `PATH_STEPS`,
`QUANTILES`) and changes none of m12's behaviour. With every option at its default an `I10Cfg`
trains exactly as `m12_cnn_wf.run` does (same rows, same RNG calls, same loss, same selection),
so base3's seeds 0 and 1 reproduce `cache_all/wf/path_all.npz`.

    cd research/volume_spike
    export MOONSHOT_UNIVERSE=all
    uv run --group research python i10_train.py run base3               # all folds, 3 seeds
    uv run --group research python i10_train.py run base3 --years 2022 --seeds 1 \\
        --set weight=day --sub 4 --epochs 1 --out smoke_day           # smoke run
    uv run --group research python i10_train.py compare base3 path_all  # reproduction check
    uv run --group research python i10_train.py selfcheck               # weight / head checks
    uv run --group research python i10_train.py list

Outputs go to the local `cache_all/wf_i10/` (never to `cache_all/wf/`, whose files are links into
another worktree): `<name>.npz` and `<name>_log.csv`. Any override (`--years`, `--seeds`, `--sub`,
`--epochs`, `--set`) needs `--out`, so a smoke run never replaces a full run.

**Options** (`I10Cfg`, defaults = m12's path_all training):

- `select`: how the epoch is chosen on the validation half-year (Y-1 July..December).
  `"corr"`: correlation of the 1h-median score with y1h clipped to +-20% (m12), patience 2, at
  most `base.epochs` epochs. `"top"`: mean of the hold-horizon net target (`hold`, default y1h,
  net of the 0.3% round trip, clipped to +-20% as for corr) over the top `top_frac` (2%) of
  validation rows by the `select_head` score; same patience. `"fixed"`: exactly `fixed_epochs`
  epochs, the last one kept. Both metrics are logged every epoch whatever the mode.
- `weight`: per-row loss weights; the batch loss is sum(w * l) / sum(w) with l the row's loss
  (quantile loss averaged over steps x quantiles). `"none"` uses m12's unweighted loss as is.
  `"day"`: every UTC day of the fold's training rows gets the same total weight (w = 1 / rows
  that day, scaled to mean 1 over the training rows). `"tail"`: w = 1 + `tail_k` x m12's crash
  mask (`Data.crash`). `"recency"`: w = 0.5 ** (age / `half_life`), age in years (365.25 days)
  back from the fold's train cut (w = 1 at the cut).
- `window_years`: None trains on every row before the cut (expanding, m12); N trains only on
  rows in [cut - N calendar years, cut). The reference rows for the fold percentiles are the
  fold's (windowed) training rows.
- `norm`: `"frozen"` (m12): channel mean / sd from every 10th row before 2021-07-01 (all 288
  candles), liq and age z-scored on all those rows. `"fold"`: the same statistics from the fold's
  own training rows (every 10th for the channels, all for the scalars), so a fold never sees
  statistics of its validation or test period. The hour scalars are never scaled.
- `volscale`: each row's trailing volatility `vol` = the standard deviation of the coin's 5m log
  close returns over the model's input window (the last `base.seq` candles; diffs of seq2
  channel 3, the log close relative to the spike close), floored at `vol_floor` (0.1% per 5m).
  Scaled: (a) the four coin price channels 0-3 (log open / high / low / close relative to the
  spike close) are divided by `vol` before normalisation, and their normalisation statistics
  are computed on the scaled values (the frozen or fold rows as `norm` says; the other channels
  keep their statistics); (b) the path target (percent, net of cost, clipped to +-30%, minus
  `target_offset` if given) is divided by 100 x `vol`, i.e. it is in units of the coin's 5m
  volatility. Nothing else (volume, trades, BTC / ETH / market channels, scalars) is scaled.
  The network's outputs are multiplied back by 100 x `vol` (percent) before any head, selection
  metric or percentile, so `path`, `raw*` and `pct*` are in return units; the scaled 8h path
  (`path_scaled`) and its 1h median (`raw_scaled`, percentile `pct_scaled`) are saved as well.
  The risk-adjusted head is scale-free, but its floor applies in return units.
- `target_offset`: None, or the name of a local `.npy` in `cache_all/` with an (n_rows,
  len(PATH_STEPS)) array in the path target's units (percent) that is subtracted from the
  target (M4, the target relative to the universe). Only the training target changes.

**Heads** (path models): `h1` 1h median (step 3, q 0.5; m12's score), `h2` 2h median (step 7),
`h2ra` 2h median / max(q90 - q10 at 2h, `RA_FLOOR` = 0.05 percentage points). Each becomes a
fold percentile as m12 does it: searchsorted against the sorted head scores of the fold's
reference training rows (every k-th, about 50,000). The seed mean averages the seeds' raw
outputs first (as m12), then takes the head. npz keys: `pct` (= h1), `pct_h2`, `pct_h2ra`, `raw`,
`raw_h2`, `raw_h2ra`, `pct_seed` (h1 per seed, (seeds, n)), `pct_seed_h2`, `pct_seed_h2ra`,
`path` (float16 seed-mean (n, steps, quantiles)), `cfg` (the config text), plus the volscale
keys above. The log CSV has
one row per (fold, seed, epoch): the selection metric, valid corr, valid top-2% mean and the
epoch's seconds (training + validation). Fold 2026 validates on 2025 H2, a sealed period: its
`valid_top` (a mean return) is not printed, and the log is not to be read for it before M5.
"""

from __future__ import annotations

import ast
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path

import m12_cnn_wf as m12
import numpy as np
import polars as pl
import torch
from common import CACHE
from torch import nn

WF_I10 = CACHE / "wf_i10"
YEARS = m12.YEARS
PRICE = (0, 1, 2, 3)  # coin open, high, low, close (log ratio to the spike close)
CLOSE = 3
RA_FLOOR = 0.05  # percentage points: floor of the 2h q90 - q10 band in the h2ra head
HEADS = ("h1", "h2", "h2ra")
STEP_1H, STEP_2H = 3, 7  # indices into PATH_STEPS: bars 11 (1h) and 23 (2h)
US_PER_DAY = 86_400_000_000
US_PER_YEAR = 365.25 * US_PER_DAY
SEALED_FROM = datetime(2025, 1, 1, tzinfo=UTC)  # 2025-26 is sealed (plan, Evaluation protocol)
CHUNK = 10_000  # rows per read when computing statistics from seq2


@dataclass(frozen=True)
class I10Cfg:
    """An m12 config plus the issue #10 options (see the module docstring)."""

    base: m12.Cfg
    select: str = "corr"  # corr | top | fixed
    patience: int = 2
    fixed_epochs: int = 6
    top_frac: float = 0.02
    hold: str = "y1h"  # target column of the top metric
    select_head: str = "h1"  # score of the top metric
    weight: str = "none"  # none | day | tail | recency
    tail_k: float = 3.0
    half_life: float = 1.0  # years
    window_years: int | None = None
    norm: str = "frozen"  # frozen | fold
    volscale: bool = False
    vol_floor: float = 1e-3
    target_offset: str | None = None
    note: str = field(default="", compare=False)

    def check(self) -> None:
        if self.base.target != "path" or self.base.loss != "quantile":
            raise ValueError("the harness trains path models (target path, loss quantile)")
        for name, value, ok in (
            ("select", self.select, ("corr", "top", "fixed")),
            ("weight", self.weight, ("none", "day", "tail", "recency")),
            ("norm", self.norm, ("frozen", "fold")),
            ("select_head", self.select_head, HEADS),
        ):
            if value not in ok:
                raise ValueError(f"{name}={value!r}, expected one of {ok}")

    def describe(self) -> str:
        """The config without notes, for the ledger's source text (part of the config hash)."""
        d = asdict(self)
        d.pop("note")
        d["base"].pop("note")
        return str(d)


PATH_ALL = m12.VARIANTS["path_all"]
I10: dict[str, I10Cfg] = {
    "base3": I10Cfg(
        base=replace(PATH_ALL, seeds=3, note="path_all, 3 seeds"),
        note="M1 baseline: path_all retrained with the harness, all defaults, 3 seeds",
    ),
}
_B3 = I10["base3"]
# M3 (ideas 1-2): each changes only the named options of base3.
# fixed_epochs=2: median over base3's (fold, seed) runs in 2022-24 of (kept epoch + 1); see M3.md.
I10 |= {
    "es_top": replace(_B3, select="top", note="M3: early stop on top-2% validation y1h return"),
    "tail": replace(
        _B3, select="top", weight="tail", note="M3: tail-weighted loss (1 + 3 x crash) + es_top"
    ),
    "day": replace(_B3, weight="day", note="M3: equal total loss weight per UTC day"),
    "fixed": replace(_B3, select="fixed", fixed_epochs=2, note="M3: fixed 2 epochs, no early stop"),
}
# M4 (ideas 5, 6, 8): each changes only the named options of base3. univ_fwd.npy: i10_univ.py.
I10 |= {
    "xs_univ": replace(
        _B3,
        target_offset="wf_i10/univ_fwd.npy",
        note="M4: target minus the universe's equal-weight forward return (same window)",
    ),
    "volscale": replace(_B3, volscale=True, note="M4: target and price channels / trailing vol"),
    "recency": replace(
        _B3, weight="recency", half_life=1.0, note="M4: recency-weighted loss, half-life 1 year"
    ),
    "roll3y": replace(_B3, window_years=3, note="M4: rolling 3-year training window"),
    "foldnorm": replace(_B3, norm="fold", note="M4: normalisation statistics per fold"),
}


# ----------------------------------------------------------------------------- weights


def day_weights(day: np.ndarray) -> np.ndarray:
    """Equal total weight per UTC day (`day`: integer day of each training row), mean 1."""
    _, inv, counts = np.unique(day, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w / w.mean()


def recency_weights(t_us: np.ndarray, cut_us: int, half_life: float) -> np.ndarray:
    """0.5 ** (age / half_life), age in years back from the train cut (1 at the cut)."""
    age = (cut_us - t_us) / US_PER_YEAR
    return np.power(0.5, age / half_life)


def tail_weights(crash: np.ndarray, k: float) -> np.ndarray:
    return 1.0 + k * crash.astype(np.float64)


def row_weights(icfg: I10Cfg, data: m12.Data, tr: np.ndarray, cut: datetime) -> np.ndarray | None:
    """Per-row loss weights (float32, full length; only the training rows matter)."""
    if icfg.weight == "none":
        return None
    w = np.ones(len(data.meta), dtype=np.float32)
    t_us = data.meta["open_time"].dt.epoch("us").to_numpy()
    if icfg.weight == "day":
        w[tr] = day_weights(t_us[tr] // US_PER_DAY)
    elif icfg.weight == "tail":
        w[tr] = tail_weights(data.crash[tr], icfg.tail_k)
    else:
        w[tr] = recency_weights(t_us[tr], _us(cut), icfg.half_life)
    return w


def _us(t: datetime) -> int:
    return int((t - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds()) * 1_000_000


def row_loss(cfg: m12.Cfg, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """m12's quantile loss per row (averaged over steps x quantiles)."""
    o = out.view(len(out), len(m12.PATH_STEPS), len(m12.QUANTILES))
    q = torch.tensor(m12.QUANTILES, device=out.device)[None, None, :]
    e = y[:, :, None] - o
    return torch.maximum(q * e, (q - 1) * e).mean((1, 2))


def batch_loss(
    cfg: m12.Cfg, out: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None
) -> torch.Tensor:
    if w is None:
        return m12.loss_fn(cfg, out, y)
    return (w * row_loss(cfg, out, y)).sum() / w.sum()


# ----------------------------------------------------------------------------- data


def vol_from_close(close: np.ndarray) -> np.ndarray:
    """Std of the 5m log returns of (rows, candles) log closes (seq2 channel 3)."""
    return np.diff(close, axis=1).std(1)


def vol_of(data: m12.Data, seq: int, floor: float) -> np.ndarray:
    """Per-row std of the 5m log close returns over the last `seq` candles, floored (float32).

    Cached unfloored as `wf_i10/vol_s<seq>.npy` (one pass over seq2)."""
    path = WF_I10 / f"vol_s{seq}.npy"
    if path.exists():
        vol = np.load(path)
    else:
        n = len(data.meta)
        vol = np.empty(n, dtype=np.float32)
        for k in range(0, n, 50_000):
            c = np.asarray(data.seq[k : k + 50_000, CLOSE, -seq:], dtype=np.float32)
            vol[k : k + 50_000] = vol_from_close(c)
        np.save(_new_file(path), vol)
    return np.maximum(vol, floor).astype(np.float32)


def channel_stats(
    data: m12.Data, rows: np.ndarray, vol: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd per channel over `rows` and all stored candles (as m12's frozen statistics);
    with `vol`, the price channels are divided by each row's vol first."""
    s1 = np.zeros(data.seq.shape[1])
    s2 = np.zeros(data.seq.shape[1])
    count = 0
    for k in range(0, len(rows), CHUNK):
        r = rows[k : k + CHUNK]
        x = data.seq[r].astype(np.float64)
        if vol is not None:
            x[:, list(PRICE)] /= vol[r][:, None, None]
        s1 += x.sum((0, 2))
        s2 += (x * x).sum((0, 2))
        count += x.shape[0] * x.shape[2]
    mu = s1 / count
    sd = np.sqrt(np.maximum(s2 / count - mu * mu, 0.0)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


class FoldData(m12.Data):
    """m12's `Data` with a fold's own normalisation and optional volatility scaling.

    Built from a loaded `Data` (shares its arrays; `__init__` is not run again)."""

    vol: np.ndarray | None

    @classmethod
    def of(
        cls,
        base: m12.Data,
        mu: np.ndarray,
        sd: np.ndarray,
        scal: dict[str, np.ndarray],
        vol: np.ndarray | None,
    ) -> FoldData:
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.mu, obj.sd, obj.scal, obj.vol = mu, sd, scal, vol
        return obj

    def batch(self, idx: np.ndarray, cfg: m12.Cfg) -> tuple[torch.Tensor, torch.Tensor]:
        ch = list(cfg.channels)
        x = self.seq[idx][:, ch, -cfg.seq :].astype(np.float32)
        if self.vol is not None:
            pos = [i for i, c in enumerate(ch) if c in PRICE]
            x[:, pos] /= self.vol[idx][:, None, None]
        x = np.clip((x - self.mu[ch][None, :, None]) / self.sd[ch][None, :, None], -10, 10)
        s = np.concatenate([self.scal[k][idx] for k in cfg.scalars], 1)
        return torch.from_numpy(x).to(m12.DEVICE), torch.from_numpy(s).to(m12.DEVICE)


EARLY = datetime(2021, 7, 1, tzinfo=UTC)  # m12's frozen statistics: rows before this


def fold_data(data: m12.Data, icfg: I10Cfg, tr: np.ndarray, vol: np.ndarray | None) -> m12.Data:
    """The data view one fold trains and scores with (m12's own `data` with default options)."""
    if icfg.norm == "frozen" and vol is None:
        return data
    m = data.meta
    early = np.flatnonzero((m["open_time"] < EARLY).to_numpy())
    rows = tr if icfg.norm == "fold" else early
    mu, sd = channel_stats(data, rows[::10], vol)
    scal = dict(data.scal)
    if icfg.norm == "frozen":  # only the price channels change (their scaled statistics)
        mu0, sd0 = data.mu.copy(), data.sd.copy()
        mu0[list(PRICE)], sd0[list(PRICE)] = mu[list(PRICE)], sd[list(PRICE)]
        mu, sd = mu0, sd0
    else:
        liq = np.log(m["qv_base"].to_numpy().clip(1, None))
        age = np.log1p(m["listed_days"].to_numpy())
        for key, v in (("liq", liq), ("age", age)):
            scal[key] = ((v - v[rows].mean()) / v[rows].std()).astype(np.float32)[:, None]
    return FoldData.of(data, mu, sd, scal, vol)


def target_of(data: m12.Data, icfg: I10Cfg, vol: np.ndarray | None) -> np.ndarray:
    y = data.target(icfg.base)
    if icfg.target_offset is None and vol is None:
        return y  # m12's array, untouched
    if icfg.target_offset is not None:
        off = np.load(CACHE / icfg.target_offset, mmap_mode="r")
        if off.shape != y.shape:
            raise ValueError(f"{icfg.target_offset}: shape {off.shape}, target {y.shape}")
        y = y - np.asarray(off, dtype=np.float32)
    if vol is not None:
        y = y / (100 * vol)[:, None]
    return np.ascontiguousarray(y, dtype=np.float32)


def heads(o3: np.ndarray) -> dict[str, np.ndarray]:
    """Score heads of a (rows, steps, quantiles) forecast in percent."""
    med2 = o3[:, STEP_2H, 1]
    band = np.maximum(o3[:, STEP_2H, 2] - o3[:, STEP_2H, 0], RA_FLOOR)
    return {"h1": o3[:, STEP_1H, 1], "h2": med2, "h2ra": med2 / band}


def to_path(out: np.ndarray, rows: np.ndarray, vol: np.ndarray | None) -> np.ndarray:
    """Network output -> (rows, steps, quantiles) in percent (undoing the vol scaling)."""
    o3 = out.reshape(len(rows), len(m12.PATH_STEPS), len(m12.QUANTILES))
    if vol is not None:
        o3 = o3 * (100 * vol[rows])[:, None, None]
    return o3


def top_mean(score: np.ndarray, y: np.ndarray, frac: float) -> float:
    """Mean of `y` over the rows with the top `frac` of `score`."""
    k = max(1, round(len(score) * frac))
    return float(y[np.argsort(-score, kind="stable")[:k]].mean())


# ----------------------------------------------------------------------------- training


def train_rows(data: m12.Data, icfg: I10Cfg, cut: datetime) -> np.ndarray:
    t = data.meta["open_time"]
    uni = data.crash if icfg.base.universe == "crash" else np.ones(len(t), dtype=bool)
    ok = (t < cut).to_numpy() & uni
    if icfg.window_years is not None:
        start = cut.replace(year=cut.year - icfg.window_years)
        ok &= (t >= start).to_numpy()
    return np.flatnonzero(ok)


def fit(
    data: m12.Data,
    icfg: I10Cfg,
    y: np.ndarray,
    w: np.ndarray | None,
    tr: np.ndarray,
    va: np.ndarray,
    vol: np.ndarray | None,
    seed: int,
    fold: str,
    sealed: bool = False,
) -> tuple[nn.Module, list[dict[str, object]]]:
    """m12's `fit_fold` with the harness options. The order of the RNG calls is m12's.

    `sealed`: the validation rows reach into 2025-26, so the top-2% mean return is only logged
    (to the gitignored log CSV), never printed."""
    cfg = icfg.base
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_out = len(m12.PATH_STEPS) * len(m12.QUANTILES)
    net = m12.make(cfg, sum(data.scal[k].shape[1] for k in cfg.scalars), n_out).to(m12.DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    yt = torch.from_numpy(y)
    wt = None if w is None else torch.from_numpy(w)
    y1h = np.clip(data.meta["y1h"].to_numpy(), -0.2, 0.2)
    y_hold = np.clip(data.meta[icfg.hold].to_numpy(), -0.2, 0.2)
    epochs = icfg.fixed_epochs if icfg.select == "fixed" else cfg.epochs
    best, best_state, bad = -np.inf, None, 0
    log: list[dict[str, object]] = []
    for epoch in range(epochs):
        net.train()
        t0 = time.time()
        perm = rng.permutation(tr)
        for k in range(0, len(perm), m12.BATCH):
            b = np.sort(perm[k : k + m12.BATCH])
            if len(b) < 32:
                continue
            x, s = data.batch(b, cfg)
            wb = None if wt is None else wt[b].to(m12.DEVICE)
            loss = batch_loss(cfg, net(x, s), yt[b].to(m12.DEVICE), wb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        hv = heads(to_path(m12.predict(net, data, cfg, va), va, vol))
        corr = float(np.corrcoef(hv["h1"], y1h[va])[0, 1]) if len(va) > 10 else 0.0
        top = top_mean(hv[icfg.select_head], y_hold[va], icfg.top_frac) if len(va) else 0.0
        metric = top if icfg.select == "top" else corr
        log.append(
            {
                "fold": fold,
                "seed": seed,
                "epoch": epoch,
                "select": icfg.select,
                "metric": metric,
                "valid_corr": corr,
                "valid_top": top,
                "secs": time.time() - t0,
            }
        )
        print(
            f"  fold {fold} seed {seed} epoch {epoch}: corr {corr:.4f}"
            + ("" if sealed else f" top {top:+.5f}")
            + f" ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if icfg.select == "fixed":
            continue
        if metric > best:
            best, bad = metric, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= icfg.patience:
                break
    if icfg.select != "fixed":
        assert best_state is not None
        net.load_state_dict(best_state)
    return net, log


def _new_file(path: Path) -> Path:
    """Refuse to write through a link or outside `cache_all/wf_i10/` (shared caches)."""
    WF_I10.mkdir(exist_ok=True)
    if WF_I10.is_symlink() or path.is_symlink() or path.resolve().parent != WF_I10.resolve():
        raise RuntimeError(f"refusing to write {path}")
    return path


def run(
    name: str,
    data: m12.Data,
    icfg: I10Cfg,
    years: tuple[int, ...] = YEARS,
    sub: int = 1,
) -> None:
    """Train `icfg` walk-forward (yearly folds) and save `wf_i10/<name>.npz` and the log.

    `sub` keeps every `sub`-th training row (smoke runs only)."""
    icfg.check()
    cfg = icfg.base
    out_npz, out_log = _new_file(WF_I10 / f"{name}.npz"), _new_file(WF_I10 / f"{name}_log.csv")
    n = len(data.meta)
    t = data.meta["open_time"]
    uni = data.crash if cfg.universe == "crash" else np.ones(n, dtype=bool)
    vol = vol_of(data, cfg.seq, icfg.vol_floor) if icfg.volscale else None
    y = target_of(data, icfg, vol)
    nan = np.float32(np.nan)
    pct = {h: np.full(n, nan, dtype=np.float32) for h in HEADS}
    raw = {h: np.full(n, nan, dtype=np.float32) for h in HEADS}
    pct_seed = {h: np.full((cfg.seeds, n), nan, dtype=np.float32) for h in HEADS}
    shape = (n, len(m12.PATH_STEPS), len(m12.QUANTILES))
    path = np.full(shape, nan, dtype=np.float16)
    path_scaled = np.full(shape, nan, dtype=np.float16) if vol is not None else None
    raw_scaled = np.full(n, nan, dtype=np.float32)
    pct_scaled = np.full(n, nan, dtype=np.float32)
    logs: list[dict[str, object]] = []
    timing: list[dict[str, object]] = []
    t_start = time.time()
    for label, test_mask, train_cut, valid_end in m12.folds_of("yearly", data, years):
        year = int(label[:4])
        valid_end = valid_end or datetime(year, 1, 1, tzinfo=UTC)
        tr_all = train_rows(data, icfg, train_cut)
        tr = tr_all[::sub]
        va = np.flatnonzero(((t >= train_cut) & (t < valid_end)).to_numpy() & uni)
        test = np.flatnonzero(test_mask & uni)
        ref = np.sort(tr_all[:: max(1, len(tr_all) // 50_000)])
        fd = fold_data(data, icfg, tr_all, vol)
        w = row_weights(icfg, data, tr_all, train_cut)
        outs_test, outs_ref = [], []
        for seed in range(cfg.seeds):
            t0 = time.time()
            sealed = valid_end > SEALED_FROM
            net, log = fit(fd, icfg, y, w, tr, va, vol, seed, label, sealed)
            logs += log
            outs_test.append(m12.predict(net, fd, cfg, test))
            outs_ref.append(m12.predict(net, fd, cfg, ref))
            timing.append({"fold": label, "seed": seed, "minutes": (time.time() - t0) / 60})
            del net
        # seed mean of the raw outputs first, as m12
        o_test, o_ref = np.mean(outs_test, 0), np.mean(outs_ref, 0)
        p_test, p_ref = to_path(o_test, test, vol), to_path(o_ref, ref, vol)
        path[test] = p_test.astype(np.float16)
        h_test, h_ref = heads(p_test), heads(p_ref)
        for h in HEADS:
            raw[h][test] = h_test[h]
            pct[h][test] = np.searchsorted(np.sort(h_ref[h]), h_test[h]) / len(h_ref[h])
        if path_scaled is not None:
            s_test, s_ref = to_path(o_test, test, None), to_path(o_ref, ref, None)
            path_scaled[test] = s_test.astype(np.float16)
            raw_scaled[test] = s_test[:, STEP_1H, 1]
            pct_scaled[test] = np.searchsorted(
                np.sort(s_ref[:, STEP_1H, 1]), s_test[:, STEP_1H, 1]
            ) / len(ref)
        for j, (ot, orf) in enumerate(zip(outs_test, outs_ref, strict=True)):
            ht, hr = heads(to_path(ot, test, vol)), heads(to_path(orf, ref, vol))
            for h in HEADS:
                pct_seed[h][j, test] = np.searchsorted(np.sort(hr[h]), ht[h]) / len(hr[h])
        print(
            f"{name} {label}: {len(tr)} train rows, {len(test)} scored,"
            f" {time.time() - t_start:.0f}s",
            flush=True,
        )
    out: dict[str, np.ndarray] = {"pct": pct["h1"], "raw": raw["h1"], "pct_seed": pct_seed["h1"]}
    for h in ("h2", "h2ra"):
        out |= {f"pct_{h}": pct[h], f"raw_{h}": raw[h], f"pct_seed_{h}": pct_seed[h]}
    out["path"] = path
    out["cfg"] = np.array(icfg.describe() + ("" if sub == 1 else f" sub {sub}"))
    if path_scaled is not None:
        out |= {"path_scaled": path_scaled, "raw_scaled": raw_scaled, "pct_scaled": pct_scaled}
    np.savez(out_npz, **out)  # type: ignore[arg-type]
    pl.DataFrame(logs).with_columns(variant=pl.lit(name)).write_csv(out_log)
    tm = pl.DataFrame(timing)
    print(f"{name}: saved {out_npz.name}; minutes per (fold, seed):")
    with pl.Config(tbl_rows=40, float_precision=1):
        print(tm)
    print(f"total {(time.time() - t_start) / 60:.1f} min", flush=True)


# ----------------------------------------------------------------------------- checks


def compare(name: str, ref_name: str) -> None:
    """Per-seed pct (and, with the same seed count, the seed mean) of `wf_i10/<name>.npz` vs
    `wf_i10/<ref_name>.npz` or else the read-only `wf/<ref_name>.npz`, on the rows both scored.
    Scores only, no returns."""
    ref = WF_I10 / f"{ref_name}.npz"
    ref = ref if ref.exists() else m12.WF / f"{ref_name}.npz"
    with np.load(WF_I10 / f"{name}.npz") as a, np.load(ref) as b:
        pa, pb = a["pct_seed"], b["pct_seed"]
        years = pl.read_parquet(CACHE / "seq2_meta.parquet", columns=["open_time"])[
            "open_time"
        ].dt.year()
        yr = years.to_numpy()
        for j in range(min(len(pa), len(pb))):
            ok = np.isfinite(pa[j]) & np.isfinite(pb[j])
            d = np.abs(pa[j][ok] - pb[j][ok])
            by = {int(y): int((ok & (yr == y)).sum()) for y in np.unique(yr[ok])}
            print(
                f"pct_seed[{j}]: {ok.sum()} rows compared {by}; max abs diff {d.max():.3g};"
                f" rows differing {(d > 0).sum()}"
            )
        if len(pa) == len(pb):
            for key in ("pct", "raw"):
                ok = np.isfinite(a[key]) & np.isfinite(b[key])
                d = np.abs(a[key][ok] - b[key][ok])
                print(f"{key}: {ok.sum()} rows, max abs diff {d.max():.3g}")


def _expect(ok: object, what: str) -> None:
    if not ok:
        raise AssertionError(f"selfcheck failed: {what}")


def selfcheck() -> None:
    """Checks of the weights, the weighted loss, the heads and the vol scaling (no training)."""
    rng = np.random.default_rng(0)
    # day weights: equal total per day, mean 1
    day = np.sort(rng.integers(0, 50, 5000))
    w = day_weights(day)
    tot = np.bincount(day, w)[np.unique(day)]
    _expect(np.allclose(tot, tot[0]) and np.isclose(w.mean(), 1.0), "day weights")
    # recency: 1 at the cut, 0.5 one half-life back, 0.25 two back
    cut = datetime(2024, 7, 1, tzinfo=UTC)
    t = np.array([_us(cut), _us(cut) - int(US_PER_YEAR), _us(cut) - int(2 * US_PER_YEAR)])
    _expect(np.allclose(recency_weights(t, _us(cut), 1.0), [1.0, 0.5, 0.25]), "recency")
    _expect(np.allclose(recency_weights(t, _us(cut), 2.0)[1], 0.5**0.5), "recency half-life")
    # tail: 1 + k on crash rows
    _expect(np.allclose(tail_weights(np.array([True, False]), 3.0), [4.0, 1.0]), "tail")
    # weighted loss: unit weights give m12's loss; weights act per row
    cfg = PATH_ALL
    out = torch.randn(64, len(m12.PATH_STEPS) * len(m12.QUANTILES))
    y = torch.randn(64, len(m12.PATH_STEPS))
    ref = m12.loss_fn(cfg, out, y)
    _expect(torch.allclose(batch_loss(cfg, out, y, torch.ones(64)), ref), "unit weights")
    wr = torch.zeros(64)
    wr[5] = 2.0
    _expect(torch.allclose(batch_loss(cfg, out, y, wr), row_loss(cfg, out, y)[5]), "row weight")
    _expect(torch.allclose(row_loss(cfg, out, y).mean(), ref), "row loss mean")
    # heads
    o3 = np.zeros((2, len(m12.PATH_STEPS), len(m12.QUANTILES)), dtype=np.float32)
    o3[:, STEP_1H, 1] = [0.5, -0.2]
    o3[:, STEP_2H] = [[-1.0, 1.0, 3.0], [0.0, 0.3, 0.0]]
    h = heads(o3)
    _expect(np.allclose(h["h1"], [0.5, -0.2]) and np.allclose(h["h2"], [1.0, 0.3]), "heads")
    _expect(np.allclose(h["h2ra"], [0.25, 0.3 / RA_FLOOR]), "h2ra and its floor")
    _expect(m12.PATH_STEPS[STEP_1H] == 11 and m12.PATH_STEPS[STEP_2H] == 23, "head steps")
    # vol scaling: the target / output round trip, and vol from a synthetic close channel
    vol = np.array([0.002, 0.01], dtype=np.float32)
    rows = np.array([0, 1])
    flat = rng.normal(size=(2, len(m12.PATH_STEPS) * len(m12.QUANTILES))).astype(np.float32)
    scaled = flat / (100 * vol)[:, None]
    _expect(np.allclose(to_path(scaled, rows, vol), to_path(flat, rows, None), atol=1e-5), "vs")
    r = rng.normal(0, 0.004, 95)
    close = np.concatenate([[0.0], np.cumsum(r)])
    close -= close[-1]  # log close relative to the last (spike) candle
    _expect(np.isclose(vol_from_close(close[None])[0], r.std()), "vol of log returns")
    # top metric
    _expect(np.isclose(top_mean(np.arange(100.0), np.arange(100.0), 0.02), 98.5), "top mean")
    # real rows: windows, weights per day, no fold row past its cut
    m = pl.read_parquet(CACHE / "seq2_meta.parquet", columns=["open_time"])
    t_us = m["open_time"].dt.epoch("us").to_numpy()
    for years in (None, 3):
        icfg = replace(I10["base3"], window_years=years)
        cut = datetime(2024, 7, 1, tzinfo=UTC)
        ok = (t_us < _us(cut)) & (t_us >= (_us(cut.replace(year=cut.year - years)) if years else 0))
        tr = np.flatnonzero(ok)
        _expect(t_us[tr].max() < _us(cut), "train rows before the cut")
        wd = day_weights(t_us[tr] // US_PER_DAY)
        per_day = np.bincount(t_us[tr] // US_PER_DAY - t_us[tr].min() // US_PER_DAY, wd)
        per_day = per_day[per_day > 0]
        _expect(np.allclose(per_day, per_day[0]) and np.isclose(wd.mean(), 1.0), "real days")
        wr_ = recency_weights(t_us[tr], _us(cut), icfg.half_life)
        _expect(wr_.max() <= 1.0 and wr_.min() > 0, "recency range")
        print(
            f"window {years}: {len(tr)} rows from"
            f" {datetime.fromtimestamp(t_us[tr].min() / 1e6, tz=UTC):%Y-%m-%d},"
            f" {len(per_day)} days, day weight {per_day[0]:.1f} each,"
            f" recency {wr_.min():.3f}..{wr_.max():.3f}"
        )
    print("selfcheck passed")


# ----------------------------------------------------------------------------- CLI


def _value(text: str) -> object:
    if text.lower() == "none":
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_run(argv: list[str]) -> tuple[str, I10Cfg, tuple[int, ...], int, str]:
    name, rest = argv[0], argv[1:]
    icfg = I10[name]
    years, sub, out = YEARS, 1, None
    overridden = False
    opts = {f.name for f in fields(I10Cfg)} - {"base", "note"}
    i = 0
    while i < len(rest):
        flag, val = rest[i], rest[i + 1]
        i += 2
        if flag == "--out":
            out = val
            continue
        overridden = True
        if flag == "--years":
            years = tuple(int(y) for y in val.split(","))
        elif flag == "--seeds":
            icfg = replace(icfg, base=replace(icfg.base, seeds=int(val)))
        elif flag == "--epochs":
            icfg = replace(icfg, base=replace(icfg.base, epochs=int(val)), fixed_epochs=int(val))
        elif flag == "--sub":
            sub = int(val)
        elif flag == "--set":
            key, _, v = val.partition("=")
            if key not in opts:
                raise SystemExit(f"unknown option {key}; options: {sorted(opts)}")
            icfg = replace(icfg, **{key: _value(v)})  # type: ignore[arg-type]
        else:
            raise SystemExit(f"unknown flag {flag}")
    if overridden and out is None:
        raise SystemExit("overrides need --out NAME (a smoke run must not replace a full run)")
    return name, icfg, years, sub, out or name


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        variant, icfg, years, sub, out = parse_run(sys.argv[2:])
        print(f"{out}: {icfg.describe()} years {years} sub {sub}", flush=True)
        run(out, m12.Data(), icfg, years, sub)
    elif cmd == "compare":
        compare(sys.argv[2], sys.argv[3])
    elif cmd == "selfcheck":
        selfcheck()
    elif cmd == "list":
        for k, v in I10.items():
            print(f"{k}: {v.note}\n  {v.describe()}")
    else:
        raise SystemExit(__doc__)
