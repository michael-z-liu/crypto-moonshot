"""M4: the walk-forward training harness and the base CNN (docs/plans/ma-rebound.md, "M4 spec").

    uv run --group research python train.py run base                  # 5 folds x 3 seeds
    uv run --group research python train.py run path --years 2022 --seeds 1 --out smoke_path
    uv run --group research python train.py list

**Rows.** Study candidates of the variant's family (`cache/m4/rows.parquet`). Trainable rows: the
family's primary entry filled in the base scenario with a done (not censored) E1 label (limit:
buffer 0.05u; touch: the touch-market entry), plus the variant's own label (path: a forward path;
bracket: a done E3 label). Every candidate of the family is scored at its decision.

**Targets.** `ret` (base): the E1 gross return from the fill, `exit / entry - 1`, over the
candidate's u, clipped to [-5, 5], Huber loss (delta 1). `path`: the 32-step forward path of
`fwd.parquet` (already in units of u, clipped to [-5, 5]), quantile loss at 0.1 / 0.5 / 0.9,
score = the median at the last step. `bracket`: 1 if the E3 label's reason is `take_profit`,
binary cross-entropy, score = the probability.

**Folds** (Y = 2022..2026, cut = Y-1-07-01): train on trainable rows whose decision and label
end are before the cut; validate on trainable rows decided in [cut, Y-01-01) whose label ends
before Y-01-01; score every candidate decided in year Y. The label end is the E1 exit time (path:
also fill + T, the forward path's end; bracket: also the E3 exit). Normalisation statistics
(channel mean / sd over valid positions, scalar mean / sd) come from the fold's training rows.

**Training.** AdamW (lr 1e-3, weight decay 1e-4), batches of 1024, at most 12 epochs; after each
epoch the Spearman correlation between the score and the `ret` target on the validation rows;
patience 2; the best epoch's weights are restored. Seeds 0..n-1 (torch and numpy seeded).

**Scores.** Per seed: the percentile of the raw score among that seed's scores on the fold's
training rows (fraction strictly below). Seed mean: the percentile of the mean raw score over
the seeds among the training rows' mean raw scores. Per-timeframe-group variants do all of this
per group. Output in `cache/m4/scores/<out>/`: `seed<i>.parquet` and `mean.parquet` (`cid`,
`score`, `raw`, `fold`), `folds.npz` (each fold's row sets), `run.json`, and `log.csv` (one row
per fold, group, seed and epoch) for folds 2022-2025.

**Holdout.** Fold 2026 validates on 2025 H2. Nothing of it is printed: its log goes to
`results/sealed/m4/<run id>-fold2026.csv` and the console says only "fold 2026 done".

**Options** (one registry, `VARIANTS`): channel groups (`coin`, `btc`, `breadth`, `otherma`),
window lengths, target, pooled or per timeframe group, family. Every variant is a named config.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from functools import cache

import build_inputs as bi
import numpy as np
import numpy.typing as npt
import polars as pl
import torch
from common import CACHE, SEALED, TIMEFRAMES
from torch import nn

type F64 = npt.NDArray[np.float64]
type F32 = npt.NDArray[np.float32]
type F16 = npt.NDArray[np.float16]
type I64 = npt.NDArray[np.int64]
type Bools = npt.NDArray[np.bool_]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SCORES = bi.M4 / "scores"
M3 = CACHE / "m3"
YEARS = (2022, 2023, 2024, 2025, 2026)
SEALED_YEAR = 2026  # validates on 2025 H2: nothing of it is printed
BATCH = 1024
PRED_BATCH = 8192
CLIP_Y = 5.0
HUBER_DELTA = 1.0
QUANTILES = (0.1, 0.5, 0.9)
FWD_STEPS = 32
PRELOAD_BYTES = 7 * 2**30  # above this the batches read the memory maps directly
TF_GROUPS = {"1h+4h": ("1h", "4h"), "1d": ("1d",), "1w+1M": ("1w", "1M")}
US = 1_000_000

# channel groups -> channel names (build_inputs.FINE_CH / COARSE_CH)
FINE_GROUPS = {
    "coin": ("open", "high", "low", "close", "logqv", "taker", "mask"),
    "btc": ("btc_open", "btc_high", "btc_low", "btc_close", "btc_logqv"),
    "breadth": ("mkt_index", "mkt_above"),
}
COARSE_GROUPS = {
    "coin": ("open", "high", "low", "close", "level", "logqv", "mask", "level_mask"),
    "btc": ("btc_open", "btc_high", "btc_low", "btc_close", "btc_logqv", "btc_mask"),
    "breadth": ("mkt_index", "mkt_above", "mkt_mask"),
}
MASK_CH = {"mask", "level_mask", "btc_mask", "mkt_mask"}
VOLUME_CH = {"logqv", "btc_logqv"}  # minus the median over the window's valid positions
FINE_MASK_OF = {c: "mask" for c in FINE_GROUPS["coin"] if c != "mask"}
COARSE_MASK_OF = {
    **{c: "mask" for c in ("open", "high", "low", "close", "logqv")},
    "level": "level_mask",
    **{c: "btc_mask" for c in FINE_GROUPS["btc"]},
    "mkt_index": "mkt_mask",
    "mkt_above": "mkt_mask",
}


@dataclass(frozen=True)
class Variant:
    """A trained variant (defaults: the base CNN)."""

    family: str = "limit"  # limit | touch
    groups: tuple[str, ...] = ("coin", "btc")  # + "breadth" (channels), "otherma" (scalars)
    fine: int = 96  # 5m candles
    coarse: int = 128  # bars of the MA's timeframe
    target: str = "ret"  # ret | path | bracket
    pooled: bool = True  # False: one model per timeframe group (TF_GROUPS)
    seeds: int = 3
    epochs: int = 12
    patience: int = 2
    note: str = field(default="", compare=False)

    def check(self) -> None:
        assert self.family in ("limit", "touch"), self.family
        assert self.target in ("ret", "path", "bracket"), self.target
        assert "coin" in self.groups, "the coin's own windows are always in"
        assert set(self.groups) <= {"coin", "btc", "breadth", "otherma"}, self.groups
        assert 1 <= self.fine <= bi.FINE, self.fine
        assert 1 <= self.coarse <= bi.COARSE, self.coarse

    def describe(self) -> str:
        d = asdict(self)
        d.pop("note")
        return json.dumps(d, sort_keys=True)

    @property
    def n_out(self) -> int:
        return FWD_STEPS * len(QUANTILES) if self.target == "path" else 1

    def fine_channels(self) -> list[str]:
        return [c for g in self.groups if g in FINE_GROUPS for c in FINE_GROUPS[g]]

    def coarse_channels(self) -> list[str]:
        return [c for g in self.groups if g in COARSE_GROUPS for c in COARSE_GROUPS[g]]


BASE = Variant(note="M4 base CNN: coin + BTC, 96 x 5m and 128 bars, ret_T / u, pooled, limit")
VARIANTS: dict[str, Variant] = {
    "base": BASE,
    # M5 options: each changes one thing from the base
    "coin_only": replace(BASE, groups=("coin",), note="coin windows only"),
    "breadth": replace(BASE, groups=("coin", "btc", "breadth"), note="+ market breadth"),
    "otherma": replace(BASE, groups=("coin", "btc", "otherma"), note="+ distances to other MAs"),
    "short": replace(BASE, fine=48, coarse=64, note="short windows (48 x 5m, 64 bars)"),
    "long": replace(BASE, fine=288, coarse=256, note="long windows (288 x 5m, 256 bars)"),
    "path": replace(BASE, target="path", note="32-step path quantiles, score = last median"),
    "bracket": replace(BASE, target="bracket", note="E3 take-profit probability"),
    "tfgroup": replace(BASE, pooled=False, note="one model per timeframe group"),
    "touch": replace(BASE, family="touch", note="touch-market candidates, their own E1"),
}


# --- data --------------------------------------------------------------------------------------


def _year_us(y: int, month: int = 1) -> int:
    return int(datetime(y, month, 1, tzinfo=UTC).timestamp()) * US


def _label(family: str, exit_name: str) -> pl.DataFrame:
    """The family's primary entry, base scenario, one exit: one row per candidate (`cand` =
    row of rows.parquet)."""
    key = "touch" if family == "touch" else "b0.05"
    df = pl.read_parquet(M3 / f"lab-{key}-base-{exit_name}.parquet")
    assert df["cand"].n_unique() == df.height, "one label per candidate"
    return df


@dataclass(frozen=True)
class Targets:
    """Per family row: whether trainable, the label end, the `ret` target (always: the early
    stopping metric), and the variant's training target (rows, n_target)."""

    trainable: Bools
    label_end: I64
    ret: F32
    y: F32


def targets(v: Variant, meta: pl.DataFrame, offset: int) -> Targets:
    n = meta.height
    hold = meta["tf"].replace_strict(
        {tf: int(TIMEFRAMES[tf].hold.total_seconds()) * US for tf in TIMEFRAMES},
        return_dtype=pl.Int64,
    )
    u = meta["u"].to_numpy()
    e1 = _label(v.family, "E1").filter(~pl.col("censored"))
    k = e1["cand"].to_numpy().astype(np.int64) - offset
    assert ((k >= 0) & (k < n)).all(), "labels of the family's rows"
    ok = np.zeros(n, dtype=bool)
    ok[k] = True
    end = np.full(n, np.iinfo(np.int64).max)
    end[k] = e1["exit_t"].to_numpy()
    ret = np.full(n, np.nan, dtype=np.float32)
    gross = e1["exit_price"].to_numpy() / e1["entry_price"].to_numpy() - 1
    ret[k] = np.clip(gross / u[k], -CLIP_Y, CLIP_Y)
    if v.target == "ret":
        y = ret[:, None].copy()
    elif v.target == "path":
        fwd = pl.read_parquet(CACHE / "fwd.parquet", columns=["cid", "buffer", "fwd"])
        fwd = fwd.filter(
            pl.col("buffer").is_null() if v.family == "touch" else pl.col("buffer") == 0.05
        ).drop_nulls("fwd")
        j = meta.select("cid").with_row_index("k").join(fwd, on="cid", how="inner")
        kk = j["k"].to_numpy().astype(np.int64)
        y = np.full((n, FWD_STEPS), np.nan, dtype=np.float32)
        y[kk] = np.clip(np.stack(j["fwd"].to_numpy()), -CLIP_Y, CLIP_Y)
        has = np.zeros(n, dtype=bool)
        has[kk] = True
        ok &= has
        fill = np.full(n, 0, dtype=np.int64)
        fill[k] = e1["entry_t"].to_numpy()
        end = np.where(ok, np.maximum(end, fill + hold.to_numpy()), end)
    else:
        e3 = _label(v.family, "E3").filter(~pl.col("censored"))
        k3 = e3["cand"].to_numpy().astype(np.int64) - offset
        y = np.full((n, 1), np.nan, dtype=np.float32)
        y[k3, 0] = (e3["reason"] == "take_profit").to_numpy().astype(np.float32)
        has = np.zeros(n, dtype=bool)
        has[k3] = True
        ok &= has
        end3 = np.full(n, np.iinfo(np.int64).max)
        end3[k3] = e3["exit_t"].to_numpy()
        end = np.where(ok, np.maximum(end, end3), end)
    return Targets(ok, end, ret, y)


def scalar_features(v: Variant, scal: F64, other: F64) -> tuple[F32, list[str], Bools]:
    """Model scalars (raw, before the fold's z-scores) and which columns are z-scored."""
    cols: list[F64] = []
    names: list[str] = []
    z: list[bool] = []

    def add(name: str, x: F64, scale: bool) -> None:
        cols.append(x)
        names.append(name)
        z.append(scale)

    s = {k: scal[:, i] for i, k in enumerate(bi.SCALARS)}
    for i, tf in enumerate(bi.TF_NAMES):
        add(f"tf_{tf}", (s["tf"] == i).astype(np.float64), False)
    add("log_n", np.log(s["n"]), True)
    add("distance", s["distance"], True)
    bst = s["bars_since_touch"]
    add("bars_since_touch", np.log1p(np.nan_to_num(bst, nan=0.0)), True)
    add("bars_since_touch_null", np.isnan(bst).astype(np.float64), False)
    sl = s["slope10"]
    add("slope10", np.clip(np.nan_to_num(sl, nan=0.0), -20, 20), True)
    add("slope10_null", np.isnan(sl).astype(np.float64), False)
    add("log_atr_d", np.log(s["atr_d"]), True)
    add("log_liq", np.log(s["liq"]), True)
    if "otherma" in v.groups:
        for m, (tf, n) in enumerate(bi.study_mas()):
            x = other[:, m]
            add(f"ma_{tf}{n}", np.sign(np.nan_to_num(x)) * np.log1p(np.abs(np.nan_to_num(x))), True)
            add(f"ma_{tf}{n}_null", np.isnan(x).astype(np.float64), False)
    out = np.column_stack(cols).astype(np.float32)
    assert np.isfinite(out).all(), "scalars are finite"
    return out, names, np.array(z)


def _window_median(x: F32, valid: F32) -> F32:
    """Median of each row's values over its valid positions (0 with none)."""
    m = np.where(valid > 0, x, np.nan)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(m, axis=1)
    return np.nan_to_num(med, nan=0.0).astype(np.float32)


def _slice(
    arr: F16, names: tuple[str, ...], chans: list[str], length: int, mask_of: dict[str, str]
) -> F16:
    """Selected channels and the last `length` positions, volume minus its window median."""
    sel = [names.index(c) for c in chans]
    x = np.asarray(arr[:, sel, -length:], dtype=np.float32)
    for ci, c in enumerate(chans):
        if c in VOLUME_CH:
            mname = mask_of.get(c)
            valid = (
                x[:, chans.index(mname)]
                if mname is not None and mname in chans
                else np.ones_like(x[:, ci])
            )
            x[:, ci] = (x[:, ci] - _window_median(x[:, ci], valid)[:, None]) * (valid > 0)
    return x.astype(np.float16)


class Data:
    """A family's inputs as the variant slices them, its scalars and targets."""

    def __init__(self, v: Variant, max_rows: int | None = None) -> None:
        t0 = time.perf_counter()
        v.check()
        self.v = v
        meta = bi.rows()
        fam = np.flatnonzero((meta["family"] == v.family).to_numpy())
        a, b = int(fam[0]), int(fam[-1]) + 1
        assert b - a == len(fam), "a family's rows are contiguous"
        self.offset = a
        self.meta = meta.slice(a, b - a)
        self.decision = self.meta["decision_t"].to_numpy()
        self.tf = self.meta["tf"].to_numpy()
        self.fine_ch = v.fine_channels()
        self.coarse_ch = v.coarse_channels()
        fine = np.load(bi.M4 / "fine.npy", mmap_mode="r")
        coarse = np.load(bi.M4 / "coarse.npy", mmap_mode="r")
        n = b - a
        size = n * 2 * (len(self.fine_ch) * v.fine + len(self.coarse_ch) * v.coarse)
        self.preloaded = size <= PRELOAD_BYTES
        self._fine_mm, self._coarse_mm = fine, coarse
        if self.preloaded:
            self.xf = np.empty((n, len(self.fine_ch), v.fine), dtype=np.float16)
            self.xc = np.empty((n, len(self.coarse_ch), v.coarse), dtype=np.float16)
            for k in range(0, n, 20_000):
                e = min(n, k + 20_000)
                self.xf[k:e] = self._fine(np.arange(a + k, a + e))
                self.xc[k:e] = self._coarse(np.arange(a + k, a + e))
        scal = np.load(bi.M4 / "scalars.npy", mmap_mode="r")[a:b]
        other = np.load(bi.M4 / "other_ma.npy", mmap_mode="r")[a:b]
        self.scal, self.scal_names, self.scal_z = scalar_features(
            v, np.asarray(scal), np.asarray(other)
        )
        self.t = targets(v, self.meta, a)
        print(
            f"data: {v.family} {n} rows, fine {len(self.fine_ch)}x{v.fine},"
            f" coarse {len(self.coarse_ch)}x{v.coarse}, {self.scal.shape[1]} scalars,"
            f" preloaded={self.preloaded} ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    def _fine(self, rows: I64) -> F16:
        return _slice(self._fine_mm[rows], bi.FINE_CH, self.fine_ch, self.v.fine, FINE_MASK_OF)

    def _coarse(self, rows: I64) -> F16:
        return _slice(
            self._coarse_mm[rows], bi.COARSE_CH, self.coarse_ch, self.v.coarse, COARSE_MASK_OF
        )

    def raw(self, idx: I64) -> tuple[F16, F16]:
        """Sliced windows of family rows `idx` (sorted)."""
        if self.preloaded:
            return self.xf[idx], self.xc[idx]
        return self._fine(idx + self.offset), self._coarse(idx + self.offset)


@dataclass(frozen=True)
class Norm:
    """A fold's normalisation: per channel mean / sd over valid positions of its training
    rows (masks: 0 / 1, not scaled), and the scalars' mean / sd."""

    f_mu: F32
    f_sd: F32
    c_mu: F32
    c_sd: F32
    f_mask: I64  # per channel: index of its mask channel, or -1
    c_mask: I64
    s_mu: F32
    s_sd: F32


def _mask_index(chans: list[str], mask_of: dict[str, str]) -> I64:
    return np.array(
        [chans.index(mask_of[c]) if c in mask_of and mask_of[c] in chans else -1 for c in chans],
        dtype=np.int64,
    )


def _chan_stats(x: F32, mask_idx: I64, chans: list[str]) -> tuple[F32, F32]:
    mu = np.zeros(len(chans), dtype=np.float64)
    sd = np.ones(len(chans), dtype=np.float64)
    for ci, c in enumerate(chans):
        if c in MASK_CH:
            continue
        v = x[:, ci].astype(np.float64)
        if mask_idx[ci] >= 0:
            v = v[x[:, mask_idx[ci]] > 0]
        v = v.ravel()
        if len(v):
            mu[ci] = v.mean()
            sd[ci] = v.std() + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def fold_norm(data: Data, tr: I64) -> Norm:
    sample = np.sort(tr[:: max(1, len(tr) // 40_000)])
    xf, xc = data.raw(sample)
    fm = _mask_index(data.fine_ch, FINE_MASK_OF)
    cm = _mask_index(data.coarse_ch, COARSE_MASK_OF)
    f_mu, f_sd = _chan_stats(np.asarray(xf, dtype=np.float32), fm, data.fine_ch)
    c_mu, c_sd = _chan_stats(np.asarray(xc, dtype=np.float32), cm, data.coarse_ch)
    s = data.scal[tr].astype(np.float64)
    s_mu = np.where(data.scal_z, s.mean(0), 0.0).astype(np.float32)
    s_sd = np.where(data.scal_z, s.std(0) + 1e-6, 1.0).astype(np.float32)
    return Norm(f_mu, f_sd, c_mu, c_sd, fm, cm, s_mu, s_sd)


def _normalise(x: torch.Tensor, mu: F32, sd: F32, mask_idx: I64) -> torch.Tensor:
    m = torch.from_numpy(mu).to(x.device)[None, :, None]
    s = torch.from_numpy(sd).to(x.device)[None, :, None]
    z = torch.clamp((x - m) / s, -10, 10)
    has = mask_idx >= 0
    if has.any():
        mi = torch.from_numpy(np.where(has, mask_idx, 0)).to(x.device)
        gate = torch.where(torch.from_numpy(has).to(x.device)[None, :, None], x[:, mi], 1.0)
        z = z * gate
    return z


def batch(data: Data, norm: Norm, idx: I64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xf, xc = data.raw(idx)
    tf_ = torch.from_numpy(np.asarray(xf, dtype=np.float32)).to(DEVICE)
    tc = torch.from_numpy(np.asarray(xc, dtype=np.float32)).to(DEVICE)
    s = (data.scal[idx] - norm.s_mu) / norm.s_sd
    ts = torch.from_numpy(np.clip(s, -10, 10).astype(np.float32)).to(DEVICE)
    return (
        _normalise(tf_, norm.f_mu, norm.f_sd, norm.f_mask),
        _normalise(tc, norm.c_mu, norm.c_sd, norm.c_mask),
        ts,
    )


# --- model -------------------------------------------------------------------------------------


def conv_block(c_in: int, c_out: int, k: int) -> list[nn.Module]:
    return [nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.GELU()]


class Tower(nn.Module):
    """1D conv tower: conv blocks with pooling, then mean / max / last position."""

    def __init__(self, ch: int, a: int = 32, b: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            *conv_block(ch, a, 5),
            *conv_block(a, a, 5),
            nn.MaxPool1d(2),
            *conv_block(a, b, 5),
            nn.MaxPool1d(2),
            *conv_block(b, b, 3),
            *conv_block(b, b, 3),
        )
        self.out = 3 * b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return torch.cat([h.mean(2), h.amax(2), h[:, :, -1]], 1)


class Net(nn.Module):
    """Two towers (fine, coarse), a scalar MLP, and a head with dropout."""

    def __init__(self, ch_fine: int, ch_coarse: int, n_s: int, n_out: int) -> None:
        super().__init__()
        self.fine = Tower(ch_fine)
        self.coarse = Tower(ch_coarse)
        self.scal = nn.Sequential(nn.Linear(n_s, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(self.fine.out + self.coarse.out + 32, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_out),
        )

    def forward(self, xf: torch.Tensor, xc: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.fine(xf), self.coarse(xc), self.scal(s)], 1))


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def loss_fn(v: Variant, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if v.target == "path":
        o = out.view(len(out), FWD_STEPS, len(QUANTILES))
        q = torch.tensor(QUANTILES, device=out.device)[None, None, :]
        e = y[:, :, None] - o
        return torch.maximum(q * e, (q - 1) * e).mean()
    if v.target == "bracket":
        return nn.functional.binary_cross_entropy_with_logits(out[:, 0], y[:, 0])
    return nn.functional.huber_loss(out[:, 0], y[:, 0], delta=HUBER_DELTA)


def score_of(v: Variant, out: F32) -> F32:
    """One raw score per row: the prediction (ret), the last step's median (path), the
    probability (bracket)."""
    if v.target == "path":
        return out.reshape(len(out), FWD_STEPS, len(QUANTILES))[:, -1, 1]
    if v.target == "bracket":
        return (1 / (1 + np.exp(-out[:, 0]))).astype(np.float32)
    return out[:, 0]


def predict(net: nn.Module, data: Data, norm: Norm, idx: I64) -> F32:
    net.eval()
    outs = []
    with torch.no_grad():
        for k in range(0, len(idx), PRED_BATCH):
            xf, xc, s = batch(data, norm, idx[k : k + PRED_BATCH])
            outs.append(net(xf, xc, s).float().cpu().numpy())
    if not outs:
        return np.zeros(0, dtype=np.float32)
    return score_of(data.v, np.concatenate(outs))


def spearman(a: F32 | F64, b: F32 | F64) -> float:
    if len(a) < 10:
        return math.nan
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


# --- folds -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    year: int
    group: str  # "all" or a TF_GROUPS key
    train: I64  # family row indices
    valid: I64
    test: I64
    cut: int  # µs: Y-1-07-01
    end: int  # µs: Y-01-01


def folds(data: Data, years: tuple[int, ...]) -> list[Fold]:
    """The variant's folds: purged by each row's own label end (module docstring)."""
    out = []
    groups = {"all": tuple(TIMEFRAMES)} if data.v.pooled else TF_GROUPS
    for y in years:
        cut, end, nxt = _year_us(y - 1, 7), _year_us(y), _year_us(y + 1)
        d, le, ok = data.decision, data.t.label_end, data.t.trainable
        for g, tfs in groups.items():
            ing = np.isin(data.tf, tfs)
            tr = np.flatnonzero(ok & ing & (d < cut) & (le < cut))
            va = np.flatnonzero(ok & ing & (d >= cut) & (d < end) & (le < end))
            te = np.flatnonzero(ing & (d >= end) & (d < nxt))
            out.append(Fold(y, g, tr, va, te, cut, end))
    return out


# --- training ----------------------------------------------------------------------------------


@dataclass
class FitResult:
    net: nn.Module
    log: list[dict[str, object]]
    best_epoch: int


def fit(data: Data, norm: Norm, fold: Fold, seed: int, quiet: bool) -> FitResult:
    v = data.v
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(len(data.fine_ch), len(data.coarse_ch), data.scal.shape[1], v.n_out).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    y_all = torch.from_numpy(np.nan_to_num(data.t.y))
    ret_va = data.t.ret[fold.valid]
    best, best_state, best_epoch, bad = -math.inf, None, -1, 0
    log: list[dict[str, object]] = []
    for epoch in range(v.epochs):
        net.train()
        t0 = time.perf_counter()
        perm = rng.permutation(fold.train)
        total, nb = 0.0, 0
        for k in range(0, len(perm), BATCH):
            b = np.sort(perm[k : k + BATCH])
            if len(b) < 32 and nb > 0:
                continue
            if len(b) < 2:
                continue
            xf, xc, s = batch(data, norm, b)
            loss = loss_fn(v, net(xf, xc, s), y_all[b].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        t_train = time.perf_counter() - t0
        corr = spearman(predict(net, data, norm, fold.valid), ret_va)
        log.append(
            {
                "year": fold.year,
                "group": fold.group,
                "seed": seed,
                "epoch": epoch,
                "train_loss": total / max(nb, 1),
                "valid_spearman": corr,
                "secs": time.perf_counter() - t0,
                "train_secs": t_train,
            }
        )
        if not quiet:
            print(
                f"  {fold.year} {fold.group} seed {seed} epoch {epoch}:"
                f" loss {total / max(nb, 1):.4f}"
                f" valid spearman {corr:+.4f} ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
        metric = -math.inf if math.isnan(corr) else corr
        if best_state is None or metric > best:
            best, best_epoch, bad = metric, epoch, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= v.patience:
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    return FitResult(net, log, best_epoch)


def percentile(ref: F32, x: F32) -> F64:
    """Fraction of `ref` strictly below each x (NaN for an empty reference)."""
    if len(ref) == 0:
        return np.full(len(x), np.nan)
    return np.searchsorted(np.sort(ref), x, side="left") / len(ref)


def run(
    name: str,
    v: Variant,
    out: str,
    years: tuple[int, ...] = YEARS,
    max_train: int | None = None,
) -> str:
    """Train `v` walk-forward and write `cache/m4/scores/<out>/`. Returns the run id.
    `max_train` keeps an evenly spaced subset of each fold's training rows (smoke runs)."""
    t_start = time.perf_counter()
    run_id = f"{out}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    dest = SCORES / out
    dest.mkdir(parents=True, exist_ok=True)
    sealed_dir = SEALED / "m4"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    data = Data(v)
    n = data.meta.height
    raw = np.full((v.seeds, n), np.nan, dtype=np.float32)
    pct = np.full((v.seeds, n), np.nan)
    raw_mean = np.full(n, np.nan, dtype=np.float32)
    pct_mean = np.full(n, np.nan)
    fold_of = np.full(n, -1, dtype=np.int32)
    logs: list[dict[str, object]] = []
    sealed_logs: list[dict[str, object]] = []
    timing: list[dict[str, object]] = []
    fold_sets: dict[str, I64] = {}
    params = 0
    for fold in folds(data, years):
        sealed = fold.year >= SEALED_YEAR
        tr = fold.train
        if max_train is not None and len(tr) > max_train:
            tr = tr[np.linspace(0, len(tr) - 1, max_train).astype(np.int64)]
            fold = replace(fold, train=tr)
        key = f"{fold.year}_{fold.group}"
        for part in ("train", "valid", "test"):
            fold_sets[f"{key}_{part}"] = getattr(fold, part) + data.offset  # rows.parquet rows
        if not sealed:
            print(
                f"fold {fold.year} {fold.group}: train {len(fold.train)}, valid {len(fold.valid)},"
                f" test {len(fold.test)}",
                flush=True,
            )
        norm = fold_norm(data, fold.train)
        ref_raw, test_raw = [], []
        for seed in range(v.seeds):
            t0 = time.perf_counter()
            res = fit(data, norm, fold, seed, quiet=sealed)
            params = n_params(res.net)
            (sealed_logs if sealed else logs).extend(res.log)
            r_tr = predict(res.net, data, norm, fold.train)
            r_te = predict(res.net, data, norm, fold.test)
            ref_raw.append(r_tr)
            test_raw.append(r_te)
            raw[seed, fold.test] = r_te
            pct[seed, fold.test] = percentile(r_tr, r_te)
            mins = (time.perf_counter() - t0) / 60
            timing.append({"year": fold.year, "group": fold.group, "seed": seed, "minutes": mins})
            if not sealed:
                print(
                    f"  -> {fold.year} {fold.group} seed {seed}: best epoch {res.best_epoch},"
                    f" {mins:.1f} min",
                    flush=True,
                )
            del res
        m_tr, m_te = np.mean(ref_raw, 0), np.mean(test_raw, 0)
        raw_mean[fold.test] = m_te
        pct_mean[fold.test] = percentile(m_tr, m_te)
        fold_of[fold.test] = fold.year
        if sealed:
            print(f"fold {fold.year} {fold.group} done", flush=True)
    cid = data.meta["cid"]

    def table(score: F64, r: F32) -> pl.DataFrame:
        ok = np.isfinite(score)
        return pl.DataFrame(
            {"cid": cid.filter(ok), "score": score[ok], "raw": r[ok], "fold": fold_of[ok]}
        )

    for s in range(v.seeds):
        table(pct[s], raw[s]).write_parquet(dest / f"seed{s}.parquet")
    table(pct_mean, raw_mean).write_parquet(dest / "mean.parquet")
    np.savez(dest / "folds.npz", **fold_sets)  # type: ignore[arg-type]
    pl.DataFrame(logs).write_csv(dest / "log.csv") if logs else None
    if sealed_logs:
        pl.DataFrame(sealed_logs).write_csv(sealed_dir / f"{run_id}-fold2026.csv")
    meta = {
        "run_id": run_id,
        "variant": name,
        "config": v.describe(),
        "note": v.note,
        "years": list(years),
        "max_train": max_train,
        "params": params,
        "scalars": data.scal_names,
        "fine_channels": data.fine_ch,
        "coarse_channels": data.coarse_ch,
        "device": str(DEVICE),
        "minutes": (time.perf_counter() - t_start) / 60,
        "timing": [t for t in timing if int(str(t["year"])) < SEALED_YEAR],
        "timing_2026_minutes": sum(
            float(str(t["minutes"])) for t in timing if int(str(t["year"])) >= SEALED_YEAR
        ),
    }
    (dest / "run.json").write_text(json.dumps(meta, indent=1))
    print(f"{out}: run {run_id}, {params} parameters, {meta['minutes']:.1f} min", flush=True)
    return run_id


@cache
def load_scores(out: str, which: str) -> pl.DataFrame:
    return pl.read_parquet(SCORES / out / f"{which}.parquet")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("variant", choices=sorted(VARIANTS))
    r.add_argument("--years", default=",".join(map(str, YEARS)))
    r.add_argument("--seeds", type=int)
    r.add_argument("--epochs", type=int)
    r.add_argument("--max-train", type=int)
    r.add_argument("--out", help="output name (required with any override)")
    sub.add_parser("list")
    a = p.parse_args()
    if a.cmd == "list":
        for k, v in VARIANTS.items():
            print(f"{k}: {v.note}\n  {v.describe()}")
        return
    v = VARIANTS[a.variant]
    years = tuple(int(y) for y in a.years.split(","))
    overridden = years != YEARS or a.seeds or a.epochs or a.max_train
    if a.seeds:
        v = replace(v, seeds=a.seeds)
    if a.epochs:
        v = replace(v, epochs=a.epochs)
    if overridden and not a.out:
        raise SystemExit("overrides need --out NAME (a smoke run must not replace a full run)")
    run(a.variant, v, a.out or a.variant, years, a.max_train)


if __name__ == "__main__":
    main()
