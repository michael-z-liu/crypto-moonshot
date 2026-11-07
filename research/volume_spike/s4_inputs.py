"""S4 of the CNN robustness study (docs/plans/cnn-robustness.md): inputs.

    cd research/volume_spike
    export MOONSHOT_UNIVERSE=all
    uv run --group research python build_s4_inputs.py ctx|h1|m1   # the datasets (CPU)
    uv run --group research python s4_inputs.py chain              # wait for S3, train + score
    uv run --group research python s4_inputs.py train s4_ctx ...   # one GPU training each
    uv run --group research python s4_inputs.py score s4_ctx ...   # scorecard + ledger
    uv run --group research python s4_inputs.py table              # S4 table (2022-24)

Variants (fixed before any result, see the plan's S4). Each is path_all (all 13 channels, 8h
path quantile forecast, 2 seeds, yearly walk-forward, scored by the 1h median forecast) plus:

- `s4_ctx`: the coin's age (m12's `age` scalar, log1p listed_days) and 7 context scalars:
  log tick_rel, btc_ret_4h, log btc_vol_mult, log1p of the distinct coins with a spike candidate
  in the last 60 minutes, log1p of those with ret_4h <= -10%, the coin's ret_4h rank among the
  same candle's candidates (0..1) and spike_share. With path_all's `liq` (log qv_base) these are
  the nine listed in the plan. Each is z-scored on rows before 2021-07 (as m12's scalars), NaN
  becomes 0, clipped to +-10. m12's own scalars are untouched.
- `s4_long7d`: a second conv branch on `seq_1h7d.npy` (8 channels x 168 one-hour bars).
- `s4_m1last`: a second conv branch on `seq_1m60.npy` (7 channels x 60 one-minute candles).
- `s4_inputs_all`: ctx + both branches (run only if at least two of the three above ran).

Branch inputs are z-scored per channel with the mean / sd of every 10th row before 2021-07 (as
m12 does for seq2), clipped to +-10. A branch is three conv blocks (32 filters) with one pooling
step; its mean / max / last step (96 numbers) join the trunk's features before the head.
Datasets are built by `build_s4_inputs.py` (see its docstring for the exact definitions).
Scores go to `cache_all/wf/<name>.npz`; ledger variants are `S4:<name without s4_>`.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import m12_cnn_wf as m12
import numpy as np
import polars as pl
import scorecard as sc
import torch
from build_s4_inputs import H1_CH, H1_STEPS, M1_CH, M1_STEPS
from common import CACHE
from torch import nn

PATH_ALL = m12.VARIANTS["path_all"]
CTX = ("liq", "age", "ctx")
S4_CNN = {
    "s4_ctx": replace(PATH_ALL, scalars=CTX, note="path_all + age and 7 context scalars"),
    "s4_long7d": replace(PATH_ALL, arch="cnn+h1", note="path_all + a 7-day 1h-bar branch"),
    "s4_m1last": replace(PATH_ALL, arch="cnn+m1", note="path_all + a last-hour 1m branch"),
    "s4_inputs_all": replace(
        PATH_ALL, scalars=CTX, arch="cnn+h1+m1", note="path_all + ctx + 1h and 1m branches"
    ),
}
ORDER = tuple(S4_CNN)
m12.VARIANTS |= S4_CNN

BRANCHES = {"h1": ("seq_1h7d.npy", H1_CH, H1_STEPS), "m1": ("seq_1m60.npy", M1_CH, M1_STEPS)}
BRANCH_W = 32
DATASETS = {
    "ctx": "s4_ctx.parquet",
    "h1": "seq_1h7d.npy (8 ch x 168 1h bars anchored at the spike close)",
    "m1": "seq_1m60.npy (7 ch x 60 1m candles t-55m..t+4m, mask channel)",
}
# (column, transform) of s4_ctx.parquet, in the scalar's column order
CTX_COLS = (
    ("tick_rel", "log"),
    ("btc_ret_4h", "id"),
    ("btc_vol_mult", "log"),
    ("n_cand_60m", "log1p"),
    ("n_crash_60m", "log1p"),
    ("ret4h_rank", "id"),
    ("spike_share", "id"),
)
EARLY = datetime(2021, 7, 1, tzinfo=UTC)


def branches_of(cfg: m12.Cfg) -> list[str]:
    return cfg.arch.split("+")[1:] if cfg.arch.startswith("cnn+") else []


# ----------------------------------------------------------------------------- model


class BranchCNN(nn.Module):
    """m12's CNN trunk on seq2 plus one small conv branch per extra sequence input.

    The extra sequences arrive flattened after the scalars in `s` (see `Data4.batch`)."""

    def __init__(self, cfg: m12.Cfg, n_s: int, n_out: int) -> None:
        super().__init__()
        w = cfg.width
        b = 64 * w
        self.conv = m12.CNN(len(cfg.channels), n_s, n_out, w).conv
        self.n_s = n_s
        self.shapes = [(BRANCHES[k][1], BRANCHES[k][2]) for k in branches_of(cfg)]
        c = BRANCH_W
        self.branch = nn.ModuleList(
            nn.Sequential(
                *m12.conv_block(ch, c, 5),
                *m12.conv_block(c, c, 5),
                nn.MaxPool1d(2),
                *m12.conv_block(c, c, 3),
            )
            for ch, _ in self.shapes
        )
        self.head = m12.Head(3 * b + n_s + 3 * c * len(self.shapes), n_out)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        feats = [h.mean(2), h.amax(2), h[:, :, -1], s[:, : self.n_s]]
        k = self.n_s
        for (ch, steps), br in zip(self.shapes, self.branch, strict=True):
            z = br(s[:, k : k + ch * steps].reshape(len(s), ch, steps))
            k += ch * steps
            feats += [z.mean(2), z.amax(2), z[:, :, -1]]
        return self.head(torch.cat(feats, 1))


for _cfg in S4_CNN.values():
    if branches_of(_cfg):
        m12.ARCHS[_cfg.arch] = BranchCNN


# ----------------------------------------------------------------------------- data


def ctx_scalars(early: np.ndarray, events: np.ndarray) -> np.ndarray:
    """The 7 context scalars [rows, 7], z-scored on the early rows."""
    df = pl.read_parquet(CACHE / DATASETS["ctx"])
    if not np.array_equal(df["event"].to_numpy(), events):
        raise ValueError("s4_ctx.parquet is not row-aligned with seq2_meta.parquet")
    cols = []
    for name, how in CTX_COLS:
        v = df[name].cast(pl.Float64).fill_null(np.nan).to_numpy()
        if how == "log":
            v = np.log(np.clip(v, 1e-12, None))
        elif how == "log1p":
            v = np.log1p(v)
        mu, sd = np.nanmean(v[early]), np.nanstd(v[early])
        cols.append(np.clip(np.nan_to_num((v - mu) / (sd + 1e-9)), -10, 10))
    return np.stack(cols, 1).astype(np.float32)


class Data4(m12.Data):
    """m12's data plus the S4 inputs that `cfgs` need (m12's own arrays are unchanged)."""

    def __init__(self, cfgs: list[m12.Cfg]) -> None:
        super().__init__()
        t0 = time.time()
        early = (self.meta["open_time"] < EARLY).to_numpy()
        if any("ctx" in c.scalars for c in cfgs):
            self.scal["ctx"] = ctx_scalars(early, self.meta["event"].to_numpy())
        self.br: dict[str, np.ndarray] = {}
        self.br_mu: dict[str, np.ndarray] = {}
        self.br_sd: dict[str, np.ndarray] = {}
        for k in sorted({b for c in cfgs for b in branches_of(c)}):
            arr = np.load(CACHE / BRANCHES[k][0], mmap_mode="r")
            if arr.shape != (len(self.meta), BRANCHES[k][1], BRANCHES[k][2]):
                raise ValueError(f"{BRANCHES[k][0]} has shape {arr.shape}")
            sample = np.asarray(arr[np.flatnonzero(early)[::10]], dtype=np.float32)
            self.br[k] = arr
            self.br_mu[k] = sample.mean((0, 2)).astype(np.float32)[None, :, None]
            self.br_sd[k] = (sample.std((0, 2)) + 1e-6).astype(np.float32)[None, :, None]
        print(f"S4 inputs loaded in {time.time() - t0:.0f}s: {sorted(self.br)}", flush=True)

    def batch(self, idx: np.ndarray, cfg: m12.Cfg) -> tuple[torch.Tensor, torch.Tensor]:
        x, s = super().batch(idx, cfg)
        parts = [s]
        for k in branches_of(cfg):
            z = np.asarray(self.br[k][idx], dtype=np.float32)
            z = np.clip((z - self.br_mu[k]) / self.br_sd[k], -10, 10)
            parts.append(torch.from_numpy(z.reshape(len(idx), -1)).to(m12.DEVICE))
        return x, torch.cat(parts, 1) if len(parts) > 1 else s


# ----------------------------------------------------------------------------- train / score


def train(names: list[str]) -> None:
    data = Data4([m12.VARIANTS[n] for n in names])
    for name in names:
        t0 = time.time()
        m12.run(name, data)
        print(f"{name} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def ledger_name(name: str) -> str:
    return f"S4:{name.removeprefix('s4_')}"


def source_of(name: str) -> str:
    cfg = m12.VARIANTS[name]
    d = asdict(cfg)
    d.pop("note")
    inputs = [DATASETS["ctx"]] if "ctx" in cfg.scalars else []
    inputs += [DATASETS[k] for k in branches_of(cfg)]
    return (
        f"m12_cnn_wf {name} {d} + s4_inputs {', '.join(inputs)} folds yearly"
        f" (cache_all/wf/{name}.npz)"
    )


def card_text(card: sc.Card, src: str) -> str:
    return sc.show(card) + f"\n  source: {src}"


def score(names: list[str]) -> None:
    for name in names:
        src = source_of(name)
        card = sc.scorecard(
            ledger_name(name), sc.wf_scores(name), sc.Rule(), stage="S4", source=src
        )
        text = card_text(card, src)
        print(text, flush=True)
        path = sc.OUT / "cards" / f"S4_{name.removeprefix('s4_')}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


# ----------------------------------------------------------------------------- chain

LOGS = CACHE / "logs"
S3_DONE = sc.OUT / "S3.md"
# 4.5 h after S4 started (08:59Z), moved to 15:30Z by the master after a 3 h usage-limit pause
DEADLINE = datetime(2026, 11, 7, 15, 30, tzinfo=UTC)
# minutes; path_all took 21. After the first training they are scaled by its actual / estimate.
ESTIMATE_MIN = {"s4_ctx": 23, "s4_long7d": 28, "s4_m1last": 27, "s4_inputs_all": 34}
POLL = 180


def log(msg: str) -> None:
    line = f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} {msg}"
    print(line, flush=True)
    with (LOGS / "s4_chain.log").open("a") as f:
        f.write(line + "\n")


def gpu_busy() -> bool:
    r = subprocess.run(
        ["pgrep", "-f", "python.*(m12_cnn_wf|s3_train)"], capture_output=True, check=False
    )
    return r.returncode == 0


def plan(now: datetime, todo: list[str], scale: float = 1.0) -> tuple[list[str], list[str]]:
    """Cut from the end of `todo` until the (scaled) estimates fit before the deadline."""
    keep = list(todo)
    cut = []
    while keep and now + timedelta(minutes=scale * sum(ESTIMATE_MIN[n] for n in keep)) > DEADLINE:
        cut.insert(0, keep.pop())
    return keep, cut


def chain() -> None:
    """Wait for S3 to release the GPU, then train and score the variants in order, one at a
    time, cutting in reverse order if the budget would run past the deadline."""
    LOGS.mkdir(exist_ok=True)
    log("chain start; waiting for S3")
    while not S3_DONE.exists() or gpu_busy():
        time.sleep(POLL)
    log("GPU free")
    ran: list[str] = []
    scale = 1.0
    for i, name in enumerate(ORDER):
        if name == "s4_inputs_all" and len(ran) < 2:
            log(f"cut {name}: only {len(ran)} of the three single-input variants ran")
            break
        keep, cut = plan(datetime.now(UTC), list(ORDER[i:]), scale)
        if name not in keep:
            est = [round(scale * ESTIMATE_MIN[n]) for n in keep + cut]
            for c in cut:
                log(f"cut {c}: estimates {est} min for {keep + cut} run past {DEADLINE:%H:%M}Z")
            break
        est_min = scale * ESTIMATE_MIN[name]
        log(f"start {name} (estimate {est_min:.0f} min; plan {keep}, would cut {cut})")
        t0 = time.time()
        with (LOGS / f"s4_{name}.log").open("w") as out:
            rc = subprocess.run(
                [sys.executable, __file__, "train", name], stdout=out, stderr=subprocess.STDOUT
            ).returncode
        minutes = (time.time() - t0) / 60
        log(f"end {name} rc={rc} in {minutes:.1f} min")
        if rc != 0:
            continue
        if not ran:
            scale = max(1.0, minutes / ESTIMATE_MIN[name])
        ran.append(name)
        with (LOGS / f"s4_score_{name}.log").open("w") as out:
            rc = subprocess.run(
                [sys.executable, __file__, "score", name], stdout=out, stderr=subprocess.STDOUT
            ).returncode
        log(f"scored {name} rc={rc}")
    log(f"chain end; ran {ran}")


# ----------------------------------------------------------------------------- table


def fragile_of(score: np.ndarray) -> tuple[bool, dict[int, float]]:
    trades = sc.run_book(sc.signals(score, sc.Rule()), sc.Rule()).trades
    ex = sc.ex_best_trade(trades)
    return any(v < 0 for v in ex.values()), ex


def table() -> None:
    """The S4 table (2022-24 only) from the ledger, with path_all and ens3 for reference."""
    led = pl.read_csv(sc.LEDGER, infer_schema_length=None)
    rule = sc.Rule().describe()
    rows = []
    for variant in ("path_all", "ens3", *(ledger_name(n) for n in ORDER)):
        r = led.filter((pl.col("variant") == variant) & (pl.col("rule") == rule))
        r = r.sort("timestamp").tail(1)
        if r.height == 0:
            continue
        d = r.row(0, named=True)
        bp = [d[f"bp_{y}"] for y in sc.SELECT_YEARS]
        fr = d.get("fragile")
        if fr is None:
            if variant == "ens3":
                s = np.mean(np.stack([sc.wf_scores(n) for n in ("base", "market", "path_all")]), 0)
            else:
                s = sc.wf_scores(variant if variant == "path_all" else f"s4_{variant[3:]}")
            fr, _ = fragile_of(s)
        rows.append(
            {
                "variant": variant,
                **{f"bp_{y}": d[f"bp_{y}"] for y in sc.SELECT_YEARS},
                "mean_bp": d["mean_bp"],
                "worst_bp": min(bp),
                "cost06_bp": d["cost06_bp"],
                "late_bp": d["late_bp"],
                "ex5_bp": d["ex5_bp"],
                "ci_lo": d["ci_lo_bp"],
                "ci_hi": d["ci_hi_bp"],
                "trades_yr": round(float(np.mean([d[f"trades_{y}"] for y in sc.SELECT_YEARS])), 1),
                "qualifies": d["qualifies"],
                "fragile": bool(fr),
                "reasons": d["reasons"],
            }
        )
    df = pl.DataFrame(rows)
    df.write_csv(sc.OUT / "S4_table.csv")
    with pl.Config(tbl_rows=20, tbl_cols=30, tbl_width_chars=300, float_precision=1):
        print(df.drop("reasons"))


def timings() -> None:
    """Training minutes and epochs kept (best validation epoch + 1) per (year, seed)."""
    rows = []
    for name in ("path_all", *ORDER):
        f = m12.WF / f"{name}_log.csv"
        if not f.exists():
            continue
        lg = pl.read_csv(f)
        best = lg.sort("valid_corr", descending=True).group_by("year", "seed").first()
        n_run = lg.group_by("year", "seed").agg(pl.len())
        kept = (best["epoch"] + 1).to_numpy()
        rows.append(
            {
                "variant": name,
                "fits": best.height,
                "epochs_run_mean": round(float(n_run["len"].cast(pl.Float64).mean() or 0), 2),
                "epochs_kept_mean": round(float(kept.mean()), 2),
                "epochs_kept_min": int(kept.min()),
                "epochs_kept_max": int(kept.max()),
                "epoch_min": round(float(lg["secs"].sum()) / 60, 1),
                "valid_corr_best_mean": round(float(best["valid_corr"].mean() or 0), 4),  # type: ignore[arg-type]
            }
        )
    df = pl.DataFrame(rows)
    df.write_csv(sc.OUT / "S4_timings.csv")
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=200):
        print(df)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "train":
        train(sys.argv[2:])
    elif cmd == "score":
        score(sys.argv[2:])
    elif cmd == "chain":
        chain()
    elif cmd == "table":
        table()
    elif cmd == "timings":
        timings()
    else:
        raise SystemExit(__doc__)
