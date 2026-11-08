"""P1 of the CNN robustness study (docs/plans/cnn-robustness.md): forward paper test of the 2h CNN.

    uv run --group research python research/volume_spike/paper.py model    # retrain + save fold
    uv run --group research python research/volume_spike/paper.py parity   # pipeline vs shared
    uv run --group research python research/volume_spike/paper.py edge     # start-date check
    uv run --group research python research/volume_spike/paper.py update   # daily: sync + book

`update` syncs the missing candles (a new month's file is written when the month rolls over),
finds every liquid spike from 2026-11-02, builds and scores only spikes not yet in the ledger,
re-runs both books from 2026-11-02 and rewrites the ledger and the summary; rerunning it is
harmless. `--no-sync` skips the sync, `--full` rescores every spike. `check-model` redoes the
model check without retraining.

The rule (fixed in the plan before any new data was looked at): path_all walk-forward CNN, 2026
fold (trained before 2025-07-01, early-stopped on 2025 H2, percentiles of that fold's training
reference scores), liquid spikes as seq2 (>= 5k USDT qv_base), pct >= 0.98, 5 `CashCapped`
slots, hold 2h, fee 0.1% + slippage 0.05% + latency 0.05% per side, $10,000 =
`S5:path_all:d_hold2h`. Comparison: the same scores with a 1h hold (`S5:path_all:a_1h`). Signals
from 2026-11-02 00:00 UTC (the first candle after every study's data end) to the latest synced
candle.

Everything is written under `.cache/paper/` (gitignored) except the committed summary
`results/paper/summary.md` and the checks `results/paper/checks.md`:

- `data/`: a local kline store (5m candles since 2026-10-01 for the currently listed "all" coins,
  synced from Binance's public market-data host), plus copies of the shared universe lists. The
  shared data folder is never written.
- `model/`: the 2026 fold's weights (both seeds), mu/sd per channel, the liq z-score, the sorted
  reference scores and the config.
- `inputs/`: CNN inputs built for a window (parity, edge check).
- `ledger/`: signals (every liquid spike scored, with pct), trades and equity per rule.

The pipeline imports the research builders: `build_events.features` / `full_grid` (spikes and
their baselines), `build_seq2.coin_job` / `grid` / `windows` / `build_market` (the 13 channels),
`common.load` / `universe` (with `MOONSHOT_DATA_DIR` pointed at the local store), `m12_cnn_wf`
(the network, `Data.batch`, `predict`) and `s5_confirm.rules` (the two trading rules). One
deliberate difference from `build_events.process`: its look-ahead filters (48 forward candles
present and without a gap) are not applied, since a live trader cannot know them; a spike needs
only its entry candle (the next 5m open).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = Path(os.environ.get("PAPER_DIR", ROOT / ".cache" / "paper"))
SHARED_DATA = Path(os.environ.get("MOONSHOT_DATA_DIR", "/Users/mike/code/crypto-moonshot/data"))
LOCAL_DATA = Path(os.environ.get("PAPER_DATA_DIR", PAPER / "data"))
# The research modules read these at import (also in spawned workers, which inherit them).
os.environ["MOONSHOT_UNIVERSE"] = "all"
os.environ["MOONSHOT_DATA_DIR"] = str(LOCAL_DATA)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from build_events import SPIKE_MULT, WARMUP, features, full_grid  # noqa: E402
from build_seq2 import CHANNELS, LIQ, SEQ, STEP, T0, build_market, coin_job, grid, windows  # noqa: E402
from common import BARS_PER_DAY, CACHE, PEGGED, STOCKS, load, universe  # noqa: E402

from crypto_moonshot.backtest.engine import BacktestConfig, BacktestResult, run_backtest  # noqa: E402
from crypto_moonshot.backtest.source import ParquetCandleSource  # noqa: E402

MODEL = PAPER / "model"
INPUTS = PAPER / "inputs"
LEDGER = PAPER / "ledger"
LOGS = PAPER / "logs"
OUT = HERE / "results" / "paper"
SUMMARY = OUT / "summary.md"
CHECKS = OUT / "checks.md"

SYNC_START = "2026-10-01"
PAPER_START = datetime(2026, 11, 2, tzinfo=UTC)
PARITY = (datetime(2026, 10, 15, tzinfo=UTC), datetime(2026, 11, 2, tzinfo=UTC))
SHARED_END = datetime(2026, 11, 2, tzinfo=UTC)  # the shared store's data end (exclusive)
FOLD = "2026"
VARIANT = "path_all"
Q = 0.98
RULES = {"d_hold2h": "S5:path_all:d_hold2h (the paper rule)", "a_1h": "S5:path_all:a_1h (1h)"}
LISTS = ("top200.txt", "delisted.txt", "outside_pool.txt")
WORKERS = 8
INITIAL = 10_000.0


def _utc(t: Any) -> datetime:
    assert isinstance(t, datetime)
    return t if t.tzinfo is not None else t.replace(tzinfo=UTC)


# ----------------------------------------------------------------------------- universe + sync


def ensure_lists() -> None:
    """Copy the shared universe lists into the local root (read from the shared folder only)."""
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)
    for name in LISTS:
        dst = LOCAL_DATA / name
        if not dst.exists():
            shutil.copyfile(SHARED_DATA / name, dst)


def listed_symbols() -> list[str]:
    """The "all" universe's coins that were still listed at the study's data end: the pool
    (`top200.txt` minus pegged and stock tokens) plus `outside_pool.txt`. BTC and ETH are in the
    pool. Delisted coins get no new candles."""
    ensure_lists()
    top = (LOCAL_DATA / "top200.txt").read_text().split()
    pool = [s for s in top if s not in PEGGED and s not in STOCKS]
    outside = (LOCAL_DATA / "outside_pool.txt").read_text().split()
    return sorted(set(pool) | set(outside) | {"BTCUSDT", "ETHUSDT"})


def coins() -> list[str]:
    """`common.universe()` ("all") over the local store: the coins in the shared lists that have
    local candles (the listed coins, and delisted coins that traded in October 2026)."""
    ensure_lists()
    return sorted(set(universe()))


def sync(symbols: list[str]) -> dict[str, Any]:
    """Fetch the candles the local store is missing (5m, from SYNC_START to now). The month
    rolls over by itself: the sync writes whatever month files the range touches."""
    LOGS.mkdir(parents=True, exist_ok=True)
    exe = Path(sys.executable).parent / "moonshot"
    cmd = [str(exe), "data", "sync", "--data-dir", str(LOCAL_DATA), "--start", SYNC_START,
           "--progress", "none", "--json", *symbols]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (LOGS / "sync_last.err").write_text(res.stderr)
    if res.returncode not in (0, 1):  # 1: some series failed (reported), 2/3: usage or abort
        msg = f"sync failed with exit code {res.returncode}: {res.stderr[-2000:]}"
        raise RuntimeError(msg)
    out: dict[str, Any] = json.loads(res.stdout) if res.stdout.strip() else {}
    out["exit_code"] = res.returncode
    (LOGS / "sync_last.json").write_text(json.dumps(out, indent=1))
    return out


def last_open(symbol: str) -> datetime | None:
    d = LOCAL_DATA / "klines" / "spot" / "interval=5m" / f"symbol={symbol}"
    files = sorted(d.glob("*.parquet"))
    if not files:
        return None
    t = pl.read_parquet(files[-1], columns=["open_time"])["open_time"].max()
    return None if t is None else _utc(t)


def data_end() -> datetime:
    """The local data end (exclusive): the close of the last BTC and ETH candle (the earlier).
    Spikes at or after it are not built (their BTC/ETH channels would be missing)."""
    ends = [last_open(s) for s in ("BTCUSDT", "ETHUSDT")]
    if any(e is None for e in ends):
        msg = "BTCUSDT / ETHUSDT missing from the local store: run the sync first"
        raise RuntimeError(msg)
    return min(e for e in ends if e is not None) + STEP


# ----------------------------------------------------------------------------- spikes + inputs


def spike_job(symbol: str) -> tuple[str, pl.DataFrame | None]:
    """Liquid spikes of one symbol, as `build_events.process` then `build_seq2.main` select them,
    without the look-ahead filters (see the module docstring)."""
    raw = load(symbol)
    if raw.height < WARMUP + BARS_PER_DAY:
        return symbol, None
    df = features(full_grid(raw)).with_row_index("i")
    sel = df.filter(
        (pl.col("i") >= WARMUP)
        & pl.col("volume").is_not_null()
        & pl.col("entry").is_not_null()
        & (pl.col("vol_base") > 0)
        & (pl.col("vol_mult") >= SPIKE_MULT)
        & (pl.col("qv_base") >= LIQ)
    )
    if sel.height == 0:
        return symbol, None
    return symbol, sel.select(
        pl.lit(symbol).alias("symbol"),
        "open_time",
        "qv_base",
        "vol_base",
        "trades_base",
        "vol_mult",
        "ret_4h",
        "close",
        "entry",
    )


def spikes(symbols: list[str], start: datetime, end: datetime) -> pl.DataFrame:
    """Liquid spikes of `symbols` with open time in [start, end), sorted by time and symbol."""
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        frames = [f for _, f in pool.map(spike_job, symbols) if f is not None]
    if not frames:
        return pl.DataFrame()
    sp = pl.concat(frames, how="vertical_relaxed")
    sp = sp.with_columns(pl.col("open_time").dt.convert_time_zone("UTC"))
    return sp.filter((pl.col("open_time") >= start) & (pl.col("open_time") < end)).sort(
        "open_time", "symbol"
    )


def market_series(end: datetime) -> pl.DataFrame:
    """`build_seq2.build_market` over the local universe, on the global grid up to `end`'s next
    month."""
    m_end = datetime(end.year + end.month // 12, end.month % 12 + 1, 1, tzinfo=UTC)
    return build_market(coins(), end=m_end)


def build_inputs(sp: pl.DataFrame, out: Path, market: pl.DataFrame) -> np.ndarray:
    """The 13-channel seq2 tensor of the spikes `sp` (as `build_seq2.main`), memory-mapped at
    `out/seq.npy`; `out/meta.parquet` keeps the rows."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "seq.npy"
    n = sp.height
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, CHANNELS, SEQ))
    del arr
    gidx = sp.select(((pl.col("open_time") - pl.lit(T0)) / pl.lit(STEP)).cast(pl.Int64))
    gidx = gidx.to_series().to_numpy()
    sp = sp.with_columns(gidx=pl.Series(gidx))
    syms = sp["symbol"].to_numpy()
    vb = sp["vol_base"].to_numpy().astype(np.float64)
    tb = sp["trades_base"].fill_null(1.0).to_numpy().astype(np.float64)
    jobs = [(s, gidx[m], np.flatnonzero(m), str(path), vb[m], tb[m])
            for s in np.unique(syms) for m in [syms == s]]  # fmt: skip
    if n:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            assert sum(pool.map(coin_job, jobs)) == n
    seq = np.load(path, mmap_mode="r+")
    # BTC / ETH and market channels: the same code as build_seq2.main
    for ch, sym in ((7, "BTCUSDT"), (9, "ETHUSDT")):
        off, a = grid(sym)
        e = gidx - off
        seq[:, ch] = np.log(windows(a["close"], e) / a["close"][e][:, None]).astype(np.float16)
        seq[:, ch + 1] = np.log1p(windows(a["volume"], e) / a["vol_base"][e][:, None]).astype(
            np.float16
        )
    cum = np.cumsum(market["alt_r"].to_numpy())
    share = market["spike_share"].to_numpy()
    seq[:, 11] = (windows(cum, gidx) - cum[gidx][:, None]).astype(np.float16)
    seq[:, 12] = windows(share, gidx).astype(np.float16)
    seq.flush()
    sp.write_parquet(out / "meta.parquet")
    return np.load(path, mmap_mode="r")


# ----------------------------------------------------------------------------- model


def _liq_norm(meta: pl.DataFrame) -> tuple[float, float]:
    """Mean and std of log(qv_base) over rows before 2021-07, as `m12_cnn_wf.Data`."""
    early = (meta["open_time"] < datetime(2021, 7, 1, tzinfo=UTC)).to_numpy()
    liq = np.log(meta["qv_base"].to_numpy().clip(1, None))
    return float(liq[early].mean()), float(liq[early].std())


def train_model() -> None:
    """Retrain path_all's 2026 fold with m12's own `run` and save it (see `Model`)."""
    import m12_cnn_wf as m12
    import torch

    t0 = time.time()
    MODEL.mkdir(parents=True, exist_ok=True)
    data = m12.Data()
    cfg = m12.VARIANTS[VARIANT]
    saved: dict[str, np.ndarray] = {}

    def hook(label: str, c: Any, nets: list[Any], ref: np.ndarray, s_ref: np.ndarray) -> None:
        if label != FOLD:
            return
        for j, net in enumerate(nets):
            torch.save(net.state_dict(), MODEL / f"seed{j}.pt")
        saved["ref_rows"] = ref
        saved["ref_sorted"] = np.sort(s_ref)

    m12.FOLD_HOOK = hook
    m12.WF = MODEL  # run() writes <out_name>.npz and its log there
    m12.run(VARIANT, data, years=(int(FOLD),), out_name=f"{VARIANT}_{FOLD}")
    liq_mean, liq_sd = _liq_norm(data.meta)
    z = ((np.log(data.meta["qv_base"].to_numpy().clip(1, None)) - liq_mean) / liq_sd).astype(
        np.float32
    )
    assert np.array_equal(z[:, None], data.scal["liq"]), "liq z-score does not reproduce m12"
    np.savez(
        MODEL / "norm.npz",
        mu=data.mu,
        sd=data.sd,
        liq_mean=np.float64(liq_mean),
        liq_sd=np.float64(liq_sd),
        **saved,
    )
    c = asdict(cfg)
    config = {
        "variant": VARIANT,
        "fold": FOLD,
        "cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in c.items()},
        "train_before": "2025-07-01",
        "early_stop_on": "2025-07-01..2026-01-01",
        "reference": "every k-th training row (k = n_train // 50,000), seed-mean 1h median",
        "n_ref": len(saved["ref_sorted"]),
        "seeds": cfg.seeds,
        "n_out": len(m12.PATH_STEPS) * len(m12.QUANTILES),
        "score": "seed-mean output [:, 3, 1] (1h median forecast); pct = searchsorted(ref) / n",
        "trained_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seconds": round(time.time() - t0),
        "device": str(m12.DEVICE),
    }
    (MODEL / "config.json").write_text(json.dumps(config, indent=1))
    print(f"model saved in {time.time() - t0:.0f}s", flush=True)


class Model:
    """The saved 2026 fold: score seq2-style inputs into raw scores and fold percentiles."""

    def __init__(self) -> None:
        import m12_cnn_wf as m12
        import torch

        self.m12 = m12
        self.cfg = m12.VARIANTS[VARIANT]
        norm = np.load(MODEL / "norm.npz")
        self.mu, self.sd = norm["mu"], norm["sd"]
        self.liq_mean, self.liq_sd = float(norm["liq_mean"]), float(norm["liq_sd"])
        self.ref = norm["ref_sorted"]
        n_out = len(m12.PATH_STEPS) * len(m12.QUANTILES)
        self.nets = []
        for j in range(self.cfg.seeds):
            net = m12.make(self.cfg, 1, n_out).to(m12.DEVICE)
            net.load_state_dict(torch.load(MODEL / f"seed{j}.pt", map_location=m12.DEVICE))
            self.nets.append(net)

    def score(
        self, seq: np.ndarray, qv_base: np.ndarray, rows: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """(raw, pct) of `seq[rows]` (all rows by default), as `m12_cnn_wf.run` scores them."""
        m12 = self.m12
        rows = np.arange(len(seq)) if rows is None else rows
        if len(rows) == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        z = ((np.log(qv_base.clip(1, None)) - self.liq_mean) / self.liq_sd).astype(np.float32)
        data = _Inputs(seq, self.mu, self.sd, z)
        outs = [m12.predict(net, data, self.cfg, rows) for net in self.nets]  # type: ignore[arg-type]
        o = np.mean(outs, 0).reshape(len(rows), len(m12.PATH_STEPS), len(m12.QUANTILES))
        raw = o[:, 3, 1]
        # float32, as m12 stores pct
        return raw, (np.searchsorted(self.ref, raw) / len(self.ref)).astype(np.float32)


class _Inputs:
    """What `m12_cnn_wf.Data.batch` needs (seq, mu, sd, the liq scalar), for its reuse."""

    def __init__(self, seq: np.ndarray, mu: np.ndarray, sd: np.ndarray, liq: np.ndarray) -> None:
        self.seq, self.mu, self.sd = seq, mu, sd
        self.scal = {"liq": liq[:, None]}

    def batch(self, idx: np.ndarray, cfg: Any) -> Any:
        import m12_cnn_wf as m12

        return m12.Data.batch(self, idx, cfg)  # type: ignore[arg-type]


def check_model() -> list[str]:
    """The retrained fold vs `cache_all/wf/path_all.npz` on the 2026 rows, and the saved model
    (loaded from disk, scored through `Model`) vs the retrain's own output."""
    meta = pl.read_parquet(CACHE / "seq2_meta.parquet", columns=["open_time", "qv_base"])
    y2026 = np.flatnonzero((meta["open_time"].dt.year() == int(FOLD)).to_numpy())
    old = np.load(CACHE / "wf" / f"{VARIANT}.npz")
    new = np.load(MODEL / f"{VARIANT}_{FOLD}.npz")
    lines = ["## Model reproduction (2026 fold)", ""]
    for key in ("pct", "raw"):
        a, b = old[key][y2026].astype(np.float64), new[key][y2026].astype(np.float64)
        same = int(np.sum(a == b))
        lines.append(
            f"- retrain vs `path_all.npz` `{key}` on {len(y2026):,} 2026 rows: {same:,} equal"
            f" ({same / len(y2026):.4%}), max abs diff {np.max(np.abs(a - b)):.3g}"
        )
    pa, pb = old["path"][y2026].astype(np.float64), new["path"][y2026].astype(np.float64)
    lines.append(f"- `path` (all 96 quantiles): max abs diff {np.max(np.abs(pa - pb)):.3g}")
    m = Model()
    seq = np.load(CACHE / "seq2.npy", mmap_mode="r")
    raw, pct = m.score(seq, meta["qv_base"].to_numpy(), y2026)
    for key, v in (("raw", raw), ("pct", pct)):
        a = new[key][y2026].astype(np.float64)
        same = int(np.sum(a == v.astype(new[key].dtype).astype(np.float64)))
        lines.append(
            f"- saved model reloaded, `{key}` vs the retrain: {same:,} equal"
            f" ({same / len(y2026):.4%}), max abs diff {np.max(np.abs(a - v)):.3g}"
        )
    return lines


# ----------------------------------------------------------------------------- parity


def parity() -> list[str]:
    """Local pipeline vs the shared seq2 rows, inputs and scores for the PARITY window."""
    t0 = time.time()
    a, b = PARITY
    syms = coins()
    sp = spikes(syms, a, b)
    market = market_series(b)
    seq_l = build_inputs(sp, INPUTS / "parity", market)
    t_build = time.time() - t0
    shared = (
        pl.read_parquet(CACHE / "seq2_meta.parquet", columns=["symbol", "open_time", "qv_base"])
        .with_row_index("srow")
        .filter((pl.col("open_time") >= a) & (pl.col("open_time") < b))
    )
    local = sp.with_row_index("lrow").select("lrow", "symbol", "open_time", "qv_base")
    j = shared.join(local, on=["symbol", "open_time"], how="full", coalesce=True, suffix="_l")
    both = j.filter(pl.col("srow").is_not_null() & pl.col("lrow").is_not_null()).sort("srow")
    s_only = j.filter(pl.col("lrow").is_null())
    l_only = j.filter(pl.col("srow").is_null())
    seq_s = np.load(CACHE / "seq2.npy", mmap_mode="r")
    srow = both["srow"].to_numpy()
    lrow = both["lrow"].to_numpy()
    xs = np.asarray(seq_s[srow], dtype=np.float32)
    xl = np.asarray(seq_l[lrow], dtype=np.float32)
    d = np.abs(xs - xl)
    last = d[:, :, -96:]
    lines = [
        "## Pipeline parity (2026-10-15 .. 2026-11-01)",
        "",
        f"Local store: candles from {SYNC_START}; build {t_build:.0f}s.",
        "",
        f"- shared seq2 rows in the window: {shared.height:,}; local liquid spikes:"
        f" {local.height:,}; matched by (symbol, open_time): {both.height:,}"
        f" ({both.height / shared.height:.2%} of shared, {both.height / max(local.height, 1):.2%}"
        " of local)",
        f"- qv_base on matched rows: max abs diff"
        f" {np.max(np.abs(both['qv_base'] - both['qv_base_l']).to_numpy()):.3g}",
        f"- inputs, all 288 candles: {np.mean(d == 0):.4%} of values equal, max abs diff"
        f" {d.max():.3g}; rows fully equal {np.mean((d == 0).all(axis=(1, 2))):.2%}",
        f"- inputs, the 96 candles the model sees: max abs diff {last.max():.3g}; rows fully"
        f" equal {np.mean((last == 0).all(axis=(1, 2))):.2%}",
        "- max abs diff by channel (96 model candles): "
        + ", ".join(f"{c}: {v:.3g}" for c, v in enumerate(last.max(axis=(0, 2)))),
    ]
    day = both["open_time"].dt.date()
    per_day = (
        pl.DataFrame({"day": day, "eq": (last == 0).all(axis=(1, 2))})
        .group_by("day")
        .agg(rows=pl.len(), equal=pl.col("eq").sum())
        .sort("day")
    )
    lines.append(
        "- rows fully equal (model candles) by day: "
        + ", ".join(
            f"{r['day']:%m-%d} {r['equal']}/{r['rows']}" for r in per_day.iter_rows(named=True)
        )
    )
    # scores: the saved model on local inputs vs path_all.npz (and vs itself on shared inputs)
    m = Model()
    raw_l, pct_l = m.score(seq_l, local["qv_base"].to_numpy(), lrow)
    meta_s = pl.read_parquet(CACHE / "seq2_meta.parquet", columns=["qv_base"])
    raw_s, pct_s = m.score(seq_s, meta_s["qv_base"].to_numpy(), srow)
    wf = np.load(CACHE / "wf" / f"{VARIANT}.npz")
    raw_wf, pct_wf = wf["raw"][srow].astype(np.float64), wf["pct"][srow].astype(np.float64)
    ok = (last == 0).all(axis=(1, 2))
    lines += [
        f"- scores, saved model on local inputs vs `path_all.npz`: raw max abs diff"
        f" {np.max(np.abs(raw_l - raw_wf)):.3g}, pct equal {np.mean(pct_l == pct_wf):.2%},"
        f" pct max abs diff {np.max(np.abs(pct_l - pct_wf)):.3g}",
        f"  - on rows with equal inputs ({ok.sum():,}): raw max abs diff"
        f" {np.max(np.abs(raw_l - raw_wf)[ok], initial=0):.3g}, pct equal"
        f" {np.mean((pct_l == pct_wf)[ok]):.2%}",
        f"  - same model on the shared inputs vs `path_all.npz`: raw max abs diff"
        f" {np.max(np.abs(raw_s - raw_wf)):.3g}, pct equal {np.mean(pct_s == pct_wf):.2%}",
        f"  - signals at pct >= {Q}: shared {int((pct_wf >= Q).sum())}, local"
        f" {int((pct_l >= Q).sum())}, both {int(((pct_wf >= Q) & (pct_l >= Q)).sum())}",
    ]
    lines += _explain_unmatched(s_only, l_only)
    # 11-01: the shared store's outside-pool coins end 2026-10-31 23:55 (the archive's month),
    # so the shared market channels count fewer coins on 11-01 than the local store does
    shared_m = pl.read_parquet(CACHE / "market.parquet")
    ks = [int((t - T0) / STEP) for t in (datetime(2026, 10, 31, 12, tzinfo=UTC),
                                          datetime(2026, 11, 1, 12, tzinfo=UTC))]  # fmt: skip
    cs, cl = shared_m["coins"].to_numpy(), market["coins"].to_numpy()
    lines.append(
        f"- market channels (11, 12) count coins: shared {cs[ks[0]]:.0f} on 10-31 and"
        f" {cs[ks[1]]:.0f} on 11-01, local {cl[ks[0]]:.0f} and {cl[ks[1]]:.0f}. The shared"
        " store's outside-pool coins end on 2026-10-31 23:55, so on 11-01 the shared market"
        " channels cover the pool only; that is the only input difference, and the local one is"
        " the complete universe the paper test uses."
    )
    print("\n".join(lines), flush=True)
    return lines


def _explain_unmatched(s_only: pl.DataFrame, l_only: pl.DataFrame) -> list[str]:
    """Why rows are in one set only: the shared store's per-symbol data end and its look-ahead
    filter (48 forward candles), or coins missing locally."""
    local_syms = set(coins())
    ends: dict[str, datetime] = {}
    for s in set(s_only["symbol"].to_list()) | set(l_only["symbol"].to_list()):
        p = SHARED_DATA / "klines" / "spot" / "interval=5m" / f"symbol={s}"
        files = sorted(p.glob("*.parquet"))
        if files:
            t = pl.read_parquet(files[-1], columns=["open_time"])["open_time"].max()
            ends[s] = _utc(t)
    lines = [f"- rows only in shared: {s_only.height}"]
    if s_only.height:
        miss = s_only.filter(~pl.col("symbol").is_in(list(local_syms)))
        lines.append(
            f"  - coins not in the local store: {miss.height} rows"
            f" ({', '.join(sorted(set(miss['symbol'].to_list()))) or '-'})"
        )
        rest = s_only.filter(pl.col("symbol").is_in(list(local_syms)))
        lines.append(
            f"  - other: {rest.height} rows"
            + (
                ": "
                + ", ".join(
                    f"{r[0]} {r[1]:%m-%d %H:%M}"
                    for r in rest.select("symbol", "open_time").head(10).iter_rows()
                )
                if rest.height
                else ""
            )
        )
    lines.append(f"- rows only local: {l_only.height}")
    if l_only.height:
        horizon = timedelta(minutes=5 * 49)
        lo = l_only.with_columns(
            shared_end=pl.col("symbol").replace_strict(
                ends, default=None, return_dtype=pl.Datetime("us", "UTC")
            )
        )
        tail = lo.filter(pl.col("open_time") + horizon > pl.col("shared_end"))
        after = lo.filter(pl.col("open_time") > pl.col("shared_end"))
        lines += [
            f"  - after the coin's last shared candle (outside-pool coins end 2026-10-31 23:55 in"
            f" the shared store): {after.height}",
            f"  - within 49 candles of it (build_events drops spikes without 48 forward"
            f" candles): {tail.height - after.height}",
        ]
        other = lo.filter(pl.col("open_time") + horizon <= pl.col("shared_end"))
        lines.append(
            f"  - other (a gap in the next 48 candles, which build_events also drops):"
            f" {other.height}"
            + (
                ": "
                + ", ".join(
                    f"{r[0]} {r[1]:%m-%d %H:%M}"
                    for r in other.select("symbol", "open_time").head(10).iter_rows()
                )
                if other.height
                else ""
            )
        )
    return lines


# ----------------------------------------------------------------------------- edge check

EDGE_START = datetime(2026, 10, 22, tzinfo=UTC)


def truncated_store(root: Path, start: datetime) -> None:
    """A copy of the local store's 5m candles from `start` on (for the edge check)."""
    src = LOCAL_DATA / "klines" / "spot" / "interval=5m"
    for d in sorted(src.glob("symbol=*")):
        out = root / "klines" / "spot" / "interval=5m" / d.name
        for f in sorted(d.glob("*.parquet")):
            df = pl.read_parquet(f).filter(pl.col("open_time") >= start)
            if df.height:
                out.mkdir(parents=True, exist_ok=True)
                df.write_parquet(out / f.name)
    for name in LISTS:
        shutil.copyfile(LOCAL_DATA / name, root / name)


def edge_build(out: str) -> None:
    """Build the paper window's spikes and inputs into `out` (run in a subprocess whose
    `PAPER_DATA_DIR` is a truncated store)."""
    end = data_end()
    sp = spikes(coins(), PAPER_START, end)
    build_inputs(sp, Path(out), market_series(end))


def edge() -> list[str]:
    """Edge effects of the local store's start (2026-10-01): build the paper window's inputs from
    the full local store and from a copy that starts at EDGE_START; equal inputs mean nothing
    before EDGE_START (so nothing from the start of the store) reaches 2026-11-02 onward."""
    t0 = time.time()
    root = PAPER / "edge"
    if root.exists():
        shutil.rmtree(root)
    truncated_store(root / "data", EDGE_START)
    env = dict(os.environ, PAPER_DATA_DIR=str(root / "data"), PAPER_DIR=str(root))
    for d, e in ((INPUTS / "edge_full", None), (root / "inputs", env)):
        subprocess.run(
            [sys.executable, str(Path(__file__)), "edge-build", str(d)], check=True, env=e
        )
    a = pl.read_parquet(INPUTS / "edge_full" / "meta.parquet")
    b = pl.read_parquet(root / "inputs" / "meta.parquet")
    same_rows = a.select("symbol", "open_time").equals(b.select("symbol", "open_time"))
    lines = [
        f"## Edge effects (store from {SYNC_START} vs from {EDGE_START:%Y-%m-%d})",
        "",
        f"- spikes from {PAPER_START:%Y-%m-%d %H:%M} to the data end: {a.height} vs {b.height},"
        f" same rows: {same_rows}",
    ]
    if same_rows:
        xa = np.load(INPUTS / "edge_full" / "seq.npy").astype(np.float32)
        xb = np.load(root / "inputs" / "seq.npy").astype(np.float32)
        qv = np.abs(a["qv_base"] - b["qv_base"]).max()
        lines.append(
            f"- inputs equal: {np.array_equal(xa, xb)} (max abs diff"
            f" {np.max(np.abs(xa - xb), initial=0):.3g}); qv_base max abs diff {qv}"
        )
    lines.append(f"- {time.time() - t0:.0f}s")
    shutil.rmtree(root / "data")
    print("\n".join(lines), flush=True)
    return lines


# ----------------------------------------------------------------------------- paper book

SIGNALS = LEDGER / "signals.parquet"


def model_id() -> str:
    h = hashlib.sha1()
    for f in [*sorted(MODEL.glob("seed*.pt")), MODEL / "norm.npz"]:
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def score_signals(end: datetime, full: bool = False) -> tuple[pl.DataFrame, int]:
    """Every liquid spike in [PAPER_START, end), scored. Spikes already in the ledger (same
    model) are kept as they are: their inputs end at the spike candle, so new candles cannot
    change them. Only new spikes are built and scored (`full`: all of them). Returns the
    table and the number of rows scored now."""
    sp = spikes(coins(), PAPER_START, end)
    mid = model_id()
    old = None
    if SIGNALS.exists() and not full:
        old = pl.read_parquet(SIGNALS).filter(pl.col("model") == mid)
    new = sp if old is None else sp.join(old, on=["symbol", "open_time"], how="anti")
    if new.height:
        seq = build_inputs(new, INPUTS / "latest", market_series(end))
        raw, pct = Model().score(seq, new["qv_base"].to_numpy())
        new = new.with_columns(
            raw=pl.Series(raw.astype(np.float64)),
            pct=pl.Series(pct.astype(np.float64)),
            model=pl.lit(mid),
            scored_at=pl.lit(datetime.now(UTC)),
        )
    frames = [f for f in (old, new) if f is not None and f.height]
    if not frames:
        return pl.DataFrame(), 0
    out = pl.concat(frames, how="vertical_relaxed").sort("open_time", "symbol")
    LEDGER.mkdir(parents=True, exist_ok=True)
    out.write_parquet(SIGNALS)
    out.drop("scored_at").write_csv(LEDGER / "signals.csv")
    return out, new.height


def run_books(sig: pl.DataFrame, end: datetime) -> dict[str, BacktestResult]:
    """The S5 rules `d_hold2h` and `a_1h` (taken from `s5_confirm.rules`) on the paper signals,
    one continuous engine book each from PAPER_START. The config is `scorecard.book`'s, with
    the paper start, the local store as the candle source and hourly equity marks."""
    from s5_confirm import rules

    source = ParquetCandleSource(LOCAL_DATA, until=end)
    out = {}
    for name in RULES:
        gate, rule = rules()[name]
        assert gate is None
        frame = (
            sig.filter(pl.col("pct") >= rule.q)
            .select("symbol", "open_time", priority=-pl.col("pct"), pct="pct", raw="raw")
            if sig.height
            else pl.DataFrame(
                schema={"symbol": pl.String, "open_time": pl.Datetime("us", "UTC"),
                        "priority": pl.Float64, "pct": pl.Float64, "raw": pl.Float64}
            )
        )  # fmt: skip
        config = BacktestConfig(
            exit=rule.exit,
            entry_delay=rule.entry_delay,
            costs=rule.costs,
            sizing=rule.book_sizing(),
            max_positions=rule.slots,
            max_entries_per_day=rule.max_per_day,
            start=PAPER_START,
            end=None,
            use_fine=rule.use_fine,
            mark_every=timedelta(hours=1),
        )
        out[name] = run_backtest(frame, source, config)
    return out


def trade_table(res: BacktestResult, end: datetime) -> pl.DataFrame:
    t = res.trades
    if t.height == 0:
        return t
    return t.with_columns(
        status=pl.when((pl.col("exit_reason") == "series_end") & (pl.col("exit_time") >= end))
        .then(pl.lit("open (marked at the last close)"))
        .otherwise(pl.lit("closed"))
    )


def max_dd(eq: pl.DataFrame) -> float:
    if eq.height == 0:
        return 0.0
    e = np.concatenate(([INITIAL], eq["equity"].to_numpy()))
    return float((e / np.maximum.accumulate(e) - 1).min())


def write_ledger(books: dict[str, BacktestResult], end: datetime) -> None:
    LEDGER.mkdir(parents=True, exist_ok=True)
    for name, res in books.items():
        trade_table(res, end).write_csv(LEDGER / f"trades_{name}.csv")
        res.equity.write_csv(LEDGER / f"equity_{name}.csv")


# ----------------------------------------------------------------------------- summary


def _f(x: Any, fmt: str) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)


def summary(
    sig: pl.DataFrame,
    books: dict[str, BacktestResult],
    end: datetime,
    info: dict[str, Any],
) -> str:
    last = end - STEP
    days = pl.datetime_range(PAPER_START, last, timedelta(days=1), eager=True, time_zone="UTC")
    lines = [
        "# Paper test: the path_all CNN with a 2h hold (P1)",
        "",
        "Forward paper test of `S5:path_all:d_hold2h` on candles after every study's data end."
        " Plan: `docs/plans/cnn-robustness.md` (P1, rule fixed before any new data was looked at)."
        " Code: `research/volume_spike/paper.py`. Ledger: `.cache/paper/ledger/` (gitignored).",
        "",
        "- **Rule:** path_all walk-forward CNN, 2026 fold (trained before 2025-07-01, early-"
        "stopped on 2025 H2, percentiles of the fold's training reference scores); liquid spikes"
        " (vol >= 10x the 24h median, >= 5k USDT qv_base); pct >= 0.98; 5 `CashCapped` slots,"
        " highest pct first; enter at the next 5m open; hold 2h; fee 0.1% + slippage 0.05% +"
        " latency 0.05% per side (0.4% round trip); $10,000; production engine on 5m candles."
        " Comparison: the same with a 1h hold (`S5:path_all:a_1h`).",
        f"- **Window:** signals from {PAPER_START:%Y-%m-%d %H:%M} UTC to the latest candle,"
        f" {last:%Y-%m-%d %H:%M} UTC (data end {end:%Y-%m-%d %H:%M}).",
        f"- **Universe:** {info['coins']} coins in the local store ({info['listed']} listed"
        f' "all" coins synced daily; {info["traded_through"]} with candles to the data end).',
        f"- **Updated:** {info['updated']}; model `{info['model']}`.",
        "",
        "## Result",
        "",
        "| Rule | Trades | Open | Win | Mean bp/trade | P&L $ | Equity $ | Return | Max DD |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, label in RULES.items():
        res = books[name]
        t = trade_table(res, end)
        n = t.height
        pnl = float(t["pnl"].sum()) if n else 0.0
        n_open = int((t["status"] != "closed").sum()) if n else 0
        win = float((t["net_return"] > 0).mean()) if n else float("nan")  # type: ignore[arg-type]
        bp = float(t["net_return"].mean()) * 1e4 if n else float("nan")  # type: ignore[arg-type]
        eq = INITIAL + pnl
        lines.append(
            f"| {label} | {n} | {n_open} | {_f(win, '.0%')} | {_f(bp, '+.0f')} |"
            f" {pnl:+,.2f} | {eq:,.2f} | {pnl / INITIAL:+.2%} | {max_dd(res.equity):.2%} |"
        )
    n_sig = int((sig["pct"] >= Q).sum()) if sig.height else 0
    if n_sig == 0:
        lines += ["", f"No liquid spike reached pct >= {Q} in the window, so no trade was made."]
    lines += [
        "",
        "## Days",
        "",
        "Candidates: liquid spikes scored. Signals: pct >= 0.98. Top: the day's three highest pct.",
        "",
        "| Day (UTC) | Candidates | Signals | Top pct | 2h trades | 2h $ | 1h trades | 1h $ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    tt = {k: trade_table(v, end) for k, v in books.items()}
    for d in days:
        dd = d.date()
        s = sig.filter(pl.col("open_time").dt.date() == dd) if sig.height else sig
        top = (
            ", ".join(
                f"{r[0]} {r[1]:.3f}"
                for r in s.sort("pct", descending=True).select("symbol", "pct").head(3).iter_rows()
            )
            if s.height
            else "-"
        )
        cells = []
        for name in RULES:
            t = tt[name]
            td = t.filter(pl.col("entry_time").dt.date() == dd) if t.height else t
            cells += [str(td.height), f"{float(td['pnl'].sum()) if td.height else 0.0:+,.2f}"]
        ns = int((s["pct"] >= Q).sum()) if s.height else 0
        lines.append(f"| {dd} | {s.height} | {ns} | {top} | " + " | ".join(cells) + " |")
    for name, label in RULES.items():
        t = tt[name]
        lines += ["", f"## Trades: {label}", ""]
        if t.height == 0:
            lines.append("None.")
            continue
        lines += [
            "| Symbol | Signal | pct | Entry | Exit | Entry px | Exit px | Net | $ | Status |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in t.iter_rows(named=True):
            lines.append(
                f"| {r['symbol']} | {r['signal_time']:%m-%d %H:%M} | {r['pct']:.4f} |"
                f" {r['entry_time']:%m-%d %H:%M} | {r['exit_time']:%m-%d %H:%M} |"
                f" {r['entry_price']:.6g} | {r['exit_price']:.6g} | {r['net_return']:+.2%} |"
                f" {r['pnl']:+,.2f} | {r['status']} |"
            )
        skipped = {k: v for k, v in books[name].skipped.items() if v}
        if skipped:
            lines.append("")
            lines.append("Signals not taken: " + ", ".join(f"{k} {v}" for k, v in skipped.items()))
    lines += ["", "## Equity at 00:00 UTC (mark to market)", "", "| Time | 2h hold $ | 1h hold $ |"]
    lines.append("|---|---|---|")
    for d in [*days[1:], None]:
        cells = []
        for name in RULES:
            eq = books[name].equity
            if d is None:
                v = INITIAL + (float(tt[name]["pnl"].sum()) if tt[name].height else 0.0)
            else:
                before = eq.filter(pl.col("time") <= d) if eq.height else eq
                v = float(before["equity"][-1]) if before.height else INITIAL
            cells.append(f"{v:,.2f}")
        when = f"{last:%Y-%m-%d %H:%M} (latest)" if d is None else f"{d:%Y-%m-%d}"
        lines.append(f"| {when} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Run",
        "",
        "Daily: `uv run --group research python research/volume_spike/paper.py update`.",
        "",
        f"- sync: {info['sync']}",
        f"- spikes scored this run: {info['scored_now']} (of {sig.height} in the window)",
        "- timings: " + ", ".join(f"{k} {v:.0f}s" for k, v in info["timings"].items()),
    ]
    return "\n".join(lines) + "\n"


def update(do_sync: bool = True, full: bool = False) -> None:
    """The daily update: sync new candles, score new spikes, re-run both books from
    PAPER_START, rewrite the ledger and the summary. Idempotent."""
    timings: dict[str, float] = {}
    t0 = t = time.time()
    listed = listed_symbols()
    sync_note = "skipped"
    if do_sync:
        s = sync(listed)
        failed = [x["symbol"] for x in s.get("series", []) if x.get("status") != "ok"]
        sync_note = f"{s.get('candles_added', 0):,} candles added for {len(listed)} symbols" + (
            f"; failed: {', '.join(failed)}" if failed else ""
        )
    timings["sync"] = time.time() - t
    t = time.time()
    end = data_end()
    sig, n_new = score_signals(end, full)
    timings["spikes + inputs + scores"] = time.time() - t
    t = time.time()
    books = run_books(sig, end)
    write_ledger(books, end)
    timings["books"] = time.time() - t
    have = coins()
    through = sum(1 for c in have if (lo := last_open(c)) is not None and lo + STEP >= end)
    info = {
        "coins": len(have),
        "listed": len(listed),
        "traded_through": through,
        "updated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "model": model_id(),
        "sync": sync_note,
        "scored_now": n_new,
        "timings": timings,
    }
    timings["total"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(summary(sig, books, end, info))
    print(SUMMARY.read_text(), flush=True)


def write_checks(sections: dict[str, list[str]]) -> None:
    """Merge sections into results/paper/checks.md (one section per check, rerun replaces)."""
    OUT.mkdir(parents=True, exist_ok=True)
    store = LOGS / "checks.json"
    LOGS.mkdir(parents=True, exist_ok=True)
    old = json.loads(store.read_text()) if store.exists() else {}
    old.update(sections)
    store.write_text(json.dumps(old, indent=1))
    order = ["model", "parity", "edge"]
    body = ["# Paper test checks (P1)", "", "Written by `research/volume_spike/paper.py`.", ""]
    for k in order:
        if k in old:
            body += [*old[k], ""]
    CHECKS.write_text("\n".join(body))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "update"
    if cmd == "model":
        train_model()
        write_checks({"model": check_model()})
    elif cmd == "check-model":
        write_checks({"model": check_model()})
    elif cmd == "parity":
        write_checks({"parity": parity()})
    elif cmd == "edge":
        write_checks({"edge": edge()})
    elif cmd == "edge-build":
        edge_build(sys.argv[2])
    elif cmd == "update":
        update(do_sync="--no-sync" not in sys.argv, full="--full" in sys.argv)
    else:
        raise SystemExit(f"unknown command {cmd}")
