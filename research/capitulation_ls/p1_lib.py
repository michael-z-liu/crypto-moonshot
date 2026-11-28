"""P1 (perp-only coins): shared pieces. Perp 5m klines in the spot schema, liquid volume spikes as
`paper.spike_job` finds them (seq2's definition without its look-ahead filters), the 13-channel
seq2 inputs as `build_seq2.coin_job` / `paper.build_inputs` build them, the saved day5 fold models
scored as `paper.Model` (and `forward_signals.py`) scores them, and perp price paths in the layout
of `build_perp_paths.py` ([3, 256]: 96 five-minute steps, then hourly to 7 days).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

os.environ["MOONSHOT_UNIVERSE"] = "all"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "volume_spike"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from build_events import SPIKE_MULT, WARMUP, features, full_grid  # noqa: E402
from build_seq2 import CHANNELS, LIQ, SEQ, STEP, T0, windows  # noqa: E402
from common import BARS_PER_DAY, DATA_END, PAD_BARS  # noqa: E402
from common_ls import CACHE_ALL, OWN  # noqa: E402

FUT = OWN / "futures"
P5 = FUT / "perp5m"
P1 = OWN / "p1"
P1.mkdir(parents=True, exist_ok=True)
MODELS = HERE.parents[1] / ".cache" / "forward_model"
START = datetime(2022, 1, 1, tzinfo=UTC)  # the first walk-forward fold
END = datetime(2026, 11, 1, tzinfo=UTC)  # seq2_meta's data end (exclusive)
N5, NH = 96, 160


def perp_only() -> list[str]:
    return list(json.loads((FUT / "perp_only.json").read_text())["perp_only"])


def pad(df: pl.DataFrame) -> pl.DataFrame:
    """`common.load`'s padding: PAD_BARS flat candles after an early end or a >= 1 day pause."""
    t = df["open_time"]
    step = timedelta(minutes=5)
    gap = t.diff().shift(-1)
    ends = df.filter((gap >= timedelta(days=1)) | (gap.is_null() & (t < DATA_END)))
    if ends.height == 0:
        return df
    pads = []
    for row in ends.iter_rows(named=True):
        nxt = t.filter(t > row["open_time"]).min()
        n = PAD_BARS if nxt is None else min(PAD_BARS, int((nxt - row["open_time"]) / step) - 1)  # type: ignore[operator]
        times = pl.datetime_range(
            row["open_time"] + step, row["open_time"] + step * n, step, eager=True, time_zone="UTC"
        )
        c = row["close"]
        pads.append(pl.DataFrame({"open_time": times}).with_columns(
            open=pl.lit(c), high=pl.lit(c), low=pl.lit(c), close=pl.lit(c), volume=pl.lit(0.0),
            quote_volume=pl.lit(0.0), trade_count=pl.lit(0, dtype=pl.Int64),
            taker_buy_base_volume=pl.lit(0.0), taker_buy_quote_volume=pl.lit(0.0),
        ))  # fmt: skip
    return pl.concat([df, *[p.cast(df.schema) for p in pads]]).sort("open_time")


def load_perp(sym: str) -> pl.DataFrame:
    """A perp's 5m candles in the spot kline schema (padded as `common.load` pads)."""
    p = pl.read_parquet(P5 / f"{sym}.parquet")
    df = p.select(
        open_time=pl.from_epoch("ms", "ms").dt.cast_time_unit("us").dt.replace_time_zone("UTC"),
        open=pl.col("open").cast(pl.Float64), high=pl.col("high").cast(pl.Float64),
        low=pl.col("low").cast(pl.Float64), close=pl.col("close").cast(pl.Float64),
        volume=pl.col("vol").cast(pl.Float64), quote_volume=pl.col("qv").cast(pl.Float64),
        trade_count=pl.col("n").cast(pl.Int64),
        taker_buy_base_volume=pl.col("tbv").cast(pl.Float64),
        taker_buy_quote_volume=pl.lit(0.0),
    ).sort("open_time")  # fmt: skip
    return pad(df)


def coin_grid(raw: pl.DataFrame) -> dict[str, np.ndarray]:
    """`build_seq2.grid` on a given candle frame (same fills and baselines)."""
    df = full_grid(raw).with_columns(
        pl.col("close").forward_fill(),
        pl.col("volume", "taker_buy_base_volume", "trade_count").fill_null(0),
    )
    df = df.with_columns(pl.col("open", "high", "low").fill_null(pl.col("close")))
    cols = ("open", "high", "low", "close", "volume", "taker_buy_base_volume", "trade_count")
    return {c: df[c].cast(pl.Float64).to_numpy() for c in cols}


def spikes_and_coin_channels(
    sym: str, raw: pl.DataFrame, start: datetime = START, end: datetime = END
) -> tuple[pl.DataFrame, np.ndarray] | None:
    """Liquid spikes of one coin in [start, end) (`paper.spike_job`) and their 7 coin channels
    (`build_seq2.coin_job`, float32 before the float16 cast)."""
    if raw.height < WARMUP + BARS_PER_DAY:
        return None
    df = features(full_grid(raw)).with_row_index("i")
    sel = df.filter(
        (pl.col("i") >= WARMUP)
        & pl.col("volume").is_not_null()
        & pl.col("entry").is_not_null()
        & (pl.col("vol_base") > 0)
        & (pl.col("vol_mult") >= SPIKE_MULT)
        & (pl.col("qv_base") >= LIQ)
        & (pl.col("open_time") >= start)
        & (pl.col("open_time") < end)
    )
    if sel.height == 0:
        return None
    a = coin_grid(raw)
    end_i = sel["i"].to_numpy().astype(np.int64)
    assert end_i.min() >= SEQ - 1, sym
    vbase = sel["vol_base"].to_numpy().astype(np.float64)
    tbase = sel["trades_base"].fill_null(1.0).to_numpy().astype(np.float64)
    c0 = a["close"][end_i][:, None]
    out = np.empty((len(end_i), 7, SEQ), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, c in enumerate(("open", "high", "low", "close")):
            out[:, k] = np.log(windows(a[c], end_i) / c0)
        v = windows(a["volume"], end_i)
        out[:, 4] = np.log1p(v / vbase[:, None])
        tb = windows(a["taker_buy_base_volume"], end_i)
        out[:, 5] = np.where(v > 0, tb / np.where(v > 0, v, 1.0), 0.5) - 0.5
        out[:, 6] = np.log1p(windows(a["trade_count"], end_i) / np.maximum(tbase, 1.0)[:, None])
    coin = np.nan_to_num(np.clip(out, -20, 20)).astype(np.float16)
    meta = sel.select(
        pl.lit(sym).alias("symbol"), "open_time", "qv_base", "vol_base", "vol_mult", "ret_4h",
        "ret_24h", "close", "entry",
    )  # fmt: skip
    return meta, coin


class Context:
    """BTC / ETH (spot) and market channels on the global grid, as `paper.build_inputs`."""

    def __init__(self) -> None:
        from build_seq2 import grid

        self.g = {s: grid(s) for s in ("BTCUSDT", "ETHUSDT")}
        market = pl.read_parquet(CACHE_ALL / "market.parquet")
        self.cum = np.cumsum(market["alt_r"].to_numpy())
        self.share = market["spike_share"].to_numpy()

    def fill(self, seq: np.ndarray, open_time: pl.Series) -> np.ndarray:
        """Write channels 7-12 into seq [n, 13, SEQ]; returns BTC's 4h return per row."""
        gidx = (
            pl.select(((open_time - pl.lit(T0)) / pl.lit(STEP)).cast(pl.Int64))
            .to_series()
            .to_numpy()
        )
        for ch, sym in ((7, "BTCUSDT"), (9, "ETHUSDT")):
            off, a = self.g[sym]
            e = gidx - off
            seq[:, ch] = np.log(windows(a["close"], e) / a["close"][e][:, None]).astype(np.float16)
            seq[:, ch + 1] = np.log1p(windows(a["volume"], e) / a["vol_base"][e][:, None]).astype(
                np.float16
            )
        seq[:, 11] = (windows(self.cum, gidx) - self.cum[gidx][:, None]).astype(np.float16)
        seq[:, 12] = windows(self.share, gidx).astype(np.float16)
        off, a = self.g["BTCUSDT"]
        e = gidx - off
        return np.asarray(a["close"][e] / a["close"][e - 48] - 1)


def build(sym: str, raw: pl.DataFrame, ctx: Context) -> tuple[pl.DataFrame, np.ndarray] | None:
    got = spikes_and_coin_channels(sym, raw)
    if got is None:
        return None
    meta, coin = got
    seq = np.zeros((meta.height, CHANNELS, SEQ), dtype=np.float16)
    seq[:, :7] = coin
    btc4 = ctx.fill(seq, meta["open_time"])
    return meta.with_columns(btc_ret_4h=pl.Series(btc4)), seq


class FoldModel:
    """A saved day5 fold (`.cache/forward_model/<name>`), scored as `paper.Model.score`."""

    def __init__(self, name: str) -> None:
        import m12_cnn_wf as m12
        import torch

        d = MODELS / name
        self.m12 = m12
        seeds = json.loads((d / "config.json").read_text())["seeds"]
        self.cfg = replace(m12.VARIANTS["path_all"], seeds=seeds)
        norm = np.load(d / "norm.npz")
        self.mu, self.sd = norm["mu"], norm["sd"]
        self.liq_mean, self.liq_sd = float(norm["liq_mean"]), float(norm["liq_sd"])
        self.ref = norm["ref_sorted"]
        n_out = len(m12.PATH_STEPS) * len(m12.QUANTILES)
        self.nets = []
        for j in range(seeds):
            net = m12.make(self.cfg, 1, n_out).to(m12.DEVICE)
            net.load_state_dict(torch.load(d / f"seed{j}.pt", map_location=m12.DEVICE))
            self.nets.append(net)

    def score(self, seq: np.ndarray, qv_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m12 = self.m12
        rows = np.arange(len(seq))
        if len(rows) == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        z = ((np.log(qv_base.clip(1, None)) - self.liq_mean) / self.liq_sd).astype(np.float32)
        data = _Inputs(seq, self.mu, self.sd, z)
        outs = [m12.predict(net, data, self.cfg, rows) for net in self.nets]  # type: ignore[arg-type]
        o = np.mean(outs, 0).reshape(len(rows), len(m12.PATH_STEPS), len(m12.QUANTILES))
        raw = o[:, 3, 1]
        return raw, (np.searchsorted(self.ref, raw) / len(self.ref)).astype(np.float32)


class _Inputs:
    def __init__(self, seq: np.ndarray, mu: np.ndarray, sd: np.ndarray, liq: np.ndarray) -> None:
        self.seq, self.mu, self.sd = seq, mu, sd
        self.scal = {"liq": liq[:, None]}

    def batch(self, idx: np.ndarray, cfg: Any) -> Any:
        import m12_cnn_wf as m12

        return m12.Data.batch(self, idx, cfg)  # type: ignore[arg-type]


def fold_name(year: int) -> str:
    return "day5" if year == 2026 else f"day5_{year}"


def perp_paths(sym: str, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perp paths [n, 3, 256] for spikes at open times t (epoch s; entry = open of the next 5m
    candle, which must print), as `build_perp_paths.job`. Returns (ok mask, paths of ok rows)."""
    p = pl.read_parquet(P5 / f"{sym}.parquet", columns=["ms", "open", "high", "low", "close"])
    ms = p["ms"].to_numpy() // 1000
    o, h, lo, c = (p[k].to_numpy().astype(np.float64) for k in ("open", "high", "low", "close"))
    g0 = int(ms[0])
    n = int((ms[-1] - g0) // 300) + 1
    tail = 168 * 12 + 2
    gi = ((ms - g0) // 300).astype(np.int64)
    C = np.full(n + tail, np.nan)
    C[gi] = c
    C = pl.Series(C).fill_nan(None).forward_fill().to_numpy().astype(np.float64)
    Op, Hi, Lo = C.copy(), C.copy(), C.copy()
    Op[gi], Hi[gi], Lo[gi] = o, h, lo
    printed = np.zeros(n + tail, dtype=bool)
    printed[gi] = True
    ok = np.zeros(len(t), dtype=bool)
    out = []
    for j, ti in enumerate(t):
        e = int((ti + 300 - g0) // 300)
        if e < 1 or e >= n or not printed[e]:
            continue
        base = Op[e]
        k = e + np.arange(168 * 12)
        hh, ll, cc = Hi[k] / base - 1, Lo[k] / base - 1, C[k] / base - 1
        q = np.empty((3, N5 + NH), np.float32)
        q[0, :N5], q[1, :N5], q[2, :N5] = hh[:N5], ll[:N5], cc[:N5]
        q[0, N5:] = hh[N5:].reshape(NH, 12).max(1)
        q[1, N5:] = ll[N5:].reshape(NH, 12).min(1)
        q[2, N5:] = cc[N5:].reshape(NH, 12)[:, -1]
        ok[j] = True
        out.append(q)
    return ok, (np.stack(out) if out else np.zeros((0, 3, N5 + NH), np.float32))
