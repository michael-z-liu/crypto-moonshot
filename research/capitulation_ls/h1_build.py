"""H1 (holdout): the seq2 dataset of 2017-08 .. 2019-12 from the holdout store written by
h1_fetch.py, built with the volume-spike builders' own functions so the definitions are the same:

- spikes and liquidity: `build_events.process` selection (vol_mult >= 10 over the previous day's
  median, a symbol's first week skipped, 48 gap-free forward candles) then `build_seq2.main`'s
  liquidity filter qv_base >= 5,000 USDT per 5m candle;
- the 13 CNN channels over 288 candles: `build_seq2.coin_job`, the BTC / ETH loop of
  `build_seq2.main`, and `build_seq2.build_market` over the holdout universe;
- outcomes: `build_seq2.fwd_job` (fwd2, 8h of 5m bars) and `build_fwd72.job` (fwd72, 72 hourly
  bars, last close held after a series ends).

`common.load` reads the holdout store (`MOONSHOT_DATA_DIR`), with the volume-spike archive store
switched off and DATA_END moved to the holdout's end (a series stopping earlier gets the usual
300 flat pad candles). The global grid starts at T0 = 2017-07-01 instead of 2020-01-01 (patched
at import so the spawned workers see it too). Spikes in 2020-01 are also built, only for a
parity check against the shared seq2 rows of the same spikes; analyses use open time < 2020.

Writes $DATA/cache/capitulation_ls/holdout/{seq2.npy, seq2_meta.parquet, fwd2.npy, fwd72.npy,
market.parquet}.

Run: uv run --group research python h1_build.py
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOLDOUT = Path("/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/holdout")
os.environ["MOONSHOT_UNIVERSE"] = "all"
os.environ["MOONSHOT_DATA_DIR"] = str(HOLDOUT)
sys.path.insert(0, str(HERE.parent / "volume_spike"))

import build_events  # noqa: E402
import build_fwd72  # noqa: E402
import build_seq2  # noqa: E402
import common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

T0_H = datetime(2017, 7, 1, tzinfo=UTC)
END_H = datetime(2020, 2, 1, tzinfo=UTC)
START_H = datetime(2017, 8, 1, tzinfo=UTC)
MARKET_END = datetime(2020, 2, 5, tzinfo=UTC)
SPIKES_END = datetime(2020, 1, 29, tzinfo=UTC)  # 2020-01 rows: parity check only
common.ARCHIVE_5M = HOLDOUT / "no_archive"  # never read the volume-spike archive store
common.DATA_END = END_H
build_seq2.T0 = T0_H
assert common.KLINES_5M == HOLDOUT / "klines" / "spot" / "interval=5m"

BPD = common.BARS_PER_DAY
STEP = build_seq2.STEP
LIQ = build_seq2.LIQ
SEQ = build_seq2.SEQ


def holdout_symbols() -> list[str]:
    return sorted(p.name.removeprefix("symbol=") for p in common.KLINES_5M.glob("symbol=*"))


def spike_job(symbol: str) -> pl.DataFrame | None:
    """`build_events.process` (spikes only) for one symbol, plus seq2's columns."""
    raw = common.load(symbol)
    if raw.height < build_events.WARMUP + BPD:
        return None
    df = build_events.features(build_events.full_grid(raw)).with_row_index("i")
    n = df.height
    sel = df.filter(
        (pl.col("i") >= build_events.WARMUP)
        & (pl.col("i") < n - common.HORIZON - 1)
        & pl.col("volume").is_not_null()
        & pl.col("entry").is_not_null()
        & (pl.col("vol_base") > 0)
        & (pl.col("vol_mult") >= build_events.SPIKE_MULT)
    )
    if sel.height == 0:
        return None
    idx = sel["i"].to_numpy()
    hi, lo, cl = (df[c].to_numpy() for c in ("high", "low", "close"))
    entry = sel["entry"].to_numpy()
    k = idx[:, None] + 1 + np.arange(common.HORIZON)[None, :]
    fwd = np.stack([hi[k] / entry[:, None] - 1, lo[k] / entry[:, None] - 1,
                    cl[k] / entry[:, None] - 1], axis=1)  # fmt: skip
    ok = ~np.isnan(fwd).any(axis=(1, 2))  # drop windows that cross an exchange gap
    sel = sel.filter(pl.Series(ok)).with_columns(
        symbol=pl.lit(symbol), listed_days=(pl.col("i") / BPD).cast(pl.Float32)
    )
    sel = sel.filter(pl.col("qv_base") >= LIQ)
    return sel.select(
        "symbol", "open_time", "qv_base", "vol_base", "trades_base", "listed_days", "ret_4h",
        "ret_24h", "vol_mult", "close", "entry",
    )  # fmt: skip


def btc_context() -> pl.DataFrame:
    df = build_events.features(build_events.full_grid(common.load("BTCUSDT")))
    return df.select(
        "open_time",
        btc_ret_4h=pl.col("ret_4h"),
        btc_ret_24h=pl.col("ret_24h"),
        btc_ret_7d=pl.col("ret_7d"),
        btc_vol_mult=pl.col("vol_mult"),
    )


def gidx_of(df: pl.DataFrame) -> np.ndarray:
    return (
        df.select(((pl.col("open_time") - pl.lit(T0_H)) / pl.lit(STEP)).cast(pl.Int64))
        .to_series()
        .to_numpy()
    )


def main() -> None:
    symbols = holdout_symbols()
    print(len(symbols), "symbols", file=sys.stderr)
    with ProcessPoolExecutor(max_workers=8) as pool:
        frames = [f for f in pool.map(spike_job, symbols) if f is not None]
    sp = pl.concat(frames, how="vertical_relaxed").with_columns(
        pl.col("open_time").dt.convert_time_zone("UTC")
    )
    sp = sp.filter((pl.col("open_time") >= START_H) & (pl.col("open_time") < SPIKES_END))
    sp = sp.sort("open_time", "symbol").join(
        btc_context(), on="open_time", how="left", maintain_order="left"
    )
    sp = sp.with_columns(hour=pl.col("open_time").dt.hour())
    gidx = gidx_of(sp)
    print(sp.height, "liquid spikes", file=sys.stderr)

    market = build_seq2.build_market(symbols, end=MARKET_END)  # room for pad candles
    market.write_parquet(HOLDOUT / "market.parquet")

    path = HOLDOUT / "seq2.npy"
    arr = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(sp.height, build_seq2.CHANNELS, SEQ)
    )
    del arr
    syms = sp["symbol"].to_numpy()
    vb = sp["vol_base"].to_numpy().astype(np.float64)
    tb = sp["trades_base"].fill_null(1.0).to_numpy().astype(np.float64)
    jobs = [(s, gidx[m], np.flatnonzero(m), str(path), vb[m], tb[m])
            for s in np.unique(syms) for m in [syms == s]]  # fmt: skip
    with ProcessPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(build_seq2.coin_job, jobs)) == sp.height
    seq = np.load(path, mmap_mode="r+")
    for ch, sym in ((7, "BTCUSDT"), (9, "ETHUSDT")):
        off, a = build_seq2.grid(sym)
        e = gidx - off
        assert e.min() >= SEQ - 1, f"{sym} history too short for the earliest spike"
        w = build_seq2.windows
        seq[:, ch] = np.log(w(a["close"], e) / a["close"][e][:, None]).astype(np.float16)
        seq[:, ch + 1] = np.log1p(w(a["volume"], e) / a["vol_base"][e][:, None]).astype(np.float16)
    cum = np.cumsum(market["alt_r"].to_numpy())
    share = market["spike_share"].to_numpy()
    seq[:, 11] = (build_seq2.windows(cum, gidx) - cum[gidx][:, None]).astype(np.float16)
    seq[:, 12] = build_seq2.windows(share, gidx).astype(np.float16)
    seq.flush()
    print("seq", seq.shape, "nan:", int(np.isnan(seq).sum()), file=sys.stderr)

    # outcomes: fwd2 (5m, 8h) and fwd72 (hourly, 72h)
    f2 = HOLDOUT / "fwd2.npy"
    arr = np.lib.format.open_memmap(f2, mode="w+", dtype=np.float32, shape=(sp.height, 3, 96))
    del arr
    f72 = HOLDOUT / "fwd72.npy"
    arr = np.lib.format.open_memmap(f72, mode="w+", dtype=np.float32, shape=(sp.height, 3, 72))
    del arr
    jobs2 = [(s, gidx[syms == s], np.flatnonzero(syms == s), str(f2)) for s in np.unique(syms)]
    jobs72 = [(s, gidx[syms == s], np.flatnonzero(syms == s), str(f72)) for s in np.unique(syms)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(build_seq2.fwd_job, jobs2)) == sp.height
        assert sum(pool.map(build_fwd72.job, jobs72)) == sp.height
    sp.write_parquet(HOLDOUT / "seq2_meta.parquet")
    print("written", HOLDOUT, file=sys.stderr)


if __name__ == "__main__":
    main()
