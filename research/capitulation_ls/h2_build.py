"""H2 (holdout, BTC pairs): the seq2 dataset of the USDT-converted BTC pairs of 2017-08 .. 2019-12
(h2_fetch.py), built with h1_build.py's functions, which call the volume-spike builders' own:

- spikes: `h1_build.spike_job` (`build_events` selection, vol_mult >= 10, a series' first week
  skipped, 48 gap-free forward candles, qv_base >= 5,000 USDT per 5m) on each converted pair;
  a spike is kept only while its coin had no USDT pair (open time before the XXXUSDT series'
  first candle or after its last), so no coin-time is also in the USDT holdout of h1;
- the 13 CNN channels: `build_seq2.coin_job`, the BTC / ETH loop of h1_build (BTCUSDT and ETHUSDT
  from h1's holdout store) and the market channels 11-12 from `build_seq2.build_market` over the
  combined universe (h1's USDT pairs plus the converted BTC pairs: the alts that traded then);
- outcomes: `build_seq2.fwd_job` (fwd2) and `build_fwd72.job` (fwd72).

Store switch: `common.load` reads a symbol's folder from the "archive" store when it exists
there, else from the main store. h1_build points the main store at h1's holdout store; here the
archive store is pointed at holdout/klines_btc, so converted pairs come from there and BTCUSDT,
ETHUSDT and the USDT pairs from h1's store (patched at import, so spawned workers see it too).
A converted series fetched only up to the month after its coin's USDT listing ends there and
gets the usual 300 flat pad candles; kept spikes are at least that month before the end.

Writes holdout/btc/{seq2.npy, seq2_meta.parquet, fwd2.npy, fwd72.npy, market.parquet}.
Run: uv run --group research python h2_build.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import h1_build  # sets the holdout environment (and sys.path) before common is imported
from h1_build import HOLDOUT, MARKET_END, START_H, T0_H, btc_context, gidx_of, spike_job

SEQ = h1_build.SEQ

import build_fwd72  # noqa: E402
import build_seq2  # noqa: E402
import common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

BTC_STORE = HOLDOUT / "klines_btc" / "klines" / "spot" / "interval=5m"
common.ARCHIVE_5M = BTC_STORE
OUT = HOLDOUT / "btc"
SPIKES_END_B = datetime(2020, 1, 1, tzinfo=UTC)


def btc_pairs() -> list[str]:
    return sorted(p.name.removeprefix("symbol=") for p in BTC_STORE.glob("symbol=*"))


def usdt_span(symbol: str) -> tuple[datetime, datetime] | None:
    """First and last open time of the coin's USDT pair in h1's store, if it had one."""
    d = HOLDOUT / "klines" / "spot" / "interval=5m" / f"symbol={symbol.removesuffix('BTC')}USDT"
    if not d.exists():
        return None
    t = pl.read_parquet(d / "*.parquet", columns=["open_time"])["open_time"]
    lo, hi = t.min(), t.max()
    assert isinstance(lo, datetime)
    assert isinstance(hi, datetime)
    return lo, hi


def keep_free(sp: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for (sym,), g in sp.group_by(["symbol"], maintain_order=True):
        span = usdt_span(str(sym))
        if span is not None:
            t = pl.col("open_time")
            g = g.filter((t < span[0]) | (t > span[1]))
        parts.append(g)
    return pl.concat(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = btc_pairs()
    usdt = h1_build.holdout_symbols()
    assert not set(pairs) & set(usdt)
    print(len(pairs), "converted BTC pairs,", len(usdt), "USDT pairs", file=sys.stderr)
    with ProcessPoolExecutor(max_workers=8) as pool:
        frames = [f for f in pool.map(spike_job, pairs) if f is not None]
    sp = pl.concat(frames, how="vertical_relaxed").with_columns(
        pl.col("open_time").dt.convert_time_zone("UTC")
    )
    sp = sp.filter((pl.col("open_time") >= START_H) & (pl.col("open_time") < SPIKES_END_B))
    n_all = sp.height
    sp = keep_free(sp)
    print(n_all, "liquid spikes,", sp.height, "while the coin had no USDT pair", file=sys.stderr)
    sp = sp.sort("open_time", "symbol").join(
        btc_context(), on="open_time", how="left", maintain_order="left"
    )
    sp = sp.with_columns(hour=pl.col("open_time").dt.hour())
    gidx = gidx_of(sp)

    market = build_seq2.build_market(usdt + pairs, end=MARKET_END)
    market.write_parquet(OUT / "market.parquet")

    path = OUT / "seq2.npy"
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

    f2 = OUT / "fwd2.npy"
    arr = np.lib.format.open_memmap(f2, mode="w+", dtype=np.float32, shape=(sp.height, 3, 96))
    del arr
    f72 = OUT / "fwd72.npy"
    arr = np.lib.format.open_memmap(f72, mode="w+", dtype=np.float32, shape=(sp.height, 3, 72))
    del arr
    jobs2 = [(s, gidx[syms == s], np.flatnonzero(syms == s), str(f2)) for s in np.unique(syms)]
    jobs72 = [(s, gidx[syms == s], np.flatnonzero(syms == s), str(f72)) for s in np.unique(syms)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(build_seq2.fwd_job, jobs2)) == sp.height
        assert sum(pool.map(build_fwd72.job, jobs72)) == sp.height
    sp.write_parquet(OUT / "seq2_meta.parquet")
    print("written", OUT, "T0", T0_H, file=sys.stderr)


if __name__ == "__main__":
    main()
