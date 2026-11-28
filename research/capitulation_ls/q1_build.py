"""Q1 (illiquid spikes): the volume spikes that seq2 drops for liquidity (median 5m quote volume
qv_base below LIQ = 5,000 USDT), 2022-01 .. 2026-10, built and scored with the p1 pipeline.

Bands: "b1k" = 1,000 <= qv_base < 5,000 and "b500" = 500 <= qv_base < 1,000 USDT per 5m candle.

- spikes and the 13 CNN channels: `p1_lib.spikes_and_coin_channels` (seq2's spike definition, as
  `paper.spike_job`) with its liquidity floor lowered to 500 (patched at import, so the spawned
  workers see it too), then only the rows below 5,000 are kept; BTC / ETH / market channels by
  `p1_lib.Context`, as for every other spike;
- scores: the day5 walk-forward fold of the spike's year (`p1_lib.FoldModel`, pct against that
  fold's own reference, exactly as p1_score.py). The CNN never saw a spike below 5,000 in training,
  so its liquidity input is extrapolated here.

Spot coins (`common.load`, universe "all") by default; `perp` = the perp-only coins of p1 on their
perp candles (`p1_lib.load_perp`, with p1's perp activity gate column). The 13-channel inputs are
scored in buffers and never stored (3.6M spot rows would need ~27 GB); only meta + raw + pct are
written.

Writes OWN/q1/{spot,perp}_spikes.parquet.
Run: uv run --group research python q1_build.py [spot|perp]
"""

from __future__ import annotations

import sys
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import timedelta

import numpy as np
import p1_lib
import polars as pl
from p1_lib import CHANNELS, OWN, SEQ, Context, FoldModel, fold_name, load_perp

p1_lib.LIQ = 500  # the floor used inside spikes_and_coin_channels (module global)
Q1 = OWN / "q1"
Q1.mkdir(parents=True, exist_ok=True)
LIQ_HI = 5_000
BUF = 60_000


def band_expr() -> pl.Expr:
    return pl.when(pl.col("qv_base") >= 1_000).then(pl.lit("b1k")).otherwise(pl.lit("b500"))


def job(args: tuple[str, str]) -> tuple[pl.DataFrame, np.ndarray] | None:
    """One coin: its sub-5k spikes (meta + 7 coin channels) and the entry candle's volume."""
    sym, mode = args
    if mode == "spot":
        from common import load

        raw = load(sym)
    else:
        raw = load_perp(sym)
    got = p1_lib.spikes_and_coin_channels(sym, raw)
    if got is None:
        return None
    meta, coin = got
    keep = (meta["qv_base"] < LIQ_HI).to_numpy()
    if not keep.any():
        return None
    meta, coin = meta.filter(pl.Series(keep)), coin[keep]
    nxt = raw.select(
        open_time=pl.col("open_time") - timedelta(minutes=5),
        qv_entry=pl.col("quote_volume"),
        n_entry=pl.col("trade_count"),
    )
    spk = raw.select("open_time", qv_spike=pl.col("quote_volume"))
    meta = meta.join(nxt, on="open_time", how="left", maintain_order="left").join(
        spk, on="open_time", how="left", maintain_order="left"
    )
    return meta.with_columns(band=band_expr()), coin


class Scorer:
    def __init__(self) -> None:
        self.ctx = Context()
        self.models: dict[int, FoldModel] = {}
        self.metas: list[pl.DataFrame] = []
        self.coins: list[np.ndarray] = []
        self.out: list[pl.DataFrame] = []
        self.n = 0

    def add(self, meta: pl.DataFrame, coin: np.ndarray) -> None:
        self.metas.append(meta)
        self.coins.append(coin)
        self.n += meta.height
        if self.n >= BUF:
            self.flush()

    def flush(self) -> None:
        if not self.metas:
            return
        meta = pl.concat(self.metas, how="vertical_relaxed")
        seq = np.zeros((meta.height, CHANNELS, SEQ), dtype=np.float16)
        seq[:, :7] = np.concatenate(self.coins)
        btc4 = self.ctx.fill(seq, meta["open_time"])
        yr = meta["open_time"].dt.year().to_numpy()
        qv = meta["qv_base"].to_numpy()
        raw = np.full(meta.height, np.nan, np.float32)
        pct = np.full(meta.height, np.nan, np.float32)
        finite = np.isfinite(seq).all(axis=(1, 2))
        for y in np.unique(yr):
            if int(y) not in self.models:
                self.models[int(y)] = FoldModel(fold_name(int(y)))
            r = np.flatnonzero(yr == y)
            raw[r], pct[r] = self.models[int(y)].score(seq[r], qv[r])
        pct = np.where(finite, pct, np.nan)
        self.out.append(
            meta.with_columns(
                btc_ret_4h=pl.Series(btc4),
                raw=pl.Series(raw),
                pct=pl.Series(pct),
                fold=pl.Series([fold_name(int(y)) for y in yr]),
            )
        )
        print(f"scored {sum(o.height for o in self.out):,}", file=sys.stderr, flush=True)
        self.metas, self.coins, self.n = [], [], 0


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "spot"
    if mode == "spot":
        from common import universe

        syms = universe()
    else:
        syms = p1_lib.perp_only()
    print(mode, len(syms), "symbols", file=sys.stderr, flush=True)
    sc = Scorer()
    with ProcessPoolExecutor(8) as ex:
        pending: list[Future[tuple[pl.DataFrame, np.ndarray] | None]] = []
        it = iter(syms)
        done = 0
        while True:
            while len(pending) < 12:
                s = next(it, None)
                if s is None:
                    break
                pending.append(ex.submit(job, (s, mode)))
            if not pending:
                break
            got = pending.pop(0).result()
            done += 1
            if got is not None:
                sc.add(*got)
            if done % 50 == 0:
                print(f"{done}/{len(syms)} coins", file=sys.stderr, flush=True)
    sc.flush()
    d = pl.concat(sc.out, how="vertical_relaxed").with_columns(
        t=pl.col("open_time").dt.epoch("s"),
        year=pl.col("open_time").dt.year(),
        day=pl.col("open_time").dt.date(),
    )
    if mode == "perp":
        from futures_ls import has_perp, spot_name

        spot = np.array([spot_name(s) for s in d["symbol"].to_list()])
        d = d.with_columns(perp=pl.Series(has_perp(spot, d["t"].to_numpy())))
    d.write_parquet(Q1 / f"{mode}_spikes.parquet")
    print(
        d.group_by("year", "band")
        .agg(
            n=pl.len(),
            coins=pl.col("symbol").n_unique(),
            q995=(pl.col("pct") >= 0.995).sum(),
            nan=pl.col("pct").is_nan().sum(),
        )
        .sort("year", "band"),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
