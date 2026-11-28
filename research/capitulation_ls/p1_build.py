"""P1 (perp-only coins): liquid 5m volume spikes of every perp-only coin (2022-01 .. 2026-10) on the
perp's own candles, with their 13-channel CNN inputs (coin channels from the perp, BTC / ETH /
market channels from the spot caches, as for spot spikes) and the perp activity gate.

Writes OWN/p1/spikes.parquet (one row per spike) and OWN/p1/seq.npy (float16 [n, 13, 288]).
Run: uv run --group research python p1_build.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from futures_ls import has_perp, spot_name
from p1_lib import CHANNELS, P1, SEQ, Context, load_perp, perp_only, spikes_and_coin_channels


def job(sym: str) -> tuple[pl.DataFrame, np.ndarray] | None:
    return spikes_and_coin_channels(sym, load_perp(sym))


def main() -> None:
    syms = perp_only()
    with ProcessPoolExecutor(8) as ex:
        got = [g for g in ex.map(job, syms) if g is not None]
    meta = pl.concat([g[0] for g in got], how="vertical_relaxed")
    n = meta.height
    print(n, "spikes on", len(got), "coins", file=sys.stderr)
    seq = np.lib.format.open_memmap(
        P1 / "seq.npy", mode="w+", dtype=np.float16, shape=(n, CHANNELS, SEQ)
    )
    k = 0
    for _, coin in got:
        seq[k : k + len(coin), :7] = coin
        k += len(coin)
    del got
    ctx = Context()
    btc4 = np.zeros(n)
    for a in range(0, n, 50_000):
        part = np.asarray(seq[a : a + 50_000])
        btc4[a : a + 50_000] = ctx.fill(part, meta["open_time"][a : a + 50_000])
        seq[a : a + 50_000] = part
    seq.flush()
    t = meta["open_time"].dt.epoch("s").to_numpy()
    spot = np.array([spot_name(s) for s in meta["symbol"].to_list()])
    meta = meta.with_columns(
        btc_ret_4h=pl.Series(btc4),
        t=pl.Series(t),
        year=pl.col("open_time").dt.year(),
        perp=pl.Series(has_perp(spot, t)),
    )
    meta.write_parquet(P1 / "spikes.parquet")
    print(meta.group_by("year").agg(n=pl.len(), coins=pl.col("symbol").n_unique(),
                                    perp=pl.col("perp").mean()).sort("year"), file=sys.stderr)  # fmt: skip                                    perp=pl.col("perp").mean()).sort("year"), file=sys.stderr)  # fmt: skip  # noqa: E501


if __name__ == "__main__":
    main()
