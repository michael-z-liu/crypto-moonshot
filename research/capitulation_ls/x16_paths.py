"""X16 data: dense 5m spot paths of every new Binance spot listing in `e12_listings.parquet`
(397 listings 2021-01 .. 2026), from the first 5m candle (= listing) for 35 days (10080 bars:
the longest entry delay, 72 h, plus the longest hold, 30 d, plus slack).

Read only from the shared 5m spot klines. Bars with no candle inside the data range are filled
flat at the previous close with zero volume; after the last candle (delisting / end of data)
the arrays hold NaN, which the simulator treats as the end of the path.

Writes $OWN/explore/x16_paths.npz: symbols, listed (epoch s), open/high/low/close/qv (n x 10080,
float64).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from common_ls import OWN

KLINES = Path("/Users/mike/code/crypto-moonshot/data/klines/spot/interval=5m")
NB = 10080
OUT = OWN / "explore" / "x16_paths.npz"


def months(t0: int) -> list[str]:
    """YYYY-MM labels of the listing month and the two after it."""
    d = np.datetime64(t0, "s").astype("datetime64[M]")
    return [str(d + np.timedelta64(k, "M")) for k in range(3)]


def path(sym: str, t0: int) -> np.ndarray:
    """(5, NB) array of open, high, low, close, quote volume on the 5m grid from t0."""
    src = KLINES / f"symbol={sym}"
    frames = [
        pl.read_parquet(
            src / f"{m}.parquet",
            columns=["open_time", "open", "high", "low", "close", "quote_volume"],
        )
        for m in months(t0)
        if (src / f"{m}.parquet").exists()
    ]
    out = np.full((5, NB), np.nan)
    if not frames:
        return out
    d = pl.concat(frames).sort("open_time")
    k = (d["open_time"].dt.epoch("s").to_numpy() - t0) // 300
    ok = (k >= 0) & (k < NB)
    k = k[ok]
    if len(k) == 0:
        return out
    vals = d.select("open", "high", "low", "close", "quote_volume").to_numpy()[ok].T
    out[:, k] = vals
    last = int(k.max())
    c = out[3, : last + 1]
    miss = ~np.isfinite(c)
    if miss.any():  # flat bars at the previous close inside the data range
        idx = np.where(~miss, np.arange(len(c)), 0)
        np.maximum.accumulate(idx, out=idx)
        cf = c[idx]
        for r in range(4):
            out[r, : last + 1][miss] = cf[miss]
        out[4, : last + 1][miss] = 0.0
    return out


def main() -> None:
    L = pl.read_parquet(OWN / "e12_listings.parquet").sort("listed")
    syms, listed, arrs = [], [], []
    for r in L.iter_rows(named=True):
        t0 = int(r["listed"].timestamp())
        a = path(r["symbol"], t0)
        if not np.isfinite(a[3, 0]):
            print("no first candle", r["symbol"])
            continue
        syms.append(r["symbol"])
        listed.append(t0)
        arrs.append(a)
    A = np.stack(arrs)  # n x 5 x NB
    np.savez_compressed(
        OUT, symbols=np.array(syms), listed=np.array(listed, dtype=np.int64),
        open=A[:, 0], high=A[:, 1], low=A[:, 2], close=A[:, 3], qv=A[:, 4],
    )  # fmt: skip
    n_full = int(np.isfinite(A[:, 3, -1]).sum())
    print(f"{len(syms)} listings, {n_full} with the full 35 days -> {OUT}")


if __name__ == "__main__":
    main()
