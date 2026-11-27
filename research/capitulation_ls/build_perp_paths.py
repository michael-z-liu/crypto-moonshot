"""Perp price paths for every big-move event with an active perp (feats.parquet, perp == True).

Entry = the perp's open of the 5m candle after the spike candle (the same moment as the spot entry).
Path: 96 five-minute steps (8 h), then hourly steps for hours 9..168 (7 days): high, low, close
relative to the entry. After the perp's data ends the last close holds (settlement at last price).

Output: OWN/perp_paths.npz with rows (seq2 row ids) and paths float32 [n, 3, 256].
Run: uv run --group research python build_perp_paths.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from common_ls import OWN
from futures_ls import spot_name

P5 = OWN / "futures" / "perp5m"
N5, NH = 96, 160  # 5m steps, hourly steps (hours 9..168)


def job(args: tuple[str, list[str], np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    spot, perps, rows, t = args
    best = None
    for p in perps:
        f = P5 / f"{p}.parquet"
        if f.exists():
            df = pl.read_parquet(f)
            if best is None or df.height > best.height:
                best = df
    if best is None:
        return np.zeros(0, np.int64), np.zeros((0, 3, N5 + NH), np.float32)
    ms = best["ms"].to_numpy() // 1000
    o, h, lo, c = (best[k].to_numpy().astype(np.float64) for k in ("open", "high", "low", "close"))
    # Regular 5m grid from the first candle; forward-fill gaps.
    g0 = ms[0]
    n = int((ms[-1] - g0) // 300) + 1 + 168 * 12 + 2
    gi = ((ms - g0) // 300).astype(np.int64)
    C = np.full(n, np.nan)
    C[gi] = c
    C = pl.Series(C).fill_null(strategy="forward").to_numpy()
    C = np.where(np.isnan(C), np.nan, C)
    C = pl.Series(C).forward_fill().to_numpy()
    O, H, L = C.copy(), C.copy(), C.copy()
    O[gi], H[gi], L[gi] = o, h, lo
    ok_rows, out = [], []
    for r, ti in zip(rows, t, strict=True):
        e = int((ti + 300 - g0) // 300)  # index of the entry candle
        if e < 1 or e >= len(ms) + (n - len(ms)) - 168 * 12 - 2 or np.isnan(O[e]):
            continue
        if not (g0 + e * 300 in set(ms[max(0, np.searchsorted(ms, g0 + e * 300) - 1): np.searchsorted(ms, g0 + e * 300) + 1])):
            continue  # the perp must actually print the entry candle
        base = O[e]
        k = e + np.arange(168 * 12)
        hh, ll, cc = H[k] / base - 1, L[k] / base - 1, C[k] / base - 1
        p = np.empty((3, N5 + NH), np.float32)
        p[0, :N5], p[1, :N5], p[2, :N5] = hh[:N5], ll[:N5], cc[:N5]
        hr = slice(N5, None)
        p[0, hr] = hh[N5:].reshape(NH, 12).max(1)
        p[1, hr] = ll[N5:].reshape(NH, 12).min(1)
        p[2, hr] = cc[N5:].reshape(NH, 12)[:, -1]
        ok_rows.append(r)
        out.append(p)
    if not out:
        return np.zeros(0, np.int64), np.zeros((0, 3, N5 + NH), np.float32)
    return np.array(ok_rows, np.int64), np.stack(out)


def main() -> None:
    d = pl.read_parquet(OWN / "feats.parquet", columns=["row", "symbol", "t", "perp"]).filter("perp")
    have = [f.stem for f in P5.glob("*.parquet")]
    by_spot: dict[str, list[str]] = {}
    for p in have:
        by_spot.setdefault(spot_name(p), []).append(p)
    jobs = [(str(s), by_spot.get(str(s), []), g["row"].to_numpy(), g["t"].to_numpy()) for (s,), g in d.group_by(["symbol"])]
    jobs = [j for j in jobs if j[1]]
    print(len(jobs), "symbols with perp 5m data", file=sys.stderr)
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(job, jobs))
    rows = np.concatenate([r[0] for r in res])
    paths = np.concatenate([r[1] for r in res])
    order = np.argsort(rows)
    np.savez(OWN / "perp_paths.npz", rows=rows[order], paths=paths[order])
    print(len(rows), "paths of", d.height, "perp events; NaN share", float(np.isnan(paths[::50]).mean()), file=sys.stderr)


if __name__ == "__main__":
    main()
