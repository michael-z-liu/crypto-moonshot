"""X4 step 1: contagion candidates.
For each lone crash (ret_4h <= -15%, |BTC 4h| <= 1%, perp active)
rank the perps by trailing 30d hourly-return correlation (raw and BTC-residual) with the crashed
coin, keep the top 30 of each, and store 72h perp paths from the next hour's open."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN
from futures_ls import funding

EX = OWN / "explore"
T0 = int(np.datetime64("2020-01-01", "s").astype(int))
P = np.load(EX / "perp_h.npz")
syms = list(P["syms"])
PO, PH, PL, PC, PQ = (
    P[k].astype(np.float64) for k in ("open", "high", "low", "close", "quote_volume")
)
T = PC.shape[0]
HZ, WIN, TOPN = 72, 720, 30
col = {s: j for j, s in enumerate(syms)}
BTC, ETH = col["BTCUSDT"], col["ETHUSDT"]
with np.errstate(all="ignore"):
    R = np.log(PC[1:] / PC[:-1])
R = np.vstack([np.full((1, PC.shape[1]), np.nan), R])
R[~np.isfinite(R)] = np.nan
pq = np.nan_to_num(PQ)
cs = np.cumsum(pq, 0)
QV24 = cs - np.vstack([np.zeros((24, pq.shape[1])), cs[:-24]])

fd = funding().with_columns(h=(pl.col("time").dt.epoch("s") - T0) // 3600)
FC = np.zeros((T, len(syms)))
for (s,), g in fd.group_by(["spot"]):
    j = col.get(str(s))
    if j is None:
        continue
    a = np.zeros(T)
    hh = g["h"].to_numpy()
    ok = (hh >= 0) & (hh < T)
    np.add.at(a, hh[ok], g["rate"].to_numpy()[ok])
    FC[:, j] = np.cumsum(a)

f = pl.read_parquet(
    OWN / "feats.parquet", columns=["symbol", "t", "year", "ret_4h", "btc_ret_4h", "perp"]
)
f = f.filter(
    (pl.col("ret_4h") <= -0.15)
    & (pl.col("btc_ret_4h").abs() <= 0.01)
    & pl.col("perp")
    & (pl.col("year") >= 2021)
).sort("t")
last: dict[str, int] = {}
ev = []
for r in f.iter_rows(named=True):
    if r["symbol"] not in col:
        continue
    if r["t"] - last.get(r["symbol"], -(10**12)) < 86400:
        continue
    last[r["symbol"]] = r["t"]
    ev.append(r)
print("events", len(ev), "of", f.height)
crash_sorted = np.sort([-e["ret_4h"] for e in ev])

rows: list[dict] = []
paths: dict[str, list] = {k: [] for k in ("hi", "lo", "cl", "fc", "bhi", "blo", "bcl", "bfc")}
for k, e in enumerate(ev):
    i = (e["t"] - T0) // 3600  # hour containing the crash bar; entry = open of hour i+1
    ent = i + 1
    if ent - WIN < 2 or ent + HZ >= T:
        continue
    c = col[e["symbol"]]
    X = R[ent - 1 - WIN : ent - 1]  # returns of hours up to i-1 (closed)
    rb = X[:, BTC]
    x = X[:, c]
    m = np.isfinite(X) & np.isfinite(x)[:, None]
    n = m.sum(0)
    Xm = np.where(m, X, 0.0)
    xm = np.where(m, x[:, None], 0.0)
    with np.errstate(all="ignore"):
        mx, my = xm.sum(0) / n, Xm.sum(0) / n
        cov = (np.where(m, (X - my) * (x[:, None] - mx), 0)).sum(0) / n
        corr = cov / np.sqrt(
            (np.where(m, (X - my) ** 2, 0)).sum(0)
            / n
            * (np.where(m, (x[:, None] - mx) ** 2, 0)).sum(0)
            / n
        )
        # BTC-residual returns (beta over the window, pairwise)
        mb = m & np.isfinite(rb)[:, None]
        nb = mb.sum(0)
        bb = np.where(mb, rb[:, None], 0.0)
        mbm = bb.sum(0) / nb
        vb = (np.where(mb, (rb[:, None] - mbm) ** 2, 0)).sum(0) / nb
        Xb = np.where(mb, X, 0.0)
        mxb = Xb.sum(0) / nb
        betaX = (np.where(mb, (X - mxb) * (rb[:, None] - mbm), 0)).sum(0) / nb / vb
        xx = np.where(mb, x[:, None], 0.0)
        mxx = xx.sum(0) / nb
        betax = (np.where(mb, (x[:, None] - mxx) * (rb[:, None] - mbm), 0)).sum(0) / nb / vb
        RX = np.where(mb, X - betaX * rb[:, None], np.nan)
        Rx = np.where(mb, x[:, None] - betax * rb[:, None], np.nan)
        mr = mb
        a1 = np.where(mr, RX, 0).sum(0) / nb
        a2 = np.where(mr, Rx, 0).sum(0) / nb
        cv = (np.where(mr, (RX - a1) * (Rx - a2), 0)).sum(0) / nb
        rcorr = cv / np.sqrt(
            (np.where(mr, (RX - a1) ** 2, 0)).sum(0)
            / nb
            * (np.where(mr, (Rx - a2) ** 2, 0)).sum(0)
            / nb
        )
    # eligibility: enough overlap, perp trades at entry, not the crash coin / BTC / ETH
    ok = (
        (n >= 300)
        & np.isfinite(PO[ent])
        & np.isfinite(PC[i])
        & np.isfinite(PC[i - 4])
        & (QV24[i] >= 1e6)
    )
    ok[[c, BTC, ETH]] = False
    ok &= np.isfinite(PH[ent : ent + HZ]).all(0)
    ok &= np.isfinite(corr) & np.isfinite(rcorr)
    idx = np.where(ok)[0]
    if len(idx) < 40:
        continue
    cpct = np.full(len(syms), np.nan)
    rpct = np.full(len(syms), np.nan)
    cpct[idx] = (np.argsort(np.argsort(corr[idx])) + 0.5) / len(idx)
    rpct[idx] = (np.argsort(np.argsort(rcorr[idx])) + 0.5) / len(idx)
    crash_pct = float(np.searchsorted(crash_sorted, -e["ret_4h"]) / len(crash_sorted))
    sel = {}
    for kind, v in ((0, corr), (1, rcorr)):
        top = idx[np.argsort(-v[idx])[:TOPN]]
        for rk, j in enumerate(top):
            sel[(kind, int(j))] = rk
    for (kind, j), rk in sel.items():
        e0 = PO[ent, j]
        rows.append(dict(
            ev=k, kind=kind, rank=rk, symbol=syms[j], crash=e["symbol"], t=T0 + ent * 3600, year=e["year"],  # noqa: E501
            crash_ret=e["ret_4h"], crash_pct=crash_pct, corr=float(corr[j]), rcorr=float(rcorr[j]),
            cpct=float(cpct[j]), rpct=float(rpct[j]), peer_r4=float(PC[i, j] / PC[i - 4, j] - 1),
            peer_r24=float(PC[i, j] / PC[i - 24, j] - 1) if np.isfinite(PC[i - 24, j]) else np.nan,
            qv24=float(QV24[i, j]), n_peers=len(idx),
        ))  # fmt: skip
        sl = slice(ent, ent + HZ)
        paths["hi"].append(PH[sl, j] / e0 - 1)
        paths["lo"].append(PL[sl, j] / e0 - 1)
        paths["cl"].append(PC[sl, j] / e0 - 1)
        paths["fc"].append(FC[sl, j] - FC[ent - 1, j])
        b0 = PO[ent, BTC]
        paths["bhi"].append(PH[sl, BTC] / b0 - 1)
        paths["blo"].append(PL[sl, BTC] / b0 - 1)
        paths["bcl"].append(PC[sl, BTC] / b0 - 1)
        paths["bfc"].append(FC[sl, BTC] - FC[ent - 1, BTC])
D = pl.DataFrame(rows)
D.write_parquet(EX / "contagion_cands.parquet")
np.savez(EX / "contagion_paths.npz", **{k: np.array(v, dtype=np.float32) for k, v in paths.items()})
print(
    D.height,
    D["ev"].n_unique(),
    D.group_by("year").agg(pl.col("ev").n_unique()).sort("year").rows(),
)
