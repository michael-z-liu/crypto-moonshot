"""X3 step 2: events (big BTC moves) and candidate alts with lag signal and 72h paths."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN
from futures_ls import funding

EX = OWN / "explore"
T0 = int(np.datetime64("2020-01-01", "s").astype(int))
S = np.load(EX / "spot_h.npz")
P = np.load(EX / "perp_h.npz")
sym_s = list(S["syms"])
sym_p = list(P["syms"])
Op, Hh, Lw, C, QV = (S[k] for k in ("open", "high", "low", "close", "qv"))
PO, PH, PL, PC, PQ = (P[k] for k in ("open", "high", "low", "close", "quote_volume"))
T = C.shape[0]
HZ = 72
BTC = sym_s.index("BTCUSDT")
ETH = sym_s.index("ETHUSDT")
pcol = {s: j for j, s in enumerate(sym_p)}
p_of_s = np.array([pcol.get(s, -1) for s in sym_s])

# (window h, threshold) combos: small and big moves per window
COMBOS = [(4, 0.03), (4, 0.05), (12, 0.05), (12, 0.08), (24, 0.06), (24, 0.10)]
LOCK = 12
BETA_WIN = 720


def ffill(a: np.ndarray) -> np.ndarray:
    """Forward fill NaN down the rows of each column (flat after a delisting)."""
    out = a.copy()
    idx = np.where(np.isfinite(out), np.arange(len(out))[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return np.take_along_axis(out, idx, axis=0)


Cf = ffill(C)
PCf = ffill(PC)
r1 = np.diff(np.log(Cf), axis=0, prepend=np.nan)  # r1[h] = ret of hour h close vs h-1 close
# funding cumulative on the hour grid for the perp columns (by spot-named perp)
fd = funding().with_columns(h=(pl.col("time").dt.epoch("s") - T0) // 3600)
FC = np.zeros((T, len(sym_p)))
for (s,), g in fd.group_by(["spot"]):
    j = pcol.get(str(s))
    if j is None:
        continue
    a = np.zeros(T)
    hh = g["h"].to_numpy()
    ok = (hh >= 0) & (hh < T)
    np.add.at(a, hh[ok], g["rate"].to_numpy()[ok])
    FC[:, j] = np.cumsum(a)


def build() -> pl.DataFrame:
    rb = r1[:, BTC]
    rows: list[dict] = []
    paths = {k: [] for k in ("hi", "lo", "cl", "phi", "plo", "pcl", "fc")}
    alt_ok = np.ones(len(sym_s), dtype=bool)
    alt_ok[[BTC, ETH]] = False
    qv24 = np.nancumsum(np.nan_to_num(QV), axis=0)
    qv24 = qv24 - np.vstack([np.zeros((24, QV.shape[1])), qv24[:-24]])
    pq2 = np.nan_to_num(PQ)
    for ci, (w, thr) in enumerate(COMBOS):
        last = {1: -(10**9), -1: -(10**9)}
        for i in range(BETA_WIN + w + 2, T - HZ - 3):
            rw = Cf[i, BTC] / Cf[i - w, BTC] - 1
            if abs(rw) < thr:
                continue
            d = 1 if rw > 0 else -1
            if i - last[d] < LOCK:
                continue
            last[d] = i
            e = i + 1  # entry hour (open)
            sl = slice(i - w - BETA_WIN, i - w)
            x, b = r1[sl], rb[sl, None]
            m = np.isfinite(x) & np.isfinite(b)
            n = m.sum(0)
            xm = np.where(m, x, 0.0)
            bm = np.where(m, b, 0.0)
            with np.errstate(all="ignore"):
                mx, mb = xm.sum(0) / n, bm.sum(0) / n
                cov = (np.where(m, (x - mx) * (b - mb), 0)).sum(0) / n
                vb = (np.where(m, (b - mb) ** 2, 0)).sum(0) / n
                vx = (np.where(m, (x - mx) ** 2, 0)).sum(0) / n
                beta = cov / vb
                corr = cov / np.sqrt(vb * vx)
                res = np.sqrt(np.maximum(vx - beta**2 * vb, 1e-12))
                ra = Cf[i] / Cf[i - w] - 1
                lag = d * (beta * rw - ra)
                z = lag / (res * np.sqrt(w))
            elig = (
                alt_ok & (n > 400) & (beta >= 0.8) & (corr >= 0.3) & np.isfinite(ra)
                & (qv24[i] >= 1e6) & np.isfinite(Op[e]) & (np.isfinite(C[i]))
            )  # fmt: skip
            pj = np.maximum(p_of_s, 0)
            ok_p = (p_of_s >= 0) & (pq2[i, pj] >= 50_000) & (pq2[i - 1, pj] >= 50_000)
            ok_p &= np.isfinite(PO[e, pj]) & np.isfinite(PC[i - 24, pj])
            js = np.where(elig)[0]
            if len(js) < 8:
                continue
            zz = z[js]
            sig = (np.argsort(np.argsort(zz)) + 0.5) / len(js)
            for j, zj, sg in zip(js, zz, sig, strict=True):
                en = Op[e, j]
                hi, lo, cl = Hh[e : e + HZ, j], Lw[e : e + HZ, j], Cf[e : e + HZ, j]
                hi = np.where(np.isfinite(hi), hi, cl)
                lo = np.where(np.isfinite(lo), lo, cl)
                if not np.isfinite(en) or not np.isfinite(cl[-1]):
                    continue
                pk = bool(ok_p[j])
                if pk:
                    q = p_of_s[j]
                    pen = PO[e, q]
                    pcl = PCf[e : e + HZ, q]
                    phi = np.where(np.isfinite(PH[e : e + HZ, q]), PH[e : e + HZ, q], pcl)
                    plo = np.where(np.isfinite(PL[e : e + HZ, q]), PL[e : e + HZ, q], pcl)
                    fcp = FC[e : e + HZ, q] - FC[e - 1, q]
                    pk = bool(np.isfinite(pcl[-1]))
                if not pk:
                    phi = plo = pcl = np.full(HZ, np.nan)
                    pen = np.nan
                    fcp = np.zeros(HZ)
                rows.append(
                    dict(combo=ci, w=w, thr=thr, i=i, t=T0 + e * 3600, side=d, j=int(j),
                         symbol=sym_s[j], z=float(zj), signal=float(sg), lag=float(lag[j]),
                         beta=float(beta[j]), rbtc=float(rw), ra=float(ra[j]), nel=len(js),
                         qv24=float(qv24[i, j]), perp_ok=pk,
                         year=int(np.datetime64(T0 + e * 3600, "s").astype("datetime64[Y]").astype(int) + 1970))  # noqa: E501
                )  # fmt: skip
                paths["hi"].append(hi / en - 1)
                paths["lo"].append(lo / en - 1)
                paths["cl"].append(cl / en - 1)
                paths["phi"].append(phi / pen - 1)
                paths["plo"].append(plo / pen - 1)
                paths["pcl"].append(pcl / pen - 1)
                paths["fc"].append(fcp)
    df = pl.DataFrame(rows)
    np.savez(EX / "lag_paths.npz", **{k: np.array(v, dtype=np.float32) for k, v in paths.items()})
    df.write_parquet(EX / "lag_cands.parquet")
    return df


if __name__ == "__main__":
    df = build()
    print(df.shape)
    print(
        df.group_by(["combo", "side"])
        .agg(ev=pl.col("i").n_unique(), n=pl.len())
        .sort(["combo", "side"])
    )
