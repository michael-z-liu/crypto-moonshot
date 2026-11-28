"""X4 shared: candidate loading, selection, exits (perp, short/long/neutral), 10-slot per-year book."""  # noqa: E501

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN

EX = OWN / "explore"
COST, STOP_SLIP = 0.003, 0.001
KINDS = np.array(["time", "tp", "stop", "liq"])
D = pl.read_parquet(EX / "contagion_cands.parquet")
_p = np.load(EX / "contagion_paths.npz")
PATHS = {k: _p[k].astype(np.float64) for k in _p.files}
SIG = np.sqrt(
    np.where(D["kind"].to_numpy() == 0, D["cpct"].to_numpy(), D["rpct"].to_numpy())
    * D["crash_pct"].to_numpy()
)
D = D.with_columns(signal=pl.Series(SIG))


def select(kind: int, k: int, fell: float | None, thr: float):
    """Candidate rows: this correlation kind, peer not fallen below `fell` over 4h, signal >= thr,
    top-k per event by signal. Returns indices and priority."""
    m = (D["kind"].to_numpy() == kind) & (D["signal"].to_numpy() >= thr)
    if fell is not None:
        m &= D["peer_r4"].to_numpy() > fell
    sub = D.with_row_index("ix").filter(pl.Series(m))
    sub = sub.sort(["ev", "signal"], descending=[False, True]).with_columns(
        r=pl.col("signal").rank("ordinal", descending=True).over("ev")
    )
    sub = sub.filter(pl.col("r") <= k)
    idx = sub["ix"].to_numpy()
    return idx, D["signal"].to_numpy()[idx]


def exits(idx: np.ndarray, mode: str, hold: int, tp, stop):
    hi, lo, cl, fc = (PATHS[k][idx][:, :hold] for k in ("hi", "lo", "cl", "fc"))
    n = len(idx)
    r = np.arange(n)
    if mode == "neutral":  # 50% short peer + 50% long BTC, close-based path, no tp/stop
        bcl, bfc = PATHS["bcl"][idx][:, :hold], PATHS["bfc"][idx][:, :hold]
        comb = 0.5 * (-cl) + 0.5 * bcl
        k = hold - 1
        net = np.maximum(comb[:, k] - COST + 0.5 * fc[:, k] - 0.5 * bfc[:, k], -1.0)
        mfe = np.maximum.accumulate(comb, 1)[:, k].clip(0)
        mae = np.maximum.accumulate(-comb, 1)[:, k].clip(0)
        return net, np.zeros(n, dtype=int), mfe, mae, np.full(n, hold)
    side = -1 if mode == "short" else 1
    if side > 0:
        fav, adv, cc = hi, -lo, cl
    else:
        fav, adv, cc = -lo, hi, -cl
    big = hold + 1
    stop_lvl = stop if stop is not None else (1.0 if side < 0 else None)
    k_stop = np.full(n, big)
    if stop_lvl is not None:
        h = adv >= stop_lvl
        k_stop = np.where(h.any(1), h.argmax(1), big)
    k_tp = np.full(n, big)
    if tp is not None:
        h = fav >= tp
        k_tp = np.where(h.any(1), h.argmax(1), big)
    kind = np.zeros(n, dtype=int)
    k = np.full(n, hold - 1)
    ret = cc[:, hold - 1].copy()
    m_stop = (k_stop < big) & (k_stop <= k_tp)
    m_tp = (k_tp < big) & ~m_stop
    kind[m_stop], kind[m_tp] = 2, 1
    k[m_stop], k[m_tp] = k_stop[m_stop], k_tp[m_tp]
    ret[m_stop] = -(stop_lvl or 0) - STOP_SLIP
    ret[m_tp] = tp if tp is not None else 0
    liq = m_stop & (side < 0) & (stop is None)
    kind[liq] = 3
    ret[liq] = -1.0
    mfe = np.maximum.accumulate(fav, 1)[r, k].clip(0)
    mae = np.maximum.accumulate(adv, 1)[r, k].clip(0)
    f = fc[r, k]
    net = np.maximum(ret - COST + (f if side < 0 else -f), -1.0)
    return net, kind, mfe, mae, k + 1


def book(idx, prio, net, hold_h, years=range(2021, 2027), slots=10, equity0=10_000.0):
    t = D["t"].to_numpy()[idx]
    yr = D["year"].to_numpy()[idx]
    qv = D["qv24"].to_numpy()[idx]
    sym = D["symbol"].to_numpy()[idx]
    out = {}
    taken = np.zeros(len(idx), dtype=bool)
    order = np.lexsort((-prio, t))
    for y in years:
        eq = cash = equity0
        openp: list = []
        held: set = set()
        peak, dd = equity0, 0.0
        sel = [i for i in order if yr[i] == y]
        for i in sel:
            ti = t[i]
            still = []
            for p in openp:
                if p[0] <= ti:
                    pnl = p[1] * p[2]
                    cash += p[1] + pnl
                    eq += pnl
                    held.discard(p[3])
                    peak = max(peak, eq)
                    dd = min(dd, eq / peak - 1)
                else:
                    still.append(p)
            openp = still
            if len(openp) >= slots or sym[i] in held:
                continue
            s = min(eq / slots, cash, 0.005 * qv[i])
            if s <= 1:
                continue
            cash -= s
            openp.append((ti + hold_h[i] * 3600, s, net[i], sym[i]))
            held.add(sym[i])
            taken[i] = True
        for p in openp:
            eq += p[1] * p[2]
        out[y] = (eq / equity0 - 1, dd, int(sum(taken[i] for i in sel)))
    return out, taken


def run(mode, kind, k, fell, thr, hold, tp, stop, do_book=True, random_prio=False):
    idx, prio = select(kind, k, fell, thr)
    if len(idx) == 0:
        return None
    if random_prio:
        prio = np.random.default_rng(7).random(len(idx))
    net, kd, mfe, mae, hh = exits(idx, mode, hold, tp, stop)
    yr = D["year"].to_numpy()[idx]
    res = dict(
        mode=mode,
        kind=kind,
        k=k,
        fell=fell,
        thr=thr,
        hold=hold,
        tp=tp,
        stop=stop,
        n=len(idx),
        avg=float(net.mean()),
    )
    for y in range(2021, 2027):
        res[f"a{y}"] = float(net[yr == y].mean()) if (yr == y).any() else np.nan
    if do_book:
        out, taken = book(idx, prio, net, hh)
        res["book_n"] = int(taken.sum())
        res["book_avg"] = float(net[taken].mean()) if taken.any() else 0.0
        for y in range(2021, 2027):
            res[f"y{y}"], res[f"n{y}"] = out[y][0], out[y][2]
        res["worst"] = min(res[f"y{y}"] for y in range(2021, 2027))
        res["maxdd"] = min(v[1] for v in out.values())
        res["_taken"] = (idx, net, kd, mfe, mae, hh, prio, taken)
    return res
