"""X3 step 4: exits, 10-slot per-year books and the configuration grid (all in-sample)."""

from __future__ import annotations

import itertools
import sys

import numpy as np
import polars as pl
from common_ls import OWN

EX = OWN / "explore"
SPOT_COST, PERP_COST, STOP_SLIP = 0.004, 0.003, 0.001
KINDS = np.array(["time", "tp", "stop", "liq"])
D = pl.read_parquet(EX / "lag_cands.parquet")
_p = np.load(EX / "lag_paths.npz")
PATHS = {k: _p[k] for k in _p.files}

MODES = {  # name: (event side, trade side, venue, signal sign)
    "up_long": (1, 1, "spot", 1),
    "down_short": (-1, -1, "perp", 1),
    "bounce_long": (-1, 1, "spot", 1),  # long laggards (not yet fallen) after a BTC drop
    "bounce_fell": (-1, 1, "spot", -1),  # long the alts that fell most after a BTC drop
    "fade_short": (1, -1, "perp", 1),
}


def exits(idx: np.ndarray, side: int, venue: str, hold: int, tp, stop):
    pre = "" if venue == "spot" else "p"
    hi, lo, cl = (PATHS[pre + k][idx][:, :hold] for k in ("hi", "lo", "cl"))
    fc = PATHS["fc"][idx][:, :hold]
    if side > 0:
        fav, adv, cc = hi, -lo, cl
    else:
        fav, adv, cc = -lo, hi, -cl
    n = len(idx)
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
    liq = m_stop & (side < 0) & (stop is None)  # isolated 1x short: price +100% = liquidation
    kind[liq] = 3
    ret[liq] = -1.0
    r = np.arange(n)
    mfe = np.maximum.accumulate(fav, 1)[r, k].clip(0)
    mae = np.maximum.accumulate(adv, 1)[r, k].clip(0)
    cost = SPOT_COST if venue == "spot" else PERP_COST
    # a short pays -(cumulative rate): positive funding is received
    net = np.maximum(ret - cost + (fc[r, k] if side < 0 else 0.0), -1.0)
    return net, kind, mfe, mae, (k + 1)


def select(mode: str, combo: int, thr: float, years=None, random_prio: bool = False):
    ev, _side, venue, sg = MODES[mode]
    sig = D["signal"].to_numpy()
    sig = sig if sg > 0 else 1 - sig
    m = (
        (D["combo"].to_numpy() == combo)
        & (D["side"].to_numpy() == ev)
        & (sig >= thr)
        & (D["year"].to_numpy() >= 2021)
    )
    if venue == "perp":
        m &= D["perp_ok"].to_numpy()
    idx = np.where(m)[0]
    prio = sig[idx]
    if random_prio:
        prio = np.random.default_rng(7).random(len(idx))
    return idx, prio, sig


def book(idx, prio, net, hold_h, years=range(2021, 2027), slots=10, equity0=10_000.0):
    """Per-year $10k books: equal slots (equity/slots), stake <= 0.5% of 24h volume, 1 per coin."""
    t = D["t"].to_numpy()[idx]
    yr = D["year"].to_numpy()[idx]
    qv = D["qv24"].to_numpy()[idx]
    sym = D["symbol"].to_numpy()[idx]
    out = {}
    taken = np.zeros(len(idx), dtype=bool)
    stake = np.zeros(len(idx))
    order = np.lexsort((-prio, t))
    for y in years:
        eq = cash = equity0
        openp: list = []
        held: set = set()
        dd_peak, dd = equity0, 0.0
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
                    dd_peak = max(dd_peak, eq)
                    dd = min(dd, eq / dd_peak - 1)
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
            stake[i] = s
        for p in openp:
            eq += p[1] * p[2]
        out[y] = (eq / equity0 - 1, dd, int(sum(taken[i] for i in sel)))
    return out, taken, stake


def run_config(mode, combo, thr, hold, tp, stop, random_prio=False, do_book=True):
    _ev, side, venue, _ = MODES[mode]
    idx, prio, _s = select(mode, combo, thr, random_prio=random_prio)
    if len(idx) == 0:
        return None
    net, kind, mfe, mae, hh = exits(idx, side, venue, hold, tp, stop)
    res = dict(
        mode=mode,
        combo=combo,
        thr=thr,
        hold=hold,
        tp=tp,
        stop=stop,
        n=len(idx),
        avg=float(net.mean()),
        med=float(np.median(net)),
    )
    if do_book:
        out, taken, stake = book(idx, prio, net, hh)
        tn = net[taken]
        res["book_n"] = int(taken.sum())
        res["book_avg"] = float(tn.mean()) if taken.any() else 0.0
        for y in range(2021, 2027):
            r = out.get(y)
            res[f"y{y}"] = r[0] if r else np.nan
            res[f"n{y}"] = r[2] if r else 0
        res["worst"] = float(np.nanmin([res[f"y{y}"] for y in range(2021, 2027)]))
        res["maxdd"] = float(min(v[1] for v in out.values()))
        res["_taken"] = (idx, net, kind, mfe, mae, hh, prio, taken, stake)
    return res


if __name__ == "__main__":
    rows = []
    grid = itertools.product(
        MODES,
        range(6),
        (0.0, 0.8, 0.9, 0.95),
        (4, 12, 24, 72),
        (None, 0.05, 0.10, 0.20),
        (None, 0.05, 0.10),
    )
    for k, (mode, combo, thr, hold, tp, stop) in enumerate(grid):
        r = run_config(mode, combo, thr, hold, tp, stop, do_book=False)
        if r:
            rows.append(r)
        if k % 500 == 0:
            print(k, file=sys.stderr, flush=True)
    pl.DataFrame(rows).write_parquet(EX / "lag_grid.parquet")
    print(len(rows))
