"""Fast event-level portfolio simulator for the long/short search (M0).

Every candidate is a row of seq2_meta (a liquid 5m spike). A trade enters at the next 5m open
(the entry price of fwd2/fwd72) and is followed on a mixed path: the first 8 hours on 5m bars
(fwd2) and hours 9-72 on hourly bars (fwd72). Each step has a high, low and close relative to the
entry.

Exits, checked per step in this order: liquidation (leverage), stop loss, take profit, then the
time exit at the close of the step that reaches `hold_min`. Inside a step the adverse extreme is
assumed to come first (conservative). A stop or take profit fills exactly at its level; a stop
also pays `stop_slip` extra (gaps through stops in crashes).

Portfolio: `slots` equal slots of the current equity (cash capped), one position per coin,
candidates in time order, simultaneous ones by descending priority. Returns are on the margin
put up: the price return times `lev`, minus costs times `lev`, minus funding (shorts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import numpy as np
import polars as pl
from common_ls import fwd2, fwd72, meta

STEP_END_MIN = np.concatenate([np.arange(1, 97) * 5, np.arange(9, 73) * 60]).astype(np.int64)


PERP_STEP_END_MIN = np.concatenate([np.arange(1, 97) * 5, np.arange(9, 169) * 60]).astype(np.int64)


@cache
def _perp_store() -> tuple[np.ndarray, np.ndarray]:
    from common_ls import OWN

    z = np.load(OWN / "perp_paths.npz")
    return z["rows"], z["paths"]


def perp_paths(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perp paths [n, 3, 256] (5m to 8h, then hourly to 7 days) and a mask of rows that have one."""
    pr, pp = _perp_store()
    k = np.searchsorted(pr, rows)
    k = np.minimum(k, len(pr) - 1)
    ok = pr[k] == rows
    return pp[k[ok]], ok


def paths(rows: np.ndarray) -> np.ndarray:
    """[n, 3, 160] high, low, close relative to entry for the given seq2 rows."""
    order = np.argsort(rows)
    srt = rows[order]
    a = np.asarray(fwd2()[srt], dtype=np.float32)
    b = np.asarray(fwd72()[srt], dtype=np.float32)[:, :, 8:]
    out = np.empty((len(rows), 3, 160), dtype=np.float32)
    out[order] = np.concatenate([a, b], axis=2)
    return out


@dataclass(frozen=True)
class Rule:
    hold_min: int = 60
    tp: float | None = None  # take profit as a favourable price move, e.g. 0.10
    sl: float | None = None  # stop loss as an adverse price move, e.g. 0.05
    trail: float | None = None  # trailing stop from the best close so far
    trail_arm: float = 0.0  # the trailing stop is live only once the best close reaches this gain
    lev: float = 1.0
    cost: float = 0.004  # round trip, as a fraction of notional
    stop_slip: float = 0.0
    mm: float = 0.01  # maintenance margin used for the liquidation level
    slots: int = 5
    max_per_day: int | None = None
    liq_frac: float | None = 0.005  # stake at most this share of the coin's 24h quote volume
    loss_floor: float = -1.0  # worst return on the stake; below -1 = cross margin losses
    max_per_coin: int = 1  # positions allowed in the same coin at once (2 = add on a deeper spike)
    delay: int = 0  # enter this many 5m candles after the usual entry (< 96)
    delay_cond: str | None = None  # "confirm": only if price moved our way during the delay
    name: str = ""


def delayed(
    p: np.ndarray, d: int, step_end_in: np.ndarray = STEP_END_MIN
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-base paths to an entry at the close of step d-1. Returns paths, step ends (minutes after
    the new entry) and the price at the new entry relative to the old one."""
    base = 1 + p[:, 2, d - 1]
    q = (1 + p[:, :, d:]) / base[:, None, None] - 1
    pad = np.repeat(q[:, :, -1:], d, axis=2)
    q = np.concatenate([q, pad], axis=2)
    ends = np.concatenate(
        [step_end_in[d:] - 5 * d, step_end_in[-1] - 5 * d + 60 * np.arange(1, d + 1)]
    )
    return q.astype(np.float32), ends, base - 1


def exits(
    p: np.ndarray,
    side: np.ndarray,
    rule: Rule,
    step_end: np.ndarray = STEP_END_MIN,
    tp_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per trade: price return in the trade's favour (before costs), exit step, exit kind.

    kind: 0 time, 1 stop, 2 take profit, 3 liquidation, 4 trailing.
    """
    n = p.shape[0]
    s = side[:, None].astype(np.float32)
    fav = np.where(s > 0, p[:, 0], -p[:, 1])  # best move inside the step
    adv = np.where(s > 0, -p[:, 1], p[:, 0])  # worst move inside the step (positive = loss)
    cl = s * p[:, 2]
    last = int(np.searchsorted(step_end, rule.hold_min))
    last = min(last, p.shape[2] - 1)
    big = p.shape[2] + 10
    steps = np.arange(p.shape[2])[None, :]
    live = steps <= last

    def first(mask: np.ndarray) -> np.ndarray:
        mask = mask & live
        any_ = mask.any(1)
        return np.where(any_, mask.argmax(1), big)

    liq_lvl = 1.0 / rule.lev - rule.mm if rule.lev > 1 else np.inf
    t_liq = first(np.nan_to_num(adv, nan=0) >= liq_lvl)
    t_sl = first(np.nan_to_num(adv, nan=0) >= rule.sl) if rule.sl is not None else np.full(n, big)
    tp_v = tp_arr if tp_arr is not None else (np.full(n, rule.tp) if rule.tp is not None else None)
    t_tp = (
        first(np.nan_to_num(fav, nan=0) >= tp_v[:, None]) if tp_v is not None else np.full(n, big)
    )
    if rule.trail is not None:
        best = np.maximum.accumulate(np.nan_to_num(cl, nan=-1), axis=1)
        prev_best = np.concatenate([np.zeros((n, 1), np.float32), best[:, :-1]], axis=1)
        prev_best = np.maximum(prev_best, 0)
        t_tr = first(
            (np.nan_to_num(-adv, nan=0) <= prev_best - rule.trail) & (prev_best >= rule.trail_arm)
        )
    else:
        t_tr = np.full(n, big)
    t_time = np.full(n, last)
    stack = np.stack([t_time, t_sl, t_tp, t_liq, t_tr])  # kinds 0..4
    # Same step: liquidation, then stop/trailing, then take profit, then time.
    prio = np.array([3, 1, 2, 0, 1])[:, None]
    key = stack * 10 + prio
    kind = key.argmin(0)
    t_exit = stack[kind, np.arange(n)]
    ret = cl[np.arange(n), t_exit].astype(np.float64)
    ret = np.where(kind == 1, -rule.sl if rule.sl is not None else 0, ret)
    ret = np.where(kind == 2, tp_v if tp_v is not None else 0, ret)
    if rule.trail is not None:
        tr_lvl = prev_best[np.arange(n), np.minimum(t_exit, p.shape[2] - 1)] - rule.trail
        ret = np.where(kind == 4, tr_lvl, ret)
    ret = np.where(kind == 3, -liq_lvl, ret)
    ret = np.nan_to_num(ret, nan=0.0)
    return ret, t_exit, kind


@dataclass
class Book:
    trades: pl.DataFrame
    years: pl.DataFrame
    summary: dict[str, float] = field(default_factory=dict)


def simulate(
    cand: pl.DataFrame,
    rule: Rule,
    start_equity: float = 10_000,
    perp_funding: bool = False,
    source: str = "spot",
) -> Book:
    """cand: columns row (seq2 row), side (+1/-1), prio (higher first), optional fund_day (the
    funding this side pays per day held, as a fraction of notional; negative = received).
    """
    m = meta()
    cand = cand.join(
        m.select(pl.int_range(pl.len()).alias("row"), "symbol", "open_time").with_columns(
            t=pl.col("open_time").dt.epoch("s")
        ),
        on="row",
    ).sort(["t", "prio"], descending=[False, True])
    rows = cand["row"].to_numpy()
    side = cand["side"].to_numpy()
    if source == "perp":
        p, ok = perp_paths(rows)
        cand = cand.filter(pl.Series(ok))
        rows, side = rows[ok], side[ok]
        step_end = PERP_STEP_END_MIN
    else:
        p = paths(rows)
        step_end = STEP_END_MIN
    t_shift = 0
    if rule.delay > 0:
        p, step_end, moved = delayed(p, rule.delay, step_end)
        t_shift = rule.delay * 300
        if rule.delay_cond == "confirm":
            ok = side * moved > 0
            cand = cand.filter(pl.Series(ok))
            rows, side, p = rows[ok], side[ok], p[ok]
    tp_arr = cand["tp"].to_numpy().astype(np.float64) if "tp" in cand.columns else None
    ret, t_exit, kind = exits(p, side, rule, step_end, tp_arr)
    hold_min = step_end[t_exit]
    advp = np.nan_to_num(np.where(side[:, None] > 0, -p[:, 1], p[:, 0]), nan=0.0)
    adv_max = np.maximum.accumulate(advp, axis=1)[
        np.arange(len(t_exit)), np.minimum(t_exit, p.shape[2] - 1)
    ]
    favp = np.nan_to_num(np.where(side[:, None] > 0, p[:, 0], -p[:, 1]), nan=0.0)
    mfe_max = np.maximum.accumulate(favp, axis=1)[
        np.arange(len(t_exit)), np.minimum(t_exit, p.shape[2] - 1)
    ]
    fund = np.zeros(len(rows))
    if "fund_day" in cand.columns:  # funding paid by this side per day held, fraction of notional
        fund = cand["fund_day"].fill_null(0).to_numpy() * hold_min / 1440.0
    if perp_funding:
        from futures_ls import funding_paid

        t0 = cand["t"].to_numpy() + 300 + t_shift
        fund = funding_paid(cand["symbol"].to_numpy(), t0, t0 + hold_min * 60, side)
    net = rule.lev * (ret - rule.cost - (rule.stop_slip * (kind == 1)) - fund)
    net = np.maximum(net, rule.loss_floor)
    t_in = cand["t"].to_numpy() + 300 + t_shift
    t_out = t_in + hold_min * 60
    syms = cand["symbol"].to_numpy()
    cap = np.full(len(rows), np.inf)
    if rule.liq_frac is not None:
        cap = m["qv_base"].to_numpy()[rows] * 288 * rule.liq_frac

    equity = start_equity
    cash = start_equity
    open_pos: list[tuple[int, float, str, float]] = []  # (t_out, stake, symbol, net)
    held: dict[str, int] = {}
    taken = np.zeros(len(rows), dtype=bool)
    stake_arr = np.zeros(len(rows))
    np.zeros(len(rows))
    day_count: dict[int, int] = {}
    for i in range(len(rows)):
        ti = t_in[i]
        if open_pos:
            still = []
            for pos in open_pos:
                if pos[0] <= ti:
                    pnl = pos[1] * pos[3]
                    cash += pos[1] + pnl
                    equity += pnl
                    held[pos[2]] -= 1
                else:
                    still.append(pos)
            open_pos = still
        if held.get(syms[i], 0) >= rule.max_per_coin or len(open_pos) >= rule.slots:
            continue
        d = int(ti // 86400)
        if rule.max_per_day is not None and day_count.get(d, 0) >= rule.max_per_day:
            continue
        stake = min(equity / rule.slots, cash, cap[i])
        if stake <= 1:
            continue
        cash -= stake
        open_pos.append((int(t_out[i]), stake, syms[i], float(net[i])))
        held[syms[i]] = held.get(syms[i], 0) + 1
        taken[i] = True
        stake_arr[i] = stake
        day_count[d] = day_count.get(d, 0) + 1
    tr = cand.with_columns(
        net=pl.Series(net), ret=pl.Series(ret), kind=pl.Series(kind),
        hold_min=pl.Series(hold_min), stake=pl.Series(stake_arr), taken=pl.Series(taken),
        fund=pl.Series(fund), adv=pl.Series(adv_max.astype(np.float64)),
        mfe=pl.Series(mfe_max.astype(np.float64)),
    ).filter(pl.col("taken"))  # fmt: skip
    tr = tr.with_columns(
        pnl=pl.col("stake") * pl.col("net"),
        t_out=pl.col("t") + 300 + t_shift + pl.col("hold_min") * 60,
        year=pl.col("open_time").dt.year(),
    )
    # Realised equity curve in exit order.
    ex = tr.sort("t_out")
    eq = start_equity + ex["pnl"].cum_sum()
    ex = ex.with_columns(equity=eq)
    peak = ex["equity"].cum_max()
    ex = ex.with_columns(dd=ex["equity"] / peak - 1)
    yrs = []
    prev_end = start_equity
    for y in sorted(set(ex["year"].to_list())):
        e = ex.filter(
            pl.col("t_out").cast(pl.Int64) < int(np.datetime64(f"{y + 1}-01-01", "s").astype(int))
        )
        end_eq = float(e["equity"][-1]) if e.height else prev_end
        ty = tr.filter(pl.col("year") == y)
        sub = ex.filter(pl.col("year") == y)
        yrs.append({
            "year": y, "trades": ty.height, "bp": float(ty["net"].mean() or 0) * 1e4,
            "med_bp": float(ty["net"].median() or 0) * 1e4,
            "win": float((ty["net"] > 0).mean() or 0), "ret": end_eq / prev_end - 1,
            "dd": float(sub["dd"].min() or 0), "liq": int((ty["kind"] == 3).sum()),
        })  # fmt: skip
        prev_end = end_eq
    years = pl.DataFrame(yrs)
    summary = {
        "trades": tr.height,
        "bp": float(tr["net"].mean() or 0) * 1e4,
        "final": float(ex["equity"][-1]) if ex.height else start_equity,
        "maxdd": float(ex["dd"].min() or 0) if ex.height else 0.0,
    }
    return Book(tr, years, summary)


def fmt_years(b: Book, years: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025, 2026)) -> str:
    d = {r["year"]: r for r in b.years.iter_rows(named=True)}
    parts = []
    for y in years:
        r = d.get(y)
        parts.append(
            f"{y}: {r['trades']:>4}t {r['bp']:+6.0f}bp {r['ret'] * 100:+6.0f}%"
            if r
            else f"{y}:   -"
        )
    return " | ".join(parts) + f" | maxDD {b.summary['maxdd'] * 100:.0f}%"


def simulate_yearly(
    cand: pl.DataFrame,
    rule: Rule,
    start_equity: float = 10_000,
    perp_funding: bool = False,
    years: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025, 2026),
    source: str = "spot",
) -> Book:
    """One book per entry year, each starting at `start_equity` (the account size the user has),
    so a year's return does not depend on how much earlier years compounded."""
    m = meta()
    yr_of = m["year"].to_numpy()
    books = []
    for y in years:
        c = cand.filter(pl.Series(yr_of[cand["row"].to_numpy()] == y))
        if c.height == 0:
            continue
        b = simulate(c, rule, start_equity, perp_funding, source)
        if b.trades.height:
            books.append(b)
    if not books:
        return Book(
            pl.DataFrame(),
            pl.DataFrame(
                {
                    "year": [],
                    "trades": [],
                    "bp": [],
                    "med_bp": [],
                    "win": [],
                    "ret": [],
                    "dd": [],
                    "liq": [],
                }
            ),
            {"trades": 0, "bp": 0.0, "final": start_equity, "maxdd": 0.0},
        )
    trades = pl.concat([b.trades for b in books])
    yrs = pl.concat([b.years.filter(pl.col("year") == b.trades["year"][0]) for b in books])
    summary = {
        "trades": trades.height,
        "bp": float(trades["net"].mean() or 0) * 1e4,
        "final": float(np.prod(1 + yrs["ret"].to_numpy())) * start_equity,
        "maxdd": float(yrs["dd"].min() or 0),
    }
    return Book(trades, yrs, summary)
