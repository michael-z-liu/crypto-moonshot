"""M10: trading rules on top of the walk-forward CNN scores, including forecast-driven exits.

Every trade can carry its own take-profit, stop and holding time, so a path forecast can set
them: enter when the forecast upside beats costs, take profit near the forecast peak, stop
below the forecast's pessimistic band, and leave at the forecast peak time.

    uv run python research/volume_spike/m13_policy.py <variant>   # grid of rules for one variant

Chosen on 2022-2024 only (mean yearly CAGR, worst-year drawdown no worse than -25%); 2025-26
reported. Uses cache/wf/<variant>.npz from m12_cnn_wf.py.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta

import numpy as np
import polars as pl
from common import CACHE, RESULTS
from m12_cnn_wf import PATH_STEPS, WF, YEARS

COST = 0.003
SELECT = (2022, 2023, 2024)


@dataclass(frozen=True)
class Rule:
    q: float = 0.98  # score percentile threshold
    hold: int = 12  # bars; ignored when hold_from_forecast
    tp: float | None = None
    sl: float | None = None
    slots: int = 5
    gate: str = "none"  # none | flush (BTC volume >= 4x or BTC 4h <= -1%)
    sizing: str = "equal"  # equal | score
    day_cap: int | None = None  # max new positions per day
    cooldown_h: int = 0  # hours before the same coin can be bought again
    # forecast-driven (path variants only)
    rank: str = "score"  # score | peak (forecast p50 peak over the horizon)
    min_peak: float | None = None  # enter only if the p50 peak (net, %) >= this
    hold_from_forecast: bool = False  # exit at the p50 peak time
    tp_from: str | None = None  # p50 | p90: take profit at that band's peak
    tp_frac: float = 1.0
    sl_from_p10: bool = False  # stop at the p10 trough (floored at -2%)
    horizon: int = 48  # bars the forecast-driven rules look ahead (max 96)


def exits(
    fwd: np.ndarray, tp: np.ndarray, sl: np.ndarray, bars: np.ndarray, cost: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trade exits: stop first if both touched in a candle; NaN tp/sl means none."""
    hi, lo, cl = fwd[:, 0], fwd[:, 1], fwd[:, 2]
    n, h = cl.shape
    j = np.arange(h)[None, :]
    live = j < bars[:, None]
    never = h + 1
    hit_tp = live & (hi >= np.where(np.isnan(tp), np.inf, tp)[:, None])
    hit_sl = live & (lo <= -np.where(np.isnan(sl), np.inf, sl)[:, None])
    t_tp = np.where(hit_tp.any(1), hit_tp.argmax(1), never)
    t_sl = np.where(hit_sl.any(1), hit_sl.argmax(1), never)
    last = bars - 1
    ret = cl[np.arange(n), last].astype(np.float64)
    t_exit = last.copy()
    stop = (t_sl < never) & (t_sl <= t_tp)
    take = (t_tp < never) & (t_tp < t_sl)
    ret[stop] = -sl[stop]
    t_exit[stop] = t_sl[stop]
    ret[take] = tp[take]
    t_exit[take] = t_tp[take]
    return ret - cost, t_exit


def book(sig: pl.DataFrame, slots: int, day_cap: int | None, cooldown_h: int) -> dict[str, float]:
    """Slot-limited portfolio from precomputed per-trade returns and exit bars (portfolio.py)."""
    sig = sig.sort("open_time", "prio")
    equity, open_pos, rows, taken = 1.0, [], [], []
    last_exit: dict[str, object] = {}
    per_day: dict[object, int] = {}
    cool = timedelta(hours=cooldown_h)
    for r in sig.iter_rows(named=True):
        entry_t = r["open_time"] + timedelta(minutes=5)
        still = []
        for p in open_pos:
            if p[0] <= entry_t:
                equity += p[2] * p[3]
            else:
                still.append(p)
        open_pos = still
        sym, day = r["symbol"], entry_t.date()
        if len(open_pos) >= slots or any(p[1] == sym for p in open_pos):
            continue
        if day_cap is not None and per_day.get(day, 0) >= day_cap:
            continue
        if cooldown_h and sym in last_exit and entry_t < last_exit[sym] + cool:  # type: ignore[operator]
            continue
        stake = equity / slots * r["size"]
        exit_t = entry_t + timedelta(minutes=5 * (int(r["exit_bar"]) + 1))
        open_pos.append((exit_t, sym, stake, r["ret"]))
        last_exit[sym] = exit_t
        per_day[day] = per_day.get(day, 0) + 1
        rows.append((exit_t.date(), stake * r["ret"], r["ret"], sym))
        taken.append(r["i"])
    if not rows:
        return {"trades": 0, "mean_bp": np.nan, "cagr": 0.0, "max_dd": 0.0, "_rows": []}
    t = pl.DataFrame(rows, schema=["d", "pnl", "ret", "symbol"], orient="row")
    daily = t.group_by("d").agg(pl.col("pnl").sum()).sort("d")
    eq = 1 + np.cumsum(daily["pnl"].to_numpy())
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = float(min(0.0, (eq / peak - 1).min()))
    return {"trades": t.height, "mean_bp": t["ret"].mean() * 1e4, "win": (t["ret"] > 0).mean(),
            "cagr": eq[-1] - 1, "max_dd": dd, "_rows": rows, "_taken": taken}  # fmt: skip


class Ctx:
    def __init__(self, variant: str) -> None:
        self.meta = pl.read_parquet(CACHE / "seq2_meta.parquet").with_columns(
            year=pl.col("open_time").dt.year()
        )
        self.fwd = np.load(CACHE / "fwd2.npy", mmap_mode="r")
        z = np.load(WF / f"{variant}.npz")
        self.pct = np.nan_to_num(z["pct"], nan=-1.0)
        self.path = z["path"].astype(np.float32) if "path" in z.files else None
        m = self.meta
        self.flush = (
            ((m["btc_vol_mult"] >= 4) | (m["btc_ret_4h"] <= -0.01)).fill_null(False).to_numpy()
        )
        self.years = m["year"].to_numpy()
        self.cost = COST
        self.exclude = np.zeros(m.height, dtype=bool)

    def evaluate(self, rule: Rule) -> list[dict[str, object]]:
        out = []
        for year in YEARS:
            m = (self.years == year) & (self.pct >= rule.q) & ~self.exclude
            if rule.gate == "flush":
                m &= self.flush
            k = np.flatnonzero(m)
            n = len(k)
            tp = np.full(n, np.nan if rule.tp is None else rule.tp)
            sl = np.full(n, np.nan if rule.sl is None else rule.sl)
            bars = np.full(n, rule.hold)
            prio = -self.pct[k]
            uses_path = rule.rank == "peak" or rule.min_peak is not None or rule.hold_from_forecast
            uses_path = uses_path or rule.tp_from is not None or rule.sl_from_p10
            if self.path is not None and uses_path:
                hz = int(np.searchsorted(PATH_STEPS, rule.horizon - 1)) + 1
                p = self.path[k, :hz]  # [n, steps, (p10, p50, p90)] net %, after cost
                p50 = p[:, :, 1]
                peak = p50.max(1)
                if rule.min_peak is not None:
                    keep = peak >= rule.min_peak
                    k, tp, sl, bars, prio, p, p50, peak = (
                        a[keep] for a in (k, tp, sl, bars, prio, p, p50, peak)
                    )
                if rule.rank == "peak":
                    prio = -peak
                if rule.hold_from_forecast:
                    bars = PATH_STEPS[p50.argmax(1)] + 1
                if rule.tp_from:
                    band = 1 if rule.tp_from == "p50" else 2
                    gross = (p[:, :, band].max(1) / 100 + COST) * rule.tp_frac
                    tp = np.where(gross > 0.002, gross, np.nan)
                if rule.sl_from_p10:
                    trough = -(p[:, :, 0].min(1) / 100 + COST)
                    sl = np.maximum(trough, 0.02)
            if len(k) == 0:
                out.append(
                    {
                        "year": year,
                        "trades": 0,
                        "mean_bp": np.nan,
                        "cagr": 0.0,
                        "max_dd": 0.0,
                        "_rows": [],
                    }
                )
                continue
            fwd = np.asarray(self.fwd[k], dtype=np.float32)
            ret, t_exit = exits(fwd, tp, sl, bars.astype(int), self.cost)
            size = np.ones(len(k))
            if rule.sizing == "score":
                size = np.clip((self.pct[k] - rule.q) / (1 - rule.q), 0.25, 1.0) * 1.6
            sig = (
                self.meta[k]
                .select("symbol", "open_time")
                .with_columns(
                    ret=pl.Series(ret),
                    exit_bar=pl.Series(t_exit),
                    prio=pl.Series(prio),
                    size=pl.Series(size),
                    i=pl.Series(k),  # row in meta, so callers can find the trades taken
                )
            )
            out.append({"year": year, **book(sig, rule.slots, rule.day_cap, rule.cooldown_h)})
        return out


def summarise(rule: Rule, yrs: list[dict[str, object]]) -> dict[str, object]:
    by = {r["year"]: r for r in yrs}
    sel_cagr = [float(by[y]["cagr"]) for y in SELECT]  # type: ignore[arg-type]
    return {
        **{k: v for k, v in asdict(rule).items()},
        "select_cagr": float(np.mean(sel_cagr)), "select_worst_cagr": min(sel_cagr),
        "select_worst_dd": min(float(by[y]["max_dd"]) for y in SELECT),  # type: ignore[arg-type]
        "late_cagr": float(np.mean([float(by[y]["cagr"]) for y in (2025, 2026)])),  # type: ignore[arg-type]
        **{f"cagr_{y}": by[y]["cagr"] for y in YEARS},
        **{f"bp_{y}": by[y]["mean_bp"] for y in YEARS},
        **{f"n_{y}": by[y]["trades"] for y in YEARS}, **{f"dd_{y}": by[y]["max_dd"] for y in YEARS},
    }  # fmt: skip


def grid(has_path: bool) -> list[Rule]:
    rules = [Rule()]
    for q, hold, slots in itertools.product((0.95, 0.97, 0.98, 0.99), (6, 12, 24, 48), (3, 5, 10)):
        rules.append(Rule(q=q, hold=hold, slots=slots))
    for tp, sl in itertools.product((None, 0.02, 0.04, 0.08), (None, 0.03, 0.05, 0.10)):
        rules.append(Rule(tp=tp, sl=sl, hold=48))
    for gate, sizing, cap, cool in itertools.product(
        ("none", "flush"), ("equal", "score"), (None, 3, 10), (0, 24)
    ):
        rules.append(Rule(gate=gate, sizing=sizing, day_cap=cap, cooldown_h=cool))
    if has_path:
        for hz, minp, rank in itertools.product(
            (12, 48, 96), (None, 0.0, 0.5, 1.0, 2.0), ("score", "peak")
        ):
            rules.append(
                Rule(q=0.95, rank=rank, min_peak=minp, hold_from_forecast=True, horizon=hz)
            )
        for hz, tpf, frac, slp in itertools.product(
            (48, 96), ("p50", "p90"), (0.5, 1.0), (False, True)
        ):
            rules.append(Rule(q=0.95, rank="peak", min_peak=0.5, hold_from_forecast=True,
                              horizon=hz, tp_from=tpf, tp_frac=frac, sl_from_p10=slp))  # fmt: skip
        for q, minp in itertools.product((0.9, 0.95, 0.98), (0.5, 1.0, 2.0)):
            rules.append(Rule(q=q, rank="peak", min_peak=minp, hold_from_forecast=True, horizon=96))
    return list(dict.fromkeys(rules))


def main(variant: str) -> None:
    ctx = Ctx(variant)
    rows = [summarise(r, ctx.evaluate(r)) for r in grid(ctx.path is not None)]
    df = pl.DataFrame(rows, infer_schema_length=None).with_columns(variant=pl.lit(variant))
    ok = df.filter(pl.col("select_worst_dd") >= -0.25)
    df.write_csv(RESULTS / f"m13_policy_{variant}.csv")
    show = ["q", "hold", "tp", "sl", "slots", "gate", "sizing", "day_cap", "cooldown_h", "rank"]
    show += ["min_peak", "hold_from_forecast", "tp_from", "tp_frac", "sl_from_p10", "horizon"]
    show += ["select_cagr", "select_worst_cagr", "select_worst_dd", "late_cagr"]
    show += ["cagr_2025", "cagr_2026", "bp_2025", "bp_2026"]
    with pl.Config(tbl_rows=25, tbl_cols=30, tbl_width_chars=320, float_precision=2):
        print(ok.sort("select_cagr", descending=True).select(show).head(20))
        print(df.head(1).select(show))  # the default rule, for reference


if __name__ == "__main__":
    main(sys.argv[1])
