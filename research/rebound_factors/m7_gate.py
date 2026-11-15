# ruff: noqa: E501  (markdown text and table rows read better unsplit)
"""M7: the market gate described on history 2020-26. Description, not evidence.

    uv run --group research python m7_gate.py

The gate is fixed in the plan (`docs/plans/rebound-factors.md`, "Gate protocol"): open on day `D`
when BTC's last daily close is above the mean of its last 50 daily closes (`m_btc_sma50 > 0`);
closed otherwise and when the factor is null. No tuning, no alternatives. The gate was chosen
after M3 showed BTC trend leading on 2020-23 and after 2024-26 was revealed, so these tables are
description. The verdict comes from `forward.py` on candles after the data end.

Arm A: every usable entry (the frame M5 scores: no excluded entries, no entries without an
outcome), gate open against closed (a null factor counts as closed; null days are also counted
apart). Difference of the means with a month bootstrap (months resampled with replacement,
`m4_model.N_BOOT` draws, seed `SEED`, as M3 and M5 do).

Arm B: the frozen rule (`results/spec.json`) scored as in M4/M5 (2021-23 walk-forward on the
discovery entries, 2024-26 retrained each 1 January, `m5_confirm.score_years`), its picks kept
only on gate-open days, then the same 10-slot book. Open positions are held to their own exit
(the book only ever looks at picks). Each scope is run through M5's own `scope_stats` three
times' worth of numbers: the ungated rule (a reproduction of M4/M5, asserted against them), the
gated rule, and same-day random books (100 draws, `SEED + year` per year and `SEED` per period)
drawn on the gate-open days' entries with as many entries a day as the gated picks.

Outputs: `results/m7_gate.csv` (tidy) and `results/m7_gate.md` (generated from the csv).
"""

from __future__ import annotations

import math
import time
from typing import Any

import m5_confirm as m5
import numpy as np
import numpy.typing as npt
import polars as pl
from build_factors import rmean
from common import (
    CONFIRM_END_US,
    DAY,
    DISCOVERY_END_US,
    F64,
    FACTORS,
    FIRST_D_US,
    I64,
    RESULTS,
    SEALED_START_US,
    utc,
)
from m4_model import N_BOOT, SEED
from m5_confirm import Rows, col_f, fmean, pc
from model import STOP_VARIANTS, Spec, load_spec, picks

type I8 = npt.NDArray[np.int8]

OUT_CSV = RESULTS / "m7_gate.csv"
OUT_MD = RESULTS / "m7_gate.md"
GATE_FACTOR = "m_btc_sma50"

DISC_YEARS = (2020, 2021, 2022, 2023)
WF_YEARS = (2021, 2022, 2023)  # discovery walk-forward test years (M4)
CONF_YEARS = (2024, 2025, 2026)
HITS = ("hit10", "hit20", "hit_ns")
RETS = ("ret_10", "ret_20", "ret_ns")

PROTOCOL_REASON = (
    "History (2020-23 and 2024-26) is description, not evidence: the factor was chosen after "
    "seeing that it led M3 on 2020-23, and 2024-26 has been opened."
)


# --- the gate --------------------------------------------------------------------------------


def gate_state(sma50: F64) -> I8:
    """Per day: 1 open (factor > 0), 0 closed (factor <= 0), -1 null (factor missing).

    The factor is `m_btc_sma50`: BTC's last daily close over the mean of its last 50 daily closes,
    minus 1. Exactly equal (factor 0) is closed.
    """
    v = np.asarray(sma50, dtype=np.float64)
    out = np.where(np.isnan(v), -1, np.where(v > 0, 1, 0))
    return np.asarray(out, dtype=np.int8)


def sma50_factor(close: F64) -> F64:
    """`m_btc_sma50` from a series of daily closes (as `build_factors.market_series` does)."""
    c = np.asarray(close, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(c / rmean(c, 50) - 1, dtype=np.float64)


def gate_open_expr() -> pl.Expr:
    """Boolean column: the gate is open (a null factor is closed)."""
    return (pl.col(GATE_FACTOR) > 0).fill_null(False).alias("gate_ok")


def with_gate(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(gate_open_expr())


def gated_picks(pk: pl.DataFrame) -> pl.DataFrame:
    """The rule's picks kept on gate-open days (the gate acts on picks, before the book)."""
    return pk.filter(pl.col("gate_ok"))


# --- arm A -----------------------------------------------------------------------------------


def month_of(d_us: I64) -> I64:
    m = pl.from_epoch(pl.Series(d_us), time_unit="us")
    return np.asarray((m.dt.year() * 12 + m.dt.month()).to_numpy(), dtype=np.int64)


def boot_diff(month: I64, y: F64, is_open: npt.NDArray[np.bool_], seed: int) -> tuple[float, float]:
    """95% interval of mean(y | open) - mean(y | closed), resampling calendar months."""
    ok = ~np.isnan(y)
    month, y, is_open = month[ok], y[ok], is_open[ok]
    if not (is_open.any() and (~is_open).any()):
        return math.nan, math.nan
    mu, mi = np.unique(month, return_inverse=True)
    k = len(mu)
    so = np.bincount(mi, weights=np.where(is_open, y, 0.0), minlength=k)
    co = np.bincount(mi, weights=is_open.astype(float), minlength=k)
    sc = np.bincount(mi, weights=np.where(is_open, 0.0, y), minlength=k)
    cc = np.bincount(mi, weights=(~is_open).astype(float), minlength=k)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(N_BOOT, k))
    w = np.stack([np.bincount(d, minlength=k) for d in draws]).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = (w @ so) / (w @ co) - (w @ sc) / (w @ cc)
    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
    return float(lo), float(hi)


def arm_a(rows: Rows, scope: str, t: pl.DataFrame) -> None:
    """Arm A for one scope: every usable entry in `t` (needs `gate_ok`), open against closed."""
    is_open = t["gate_ok"].to_numpy().astype(np.bool_)
    month = month_of(t["d_us"].to_numpy().astype(np.int64))
    rows.add("armA", scope, "entries_open", float(is_open.sum()))
    rows.add("armA", scope, "entries_closed", float((~is_open).sum()))
    rows.add("armA", scope, "entries_null", float(t[GATE_FACTOR].is_null().sum()))
    d = t["d_us"].to_numpy()
    rows.add("armA", scope, "days_open", float(len(np.unique(d[is_open]))))
    rows.add("armA", scope, "days_closed", float(len(np.unique(d[~is_open]))))
    for c in (*HITS, *RETS):
        y = col_f(t, c)
        mo, mc = fmean(y[is_open]), fmean(y[~is_open])
        lo, hi = boot_diff(month, y, is_open, SEED)
        rows.add("armA", scope, f"{c}:open", mo, n=float(is_open.sum()))
        rows.add("armA", scope, f"{c}:closed", mc, n=float((~is_open).sum()))
        rows.add("armA", scope, f"{c}:diff", mo - mc)
        rows.add("armA", scope, f"{c}:diff_lo", lo)
        rows.add("armA", scope, f"{c}:diff_hi", hi)


# --- gate days -------------------------------------------------------------------------------


def day_table(rows: Rows) -> None:
    """Days with the gate open / closed / null, per year and per period (panel days)."""
    f = pl.read_parquet(FACTORS, columns=["d_us", GATE_FACTOR]).group_by("d_us").agg(pl.col(GATE_FACTOR).first())
    f = f.filter(
        (pl.col("d_us") >= FIRST_D_US)
        & ((pl.col("d_us") <= DISCOVERY_END_US) | ((pl.col("d_us") >= SEALED_START_US) & (pl.col("d_us") <= CONFIRM_END_US)))
    ).sort("d_us")
    st = gate_state(col_f(f, GATE_FACTOR))
    d = f["d_us"].to_numpy().astype(np.int64)
    year = np.array([utc(int(x)).year for x in d])
    scopes: list[tuple[str, npt.NDArray[np.bool_]]] = [(str(y), year == y) for y in (*DISC_YEARS, *CONF_YEARS)]
    scopes += [("2020-23", year <= 2023), ("2024-26", year >= 2024)]
    for name, m in scopes:
        rows.add("days", name, "open", float((st[m] == 1).sum()))
        rows.add("days", name, "closed", float((st[m] == 0).sum()))
        rows.add("days", name, "null", float((st[m] == -1).sum()))
        rows.add("days", name, "total", float(m.sum()))


# --- arm B -----------------------------------------------------------------------------------


def arm_b(
    rows_u: Rows, rows_g: Rows, scope: str, test: pl.DataFrame, pk_u: pl.DataFrame, seed: int
) -> None:
    """Arm B for one scope. `test`: the scope's scored entries (with `gate_ok`); `pk_u`: the
    ungated rule's picks. Writes the ungated numbers to `rows_u` and the gated ones (picks on
    open days, random draws on the open days' entries) to `rows_g`."""
    m5.scope_stats(rows_u, scope, test, pk_u, seed)
    pk_g = gated_picks(pk_u)
    assert bool((pk_g["gate_ok"]).all())
    assert pk_g.join(pk_u.select("symbol", "d_us"), on=["symbol", "d_us"], how="anti").height == 0
    if pk_g.height == 0:
        rows_g.add("picks", scope, "n_picks", 0.0)
        return
    m5.scope_stats(rows_g, scope, test.filter(pl.col("gate_ok")), pk_g, seed)


# --- csv -------------------------------------------------------------------------------------


def tidy(parts: list[tuple[str, pl.DataFrame]]) -> pl.DataFrame:
    cols = ["table", "variant", "scope", "stop", "stress", "metric", "value", "n"]
    return pl.concat(
        [f.with_columns(variant=pl.lit(v)).select(cols) for v, f in parts]
    )


# --- main ------------------------------------------------------------------------------------


def check_reproduction(spec: Spec, scored_d: pl.DataFrame, scored_c: pl.DataFrame, rows_u: Rows) -> None:
    """The ungated rule must reproduce M4's discovery bands and M5's confirm results."""
    band = spec.extra["expected_band_discovery"]["per_year"]
    ungated = Rows(list(rows_u.rows)).frame()
    look = {(r["table"], r["scope"], r["stop"], r["stress"], r["metric"]): r["value"] for r in ungated.iter_rows(named=True)}
    for y in WF_YEARS:
        b = band[str(y)]
        assert int(look[("picks", str(y), "", "", "n_picks")]) == int(b["pick_n"]), y
        assert int(look[("trades", str(y), "10", "base", "trades")]) == int(b["trades_10_none"]), y
        assert abs(look[("trades", str(y), "10", "base", "mean_ret")] - b["mean_ret_10_none"]) < 1e-9, y
        assert abs(look[("book", str(y), "10", "base", "book_return")] - b["book_return_10_none"]) < 1e-9, y
    m5csv = pl.read_csv(m5.OUT_CSV, schema_overrides={"scope": pl.String, "stop": pl.String, "stress": pl.String})
    ref = {(r["table"], r["scope"], r["stop"] or "", r["stress"] or "", r["metric"]): r["value"] for r in m5csv.iter_rows(named=True)}
    for scope in ("2024", "2025", "2026", "overall"):
        mine = "2024-26" if scope == "overall" else scope
        for key in (("picks", "n_picks", "", ""), ("trades", "trades", "10", "base"), ("trades", "mean_ret", "10", "base"), ("book", "book_return", "10", "base"), ("trades", "mean_ret", "10", "stress")):
            a = look[(key[0], mine, key[2], key[3], key[1])]
            b = ref[(key[0], scope, key[2], key[3], key[1])]
            assert abs(a - b) < 1e-6 * max(1.0, abs(b)), (scope, key, a, b)
    del scored_d, scored_c


def main() -> None:
    t0 = time.perf_counter()
    spec = load_spec(m5.SPEC_JSON)
    assert spec.rule.name == "k10_q0.95" and spec.extra["name"] == "rule_liq_dd_nogate" and spec.stop == "10"
    df = m5.build_frame()
    disc = df.filter(pl.col("split") == "discovery")
    conf = df.filter(pl.col("split") == "confirm")
    print("scoring the discovery walk-forward 2021-23 ...", flush=True)
    sd = with_gate(m5.score_years(spec, disc, WF_YEARS, discovery_only=True))
    print("scoring the confirm years ...", flush=True)
    sc = with_gate(m5.score_years(spec, df, CONF_YEARS, discovery_only=False))

    rows_a = Rows([])
    rows_days = Rows([])
    day_table(rows_days)
    disc_g, conf_g = with_gate(disc), with_gate(conf)
    for y in DISC_YEARS:
        arm_a(rows_a, str(y), disc_g.filter(pl.col("year") == y))
    arm_a(rows_a, "2020-23", disc_g)
    for y in CONF_YEARS:
        arm_a(rows_a, str(y), conf_g.filter(pl.col("year") == y))
    arm_a(rows_a, "2024-26", conf_g)
    print(f"arm A done {time.perf_counter() - t0:.0f}s", flush=True)

    rows_u, rows_g = Rows([]), Rows([])
    for scored, years, period in ((sd, WF_YEARS, "2021-23"), (sc, CONF_YEARS, "2024-26")):
        for y in years:
            ty = scored.filter(pl.col("year") == y)
            arm_b(rows_u, rows_g, str(y), ty, picks(ty, spec.rule), SEED + y)
            print(f"  arm B {y} done {time.perf_counter() - t0:.0f}s", flush=True)
        arm_b(rows_u, rows_g, period, scored, picks(scored, spec.rule), SEED)
        print(f"  arm B {period} done {time.perf_counter() - t0:.0f}s", flush=True)
    check_reproduction(spec, sd, sc, rows_u)
    print("ungated rule reproduces M4 (2021-23) and M5 (2024-26)")

    out = tidy(
        [
            ("all", rows_days.frame()),
            ("all", rows_a.frame()),
            ("ungated", rows_u.frame()),
            ("gate_open", rows_g.frame()),
        ]
    )
    out.write_csv(OUT_CSV, float_precision=6)
    write_md(out)
    print(f"wrote {OUT_CSV.name} ({out.height} rows) and {OUT_MD.name} in {time.perf_counter() - t0:.0f}s")


# --- markdown --------------------------------------------------------------------------------

ARM_A_SCOPES = ("2020", "2021", "2022", "2023", "2020-23", "2024", "2025", "2026", "2024-26")
ARM_B_SCOPES = ("2021", "2022", "2023", "2021-23", "2024", "2025", "2026", "2024-26")


def ci(lo: float, hi: float) -> str:
    return "n/a" if math.isnan(lo) else f"{lo * 100:+.1f}..{hi * 100:+.1f}"


def pp(x: float, d: int = 1) -> str:
    return "n/a" if math.isnan(x) else f"{x * 100:+.{d}f}"


def num(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:,.0f}"


def usd(x: float) -> str:
    return "n/a" if math.isnan(x) else f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


class Tidy:
    def __init__(self, df: pl.DataFrame) -> None:
        self.d: dict[tuple[str, str, str, str, str, str], float] = {
            (r["table"], r["variant"], r["scope"], r["stop"] or "", r["stress"] or "", r["metric"]): r["value"]
            for r in df.iter_rows(named=True)
        }

    def __call__(self, table: str, variant: str, scope: str, metric: str, stop: str = "", stress: str = "") -> float:
        return self.d.get((table, variant, scope, stop, stress, metric), math.nan)


def write_md(df: pl.DataFrame) -> None:
    g = Tidy(df)
    L: list[str] = []
    a = L.append
    a("# Market gate on history: description, not evidence (M7)")
    a("")
    a("**These tables describe the gate on 2020-26. They are not evidence, and nothing here")
    a("declares the gate confirmed or refuted.** The protocol (fixed 2026-11-14T22:30Z) says why:")
    a("")
    a(f"> {PROTOCOL_REASON}")
    a("")
    a("The verdict will come from `forward.py` on entries after the data end (2026-11-02), once at")
    a("least 180 days of them have complete windows and arm B has at least 100 trades (`results/forward.md`).")
    a("")
    a("**Gate.** Open on day `D` when BTC's last daily close is above the mean of its last 50 daily")
    a("closes (`m_btc_sma50 > 0`); closed otherwise, and when the factor is null. No tuning.")
    a("**Arm A**: every usable entry, gate open against closed (a null factor counts as closed).")
    a("**Arm B**: the frozen rule's picks (`results/spec.json`, unchanged) on gate-open days only, in")
    a("the same 10-slot book; positions open when the gate closes are held to their own exit.")
    a("Discovery arm B is the M4 walk-forward (2021-23, trained on earlier years only); confirm arm B")
    a("is M5's scoring (retrained each 1 January). The ungated rows reproduce M4 and M5 exactly (asserted).")
    a("Generated by `m7_gate.py` from `m7_gate.csv`.")
    a("")
    a("## Days with the gate open")
    a("")
    a("| year | days | open | closed | null | share open |")
    a("| --- | --- | --- | --- | --- | --- |")
    for s in ("2020", "2021", "2022", "2023", "2020-23", "2024", "2025", "2026", "2024-26"):
        o, c, n, t = (g("days", "all", s, k) for k in ("open", "closed", "null", "total"))
        a(f"| {s} | {t:.0f} | {o:.0f} | {c:.0f} | {n:.0f} | {pc(o / t, 0)}% |")
    a("")
    a("Days are the days with at least one panel entry (2026: to the last confirm `D`, 2026-10-02).")
    a("")
    a("## Arm A: every usable entry, gate open against closed")
    a("")
    a("Mean outcome of the entries on open days, on closed days, and the difference with a 95%")
    a("interval from a bootstrap over calendar months. Rates in %, differences in percentage points;")
    a("`ret_*` is the mean net return of the entry's trade (10% / 20% / no stop, fees and slippage in).")
    a("")
    labels = {"hit10": "hit10 (+30% before -10%)", "hit20": "hit20 (+30% before -20%)", "hit_ns": "hit_ns (+30% at any time)", "ret_10": "mean net return, 10% stop", "ret_20": "mean net return, 20% stop", "ret_ns": "mean net return, no stop"}
    for c in (*HITS, *RETS):
        a(f"### {labels[c]}")
        a("")
        a("| scope | entries open | entries closed | open | closed | open - closed | 95% interval |")
        a("| --- | --- | --- | --- | --- | --- | --- |")
        for s in ARM_A_SCOPES:
            a(
                f"| {s} | {num(g('armA', 'all', s, 'entries_open'))} | {num(g('armA', 'all', s, 'entries_closed'))} "
                f"| {pc(g('armA', 'all', s, f'{c}:open'))} | {pc(g('armA', 'all', s, f'{c}:closed'))} "
                f"| {pp(g('armA', 'all', s, f'{c}:diff'))} | {ci(g('armA', 'all', s, f'{c}:diff_lo'), g('armA', 'all', s, f'{c}:diff_hi'))} |"
            )
        a("")
    a("## Arm B: the frozen rule on gate-open days")
    a("")
    a("10% stop, 10 slots, $10,000 book per year and one continuous book per period. `hit10` of picks:")
    a("every pick before the slot rules. Mean net return per trade is the book's taken trades; stress")
    a("adds 0.3% a side. The random rows are same-day random books drawn on the gate-open days'")
    a("entries (100 draws, as many entries a day as the gated picks): median, with the 95th")
    a("percentile (5th to 95th for the book) in brackets. Fixed stake: $1,000 a trade, 10 open at most, no compounding.")
    a("")
    a("| scope | row | trades | hit10 of picks | mean net / trade | with 0.3% stress | month-bootstrap 95% (base) | book return | max drawdown | fixed-stake profit |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in ARM_B_SCOPES:
        for v, name in (("ungated", "ungated rule"), ("gate_open", "gated rule")):
            if math.isnan(g("picks", v, s, "hit10")):
                a(f"| {s} | {name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            a(
                f"| {s} | {name} | {g('trades', v, s, 'trades', '10', 'base'):.0f} | {pc(g('picks', v, s, 'hit10'))}% "
                f"| {pp(g('trades', v, s, 'mean_ret', '10', 'base'))}% | {pp(g('trades', v, s, 'mean_ret', '10', 'stress'))}% "
                f"| {ci(g('trades', v, s, 'mean_ret_ci_lo', '10', 'base'), g('trades', v, s, 'mean_ret_ci_hi', '10', 'base'))}% "
                f"| {pp(g('book', v, s, 'book_return', '10', 'base'), 0)}% | {pc(g('book', v, s, 'max_dd', '10', 'base'), 0)}% "
                f"| {usd(g('fixed', v, s, 'profit', '10', 'base'))} |"
            )
        v = "gate_open"
        if not math.isnan(g("picks", v, s, "hit10")):
            a(
                f"| {s} | random, open days | {g('random', v, s, 'trades_mean', '10', 'base'):.0f} "
                f"| {pc(g('random', v, s, 'pick_hit10_p50'))}% ({pc(g('random', v, s, 'pick_hit10_p95'))}) "
                f"| {pp(g('random', v, s, 'mean_ret_p50', '10', 'base'))}% ({pp(g('random', v, s, 'mean_ret_p95', '10', 'base'))}) "
                f"| {pp(g('random', v, s, 'mean_ret_p50', '10', 'stress'))}% ({pp(g('random', v, s, 'mean_ret_p95', '10', 'stress'))}) "
                f"| | {pp(g('random', v, s, 'book_return_p50', '10', 'base'), 0)}% ({pp(g('random', v, s, 'book_return_p5', '10', 'base'), 0)}..{pp(g('random', v, s, 'book_return_p95', '10', 'base'), 0)}) "
                f"| {pc(g('random', v, s, 'max_dd_p50', '10', 'base'), 0)}% | |"
            )
    a("")
    a("### Other stops (mean net return per trade, base / stress; book return)")
    a("")
    a("| scope | stop | ungated trades | ungated mean | ungated book | gated trades | gated mean | gated book | random book median (open days) |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in ARM_B_SCOPES:
        for stop, nm in (("20", "20%"), ("ns", "none")):
            def cell(v: str) -> tuple[str, str, str]:
                if math.isnan(g("picks", v, s, "hit10")):
                    return "0", "n/a", "n/a"
                return (
                    f"{g('trades', v, s, 'trades', stop, 'base'):.0f}",
                    f"{pp(g('trades', v, s, 'mean_ret', stop, 'base'))} / {pp(g('trades', v, s, 'mean_ret', stop, 'stress'))}%",
                    f"{pp(g('book', v, s, 'book_return', stop, 'base'), 0)}%",
                )

            u, o = cell("ungated"), cell("gate_open")
            rnd = g("random", "gate_open", s, "book_return_p50", stop, "base")
            a(f"| {s} | {nm} | {u[0]} | {u[1]} | {u[2]} | {o[0]} | {o[1]} | {o[2]} | {pp(rnd, 0)}% |")
    a("")
    a("## Share of time in the market, and how fast a gated test can run")
    a("")
    a("| scope | days open | share of days | trades, ungated (10% stop) | trades, gated | gated trades a year |")
    a("| --- | --- | --- | --- | --- | --- |")
    spans = {"2021": 1.0, "2022": 1.0, "2023": 11 / 12, "2021-23": 35 / 12, "2024": 1.0, "2025": 1.0, "2026": 9 / 12, "2024-26": 33 / 12}
    for s in ARM_B_SCOPES:
        o, t = g("days", "all", s if s != "2021-23" else "2020-23", "open"), g("days", "all", s if s != "2021-23" else "2020-23", "total")
        if s == "2021-23":
            o = sum(g("days", "all", y, "open") for y in ("2021", "2022", "2023"))
            t = sum(g("days", "all", y, "total") for y in ("2021", "2022", "2023"))
        if s == "2024-26":
            o, t = g("days", "all", "2024-26", "open"), g("days", "all", "2024-26", "total")
        tu, tg = g("trades", "ungated", s, "trades", "10", "base"), g("trades", "gate_open", s, "trades", "10", "base")
        tg = 0.0 if math.isnan(tg) else tg
        a(f"| {s} | {o:.0f} | {pc(o / t, 0)}% | {tu:.0f} | {tg:.0f} | {tg / spans[s]:.0f} |")
    a("")
    a("The 2023 span is 11 months (discovery `D` ends 2023-12-01) and 2026 is 9 months (confirm `D` ends")
    a("2026-10-02). A gate that is open for a third of the days thins a rule that already trades a few")
    a("hundred times a year: the forward test needs 100 gated trades and 180 days of complete windows,")
    a("so it cannot read out before the end of May 2027 (210 days after 2026-11-02) even if the gate\nis open all that time.")
    a("")
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
