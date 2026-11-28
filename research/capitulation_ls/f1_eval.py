"""F1 (pump shorts on perp-only coins): the fixed F rule (funding-gated pump short) and its ungated
twin P on USD-M perps without a spot market (p1 spikes) and in the pre-spot perp window of later
spot listings (p2 pre-spot spikes). Nothing is chosen here: the rule is x8_lib's F.

1. Per year 2022..2026: trades and mean net of F / P on perp-only, pre-spot and (reference) spot
   coins; day-block bootstrap CI of the mean.
2. diag.py scorecards (stopped share, giveback, ...).
3. Calibration by pump size (ret_4h bands, including 20-40% below the threshold, as if entered).
4. Books (10 slots, $10k a year): LMC+ (= LMC + L-perp-only, as p2_robust.py), + F (spot),
   + F-perp-only, + F-pre-spot.
5. Why it failed: stops, funding gate pass rate, repeat pumps, coin age, liquidity, concentration.

Writes OWN/f1/trades_*.parquet and results/explore/f_perp_only.md.
Run: uv run --group research python f1_eval.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import polars as pl
import x8_lib
from common_ls import OWN
from diag import annotate, scorecard, table
from e25_combined import book
from f1_lib import (
    F1,
    QV,
    R4,
    RF,
    diag_frame,
    gate,
    perp_only_pool,
    prespot_pool,
    short,
    to_book,
)
from futures_ls import perp_active, spot_name
from p2_lib import BOOK_COLS, YEARS, book_row, lmc, lperp, md, perp_only_spikes
from p2_lib import to_book as l_to_book
from p2_robust import day_boot
from sim import simulate

OUT = Path(__file__).resolve().parent / "results" / "explore" / "f_perp_only.md"
BANDS = (0.2, 0.3, 0.4, 0.6, 0.8, 1.2, np.inf)
KIND = {0: "time", 1: "stop", 2: "tp", 3: "liq", 4: "stop"}


def _mean(s: pl.Series) -> float:
    """Series mean as a float (nan if empty or all null)."""
    v = s.drop_nulls().cast(pl.Float64).to_numpy()
    return float(v.mean()) if len(v) else float("nan")


def _median(s: pl.Series) -> float:
    v = s.drop_nulls().cast(pl.Float64).to_numpy()
    return float(np.median(v)) if len(v) else float("nan")


# ---------------------------------------------------------------- spot reference ---------------
def spot_pool(lo: float = R4, hi: float = np.inf) -> pl.DataFrame:
    """x8 sleeves() F / P candidates on spot coins (feats.parquet), any ret_4h band."""
    f = pl.read_parquet(OWN / "feats.parquet")
    return f.filter(
        pl.col("perp") & (pl.col("qv_base") >= QV) & (pl.col("open_time").dt.year() >= 2022)
        & (pl.col("ret_4h") >= lo) & (pl.col("ret_4h") < hi)
    )  # fmt: skip


def spot_short(c: pl.DataFrame) -> pl.DataFrame:
    """x8 `_sim` for F on given feats rows (perp paths, funding by spot name)."""
    t = simulate(
        c.select("row", "ret_4h", "n_prior_pumps30", "qv_base", side=pl.lit(-1), prio="ret_4h"),
        RF, start_equity=1e12, perp_funding=True, source="perp",
    ).trades  # fmt: skip
    return t.with_columns(
        exit_kind=pl.col("kind").replace_strict(KIND, default="time"),
        t_in=pl.col("t") + 300,
        day=pl.col("open_time").dt.date(),
    )


def spot_gate(c: pl.DataFrame) -> pl.DataFrame:
    return c.filter(pl.col("rate").fill_null(0) >= 0)


# ---------------------------------------------------------------- features for "why" ----------
def prior_pumps(sp: pl.DataFrame, t: pl.DataFrame) -> np.ndarray:
    """build_feats n_prior_pumps30 on perp spikes: days with a spike of ret_4h >= 30% in the
    30 days before the trade's day (the current day excluded)."""
    days = (
        sp.select("symbol", d=pl.col("t") // 86400, p=pl.col("ret_4h") >= 0.3)
        .filter("p")
        .unique(["symbol", "d"])
    )
    dd = {k[0]: np.sort(g["d"].to_numpy()) for k, g in days.group_by(["symbol"])}
    out = []
    for s, ti in zip(t["symbol"].to_numpy(), t["t"].to_numpy(), strict=True):
        a = dd.get(s)
        d = ti // 86400
        out.append(0 if a is None else int(np.searchsorted(a, d) - np.searchsorted(a, d - 30)))
    return np.array(out)


def age_days(syms: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Days since the perp's first hourly candle (perp_1h)."""
    act = perp_active()
    out = np.full(len(syms), np.nan)
    for i, (s, ti) in enumerate(zip(syms, t, strict=True)):
        a = act.get(spot_name(str(s)))
        if a is not None:
            out[i] = (ti - a[0][0]) / 86400
    return out


# ---------------------------------------------------------------- tables -----------------------
def per_year(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows = []
    for name, t in frames.items():
        r: dict[str, object] = {"pool": name}
        for y in YEARS:
            s = t.filter(pl.col("year") == y) if t.height else t
            r[str(y)] = f"{s.height}, {_mean(s['net']) * 100:+.1f}%" if s.height else "0"
        if t.height:
            day = pl.from_epoch(t["t_in"], "s").dt.date().to_numpy()
            lo, hi = day_boot(t["net"].to_numpy(), day)
            r["all"] = f"{t.height}, {_mean(t['net']) * 100:+.1f}%"
            r["median"] = f"{_median(t['net']) * 100:+.1f}%"
            r["95% CI (days)"] = f"{lo * 100:+.1f} .. {hi * 100:+.1f}%"
            r["stopped"] = f"{_mean(t['exit_kind'] == 'stop'):.2f}"
            r["funding paid"] = f"{_mean(t['fund']) * 100:+.2f}%"
            r["coins"] = t["symbol"].n_unique()
        rows.append(r)
    return pl.DataFrame(rows)


def calib(pools: dict[str, tuple[pl.DataFrame, bool]]) -> pl.DataFrame:
    """Each band simulated as its own pool (one position per coin within the band)."""
    rows = []
    for name, (c, spot) in pools.items():
        for lo, hi in itertools.pairwise(BANDS):
            cc = c.filter((pl.col("ret_4h") >= lo) & (pl.col("ret_4h") < hi))
            t = spot_short(cc) if spot else short(cc)
            if t.height == 0:
                continue
            a = annotate(t.select(
                signal=pl.lit(1.0), threshold=pl.lit(0.9), net="net", mfe="mfe", mae="adv",
                tp=pl.lit(None, dtype=pl.Float64), stop=pl.lit(1.0), exit_kind="exit_kind",
                year="year",
            ))  # fmt: skip
            rows.append({
                "pool": name, "ret_4h": f"{lo:.0%}..{hi:.0%}" if np.isfinite(hi) else f">= {lo:.0%}",  # noqa: E501
                "entered": "yes" if lo >= R4 else "no", "n": t.height,
                "E_o": float(_mean(a["o"])), "median_o": float(_median(a["o"])),
                "win": float(_mean(a["o"] > 0)), "p_stopped": float(_mean(a["mode"] == "stopped")),
                "giveback_med": float(_median(a["giveback"])),
            })  # fmt: skip
    return pl.DataFrame(rows)


def book_line(name: str, parts: list[pl.DataFrame]) -> dict[str, object]:
    b = book(pl.concat(parts, how="vertical_relaxed"), 10)
    r = book_row(name, b)
    r["worst dd"] = f"{float(b['dd'].to_numpy().min()) * 100:.0f}%"
    r["F trades"], r["F bp"] = 0, 0
    if "nF" in b.columns:
        nf = b["nF"].fill_null(0).to_numpy()
        bf = b["bpF"].fill_null(0).to_numpy()
        r["F trades"] = int(nf.sum())
        r["F bp"] = round(float((bf * nf).sum() / max(nf.sum(), 1)))
    return r


# ---------------------------------------------------------------- main -------------------------
def main() -> None:
    sp_po = perp_only_spikes()
    c_po = perp_only_pool()
    c_pre = prespot_pool()
    c_sp = spot_pool()
    f_po, p_po = short(gate(c_po)), short(c_po)
    f_pre, p_pre = short(gate(c_pre), prespot=True), short(c_pre, prespot=True)
    f_sp, p_sp = spot_short(spot_gate(c_sp)), spot_short(c_sp)
    x8f = x8_lib.sleeves().filter(pl.col("sleeve") == "F")
    assert f_sp.height == x8f.height, (f_sp.height, x8f.height)
    assert abs(_mean(f_sp["net"]) - _mean(x8f["net"])) < 1e-9
    for k, t in (("F_perp_only", f_po), ("P_perp_only", p_po), ("F_prespot", f_pre),
                 ("P_prespot", p_pre)):  # fmt: skip
        if t.height:
            t.drop("day").write_parquet(F1 / f"trades_{k}.parquet")
    lines = [
        "# F1. Pump shorts (F, P) on perp-only coins",
        "",
        "Scripts: `f1_lib.py` (pools, perp simulation), `f1_eval.py` (this note). In-sample report of a fixed",  # noqa: E501
        "rule; nothing chosen. F = x8_lib `sleeves()` F: 5m liquid spike rows with ret_4h >= +40%, perp active,",  # noqa: E501
        "qv_base >= 20k, last settled funding rate >= 0 at the signal bar (none = 0), short the perp at the next",  # noqa: E501
        "5m open, isolated 1x (stop at +100% adverse), 72h, cost 0.3% + funding, one position per coin.",  # noqa: E501
        "P = F without the funding gate. Perp-only = p1 spikes (USD-M perps with no spot market); pre-spot =",  # noqa: E501
        "p2 pre-spot spikes (perp window before a later spot listing). Funding keyed by the perp symbol.",  # noqa: E501
        "The spot rows are the reference (re-simulated; F spot matches x8 sleeves() exactly: "
        f"{x8f.height} trades, {_mean(x8f['net']) * 100:+.2f}%).",
        "",
        f"Candidate rows (ret_4h >= 40%, active perp, qv_base >= 20k, 2022+): perp-only {c_po.height:,} "  # noqa: E501
        f"({gate(c_po).height:,} pass the gate), pre-spot {c_pre.height} ({gate(c_pre).height}), "
        f"spot {c_sp.height:,} ({spot_gate(c_sp).height:,}). Perp-only coins are mostly 2025-26 listings (p1 spikes:"  # noqa: E501
        " 1 coin in 2022, 8 in 2023, 37 in 2024, 166 in 2025, 183 in 2026), so 2022-24 are near empty.",  # noqa: E501
        "",
        "## 1. Per year (trades, mean net)",
        "",
        md(
            per_year(
                {
                    "F perp-only": f_po,
                    "P perp-only": p_po,
                    "F pre-spot": f_pre,
                    "P pre-spot": p_pre,
                    "F spot (ref)": f_sp,
                    "P spot (ref)": p_sp,
                }
            )
        ),
        "",
    ]
    # ------------------------------------------------ 2. scorecards
    cards = {
        "F perp-only": scorecard(annotate(diag_frame(f_po))),
        "P perp-only": scorecard(annotate(diag_frame(p_po))),
        "F spot (ref)": scorecard(annotate(diag_frame(f_sp))),
        "P spot (ref)": scorecard(annotate(diag_frame(p_sp))),
    }
    if f_pre.height:
        cards["F pre-spot"] = scorecard(annotate(diag_frame(f_pre)))
    if p_pre.height:
        cards["P pre-spot"] = scorecard(annotate(diag_frame(p_pre)))
    lines += ["## 2. diag scorecards", "", table(cards), ""]
    # ------------------------------------------------ 3. calibration by pump size
    c_po2 = perp_only_pool(lo=0.2)
    c_sp2 = spot_pool(lo=0.2)
    cal = calib({
        "F perp-only": (gate(c_po2), False), "P perp-only": (c_po2, False),
        "F spot (ref)": (spot_gate(c_sp2), True), "P spot (ref)": (c_sp2, True),
    })  # fmt: skip
    lines += [
        "## 3. Calibration by pump size",
        "",
        "Each ret_4h band simulated as its own pool (one position per coin within the band); bands below",  # noqa: E501
        "40% are not entered by the rule (as if entered). o = net clipped to [-1, 1].",
        "",
        md(cal),
        "",
    ]
    # ------------------------------------------------ 4. books
    base = lmc()
    plus = [*base, l_to_book(lperp(), "L").select(BOOK_COLS)]
    fs = x8_lib.as_book(x8f).with_columns(w=pl.lit(1.0)).select(BOOK_COLS)
    bpo = to_book(f_po)
    bpre = to_book(f_pre, spot_sym=True) if f_pre.height else bpo.head(0)
    rows = [
        book_line("LMC+", plus),
        book_line("LMC+ + F (spot)", [*plus, fs]),
        book_line("LMC+ + F-perp-only", [*plus, bpo]),
        book_line("LMC+ + F (spot) + F-perp-only", [*plus, fs, bpo]),
        book_line("LMC+ + F (spot) + F-perp-only + F-pre-spot", [*plus, fs, bpo, bpre]),
        book_line("LMC+ + F (spot) + P-perp-only (info)", [*plus, fs, to_book(p_po)]),
    ]
    lines += [
        "## 4. Books (10 slots, $10k each year, stake <= 0.5% of daily volume)",
        "",
        "LMC+ = e25 L, M, C + L-perp-only (p2_robust.py). F rows: x8 priority 2 + ret_4h. Per year: return (max drawdown).",  # noqa: E501
        "",
        md(pl.DataFrame(rows)),
        "",
    ]
    # ------------------------------------------------ 5. why
    lines += why(sp_po, c_po, f_po, p_po, f_sp, p_sp, c_sp)
    i = lines.index("## 1. Per year (trades, mean net)")
    lines[i:i] = verdict(f_po, p_po, f_pre, f_sp, p_sp, c_po, c_sp, rows)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def verdict(
    f_po: pl.DataFrame, p_po: pl.DataFrame, f_pre: pl.DataFrame, f_sp: pl.DataFrame,
    p_sp: pl.DataFrame, c_po: pl.DataFrame, c_sp: pl.DataFrame, rows: list[dict[str, object]],
) -> list[str]:  # fmt: skip
    def m(t: pl.DataFrame) -> str:
        return f"{_mean(t['net']) * 100:+.1f}%"

    def st(t: pl.DataFrame) -> str:
        return f"{_mean(t['exit_kind'] == 'stop') * 100:.0f}%"

    b = {str(r["book"]): r for r in rows}
    a, c = b["LMC+ + F (spot)"], b["LMC+ + F (spot) + F-perp-only"]
    return [
        "## Verdict: F does not carry over to perp-only coins",
        "",
        f"* F perp-only: {f_po.height} trades, {m(f_po)}/trade (median {_median(f_po['net']) * 100:+.1f}%), "  # noqa: E501
        f"{st(f_po)} stopped; on spot coins F is {f_sp.height} trades, {m(f_sp)}, {st(f_sp)} stopped. "  # noqa: E501
        f"P perp-only: {p_po.height} trades, {m(p_po)}, {st(p_po)} stopped. Pre-spot window: F "
        f"{f_pre.height} trades, {m(f_pre) if f_pre.height else '-'} (too few to read). Almost all "
        "perp-only trades are 2025-26 (none in 2022, 1-6 a year in 2023-24).",
        "* The typical trade works as on spot (median and non-stopped mean are the same or better); "  # noqa: E501
        "the loss is the squeeze tail: about 1 in 5 entries runs to +100% within 72h (closes, not wicks).",  # noqa: E501
        f"* The funding gate is inert there: funding is >= 0 on {_mean(c_po['rate'].fill_nan(None).fill_null(0) >= 0) * 100:.0f}% "  # noqa: E501
        f"of perp-only pump rows (spot {_mean(c_sp['rate'].fill_null(0) >= 0) * 100:.0f}%), and P -> F moves the mean "  # noqa: E501
        f"{m(p_po)} -> {m(f_po)} (spot {m(p_sp)} -> {m(f_sp)}).",
        f"* Books: LMC+ + F (spot) {a['2025']} / {a['2026']} in 2025 / 2026, worst dd {a['worst dd']}; "  # noqa: E501
        f"adding F-perp-only gives {c['2025']} / {c['2026']}, worst dd {c['worst dd']}, min year "
        f"{a['min yr']} -> {c['min yr']}. Do not add F-perp-only (or P-perp-only).",
        "",
    ]


def why(
    sp_po: pl.DataFrame,
    c_po: pl.DataFrame,
    f_po: pl.DataFrame,
    p_po: pl.DataFrame,
    f_sp: pl.DataFrame,
    p_sp: pl.DataFrame,
    c_sp: pl.DataFrame,
) -> list[str]:
    out = ["## 5. Why it failed (numbers)", ""]
    po = f_po.with_columns(
        prior=pl.Series(prior_pumps(sp_po, f_po)),
        age=pl.Series(age_days(f_po["symbol"].to_numpy(), f_po["t"].to_numpy())),
    )
    s = f_sp.with_columns(
        prior=pl.col("n_prior_pumps30"),
        age=pl.Series(age_days(f_sp["symbol"].to_numpy(), f_sp["t"].to_numpy())),
    )

    def cut(t: pl.DataFrame, name: str) -> list[dict[str, object]]:
        r = []
        for lab, e in (
            ("all", pl.lit(True)),
            ("not stopped", pl.col("exit_kind") != "stop"),
            ("stopped", pl.col("exit_kind") == "stop"),
            ("first pump (no prior 30d)", pl.col("prior") == 0),
            ("repeat pump (prior 30d)", pl.col("prior") > 0),
            ("perp age < 90d", pl.col("age") < 90),
            ("perp age 90-365d", (pl.col("age") >= 90) & (pl.col("age") < 365)),
            ("perp age >= 365d", pl.col("age") >= 365),
            ("qv_base < 100k", pl.col("qv_base") < 100_000),
            ("qv_base >= 100k", pl.col("qv_base") >= 100_000),
            ("2025-26", pl.col("year") >= 2025),
        ):
            g = t.filter(e)
            r.append({
                "pool": name, "cut": lab, "n": g.height,
                "share": g.height / t.height,
                "mean net": float(_mean(g["net"])) if g.height else float("nan"),
                "median net": float(_median(g["net"])) if g.height else float("nan"),
                "stopped": float(_mean(g["exit_kind"] == "stop")) if g.height else float("nan"),
                "mean price ret": float(_mean(g["ret"])) if g.height else float("nan"),
                "mfe med": float(_median(g["mfe"])) if g.height else float("nan"),
                "funding paid": float(_mean(g["fund"])) if g.height else float("nan"),
            })  # fmt: skip
        return r

    out += [
        "F trades cut by stop, repeat pumps, perp age (days since the perp's first candle) and liquidity:",  # noqa: E501
        "",
        md(pl.DataFrame(cut(po, "F perp-only") + cut(s, "F spot (ref)"))),
        "",
    ]
    # funding gate
    r_po = c_po["rate"].fill_nan(None)
    r_sp = c_sp["rate"]
    out += [
        "Funding at the signal bar (candidate rows):",
        "",
        md(
            pl.DataFrame(
                [
                    {
                        "pool": "perp-only",
                        "rows": c_po.height,
                        "no rate yet": int(r_po.is_null().sum()),
                        "rate >= 0 (pass)": _mean(r_po.fill_null(0) >= 0),
                        "rate < 0": float((r_po < 0).sum() / c_po.height),
                        "rate <= -0.1%": float((r_po <= -0.001).sum() / c_po.height),
                        "median rate": _median(r_po),
                    },
                    {
                        "pool": "spot (ref)",
                        "rows": c_sp.height,
                        "no rate yet": int(r_sp.is_null().sum()),
                        "rate >= 0 (pass)": _mean(r_sp.fill_null(0) >= 0),
                        "rate < 0": float((r_sp < 0).sum() / c_sp.height),
                        "rate <= -0.1%": float((r_sp <= -0.001).sum() / c_sp.height),
                        "median rate": _median(r_sp),
                    },
                ]
            ),
            5,
        ),
        "",
        f"Trades the gate removes (P minus F): perp-only {p_po.height - f_po.height} "
        f"(P {_mean(p_po['net']) * 100:+.1f}% vs F {_mean(f_po['net']) * 100:+.1f}%); spot "
        f"{p_sp.height - f_sp.height} (P {_mean(p_sp['net']) * 100:+.1f}% vs F {_mean(f_sp['net']) * 100:+.1f}%).",  # noqa: E501
        "",
    ]
    # concentration
    cs = (
        po.group_by("symbol")
        .agg(n=pl.len(), sum_net=pl.col("net").sum(), stops=(pl.col("exit_kind") == "stop").sum())
        .sort("sum_net")
    )
    worst = cs.head(10)
    tot = float(po["net"].sum())
    ex10 = po.filter(~pl.col("symbol").is_in(worst["symbol"].implode()))
    out += [
        f"Concentration: {cs.height} coins, sum of net {tot:+.2f} stakes; the 10 worst coins sum to "  # noqa: E501
        f"{float(worst['sum_net'].sum()):+.2f} ({int(worst['n'].sum())} trades, {int(worst['stops'].sum())} stops). "  # noqa: E501
        f"Without them: n {ex10.height}, mean {_mean(ex10['net']) * 100:+.1f}%, stopped "
        f"{_mean(ex10['exit_kind'] == 'stop'):.2f}. Coins with a positive sum: "
        f"{int((cs['sum_net'] > 0).sum())} of {cs.height}.",
        "",
        md(worst.with_columns(pl.col("sum_net").round(2))),
        "",
    ]
    # path shape
    pp_po = po_paths(f_po)
    pp_sp = sim_perp_paths(f_sp["row"].to_numpy())
    out += [
        "Price path after entry (perp close relative to the entry, short's view = minus the move; "
        "ignores the stop), and the share whose high reached +50% / +100% within 72h:",
        "",
        md(pl.DataFrame([horizons("F perp-only", pp_po), horizons("F spot (ref)", pp_sp)]), 3),
        "",
    ]
    # stops: wick or sustained?
    srow = []
    for name, t, p in (("F perp-only", f_po, pp_po), ("F spot (ref)", f_sp, pp_sp)):
        st = (t["exit_kind"] == "stop").to_numpy()
        cl = np.nanmax(p[:, 2, : IDX["72h"] + 1], axis=1)
        srow.append({
            "pool": name, "stopped": int(st.sum()),
            "max close >= +100% in 72h": int((st & (cl >= 1.0)).sum()),
            "max close +50..100% (stop on a wick)": int((st & (cl >= 0.5) & (cl < 1.0)).sum()),
            "max close < +50% (stop on a wick)": int((st & (cl < 0.5)).sum()),
            "close at 72h >= +100%": int((st & (p[:, 2, IDX["72h"]] >= 1.0)).sum()),
            "median hours to stop": float(np.median((t["t_out"] - t["t_in"]).to_numpy()[st]) / 3600) if st.any() else float("nan"),  # noqa: E501
        })  # fmt: skip
    out += [
        "Were the stops wicks? (stop = the perp's last-price high reached +100%; a real isolated 1x short is"  # noqa: E501
        " liquidated on the mark price, which ignores single-print wicks):",
        "",
        md(pl.DataFrame(srow), 1),
        "",
    ]
    q = (
        po.with_columns(q=pl.from_epoch("t_in", "s").dt.strftime("%Y-Q") + pl.from_epoch("t_in", "s").dt.quarter().cast(pl.Utf8))  # noqa: E501
        .group_by("q")
        .agg(n=pl.len(), coins=pl.col("symbol").n_unique(), mean_net=pl.col("net").mean(),
             median_net=pl.col("net").median(), stopped=(pl.col("exit_kind") == "stop").mean())
        .sort("q")
    )  # fmt: skip
    out += ["F perp-only by quarter of entry:", "", md(q), ""]
    return out


IDX = {"4h": 47, "24h": 111, "72h": 159, "7d": 255}


def po_paths(t: pl.DataFrame) -> np.ndarray:
    """Perp paths of perp-only trades, aligned with the rows of t."""
    import p1_lib

    out = np.zeros((t.height, 3, 256), np.float32)
    ti = t.with_row_index("_i")
    for (sym,), g in ti.group_by(["symbol"]):
        ok, p = p1_lib.perp_paths(str(sym), g["t"].to_numpy())
        assert ok.all()
        out[g["_i"].to_numpy()] = p
    return out


def sim_perp_paths(rows: np.ndarray) -> np.ndarray:
    """Perp paths of spot trades (build_perp_paths store), aligned with rows."""
    from sim import perp_paths

    p, ok = perp_paths(rows)
    assert ok.all()
    return p


def horizons(name: str, p: np.ndarray) -> dict[str, object]:
    r: dict[str, object] = {"pool": name, "n": len(p)}
    for k, i in IDX.items():
        r[f"median short move {k}"] = float(np.nanmedian(-p[:, 2, i]))
        r[f"mean short move {k}"] = float(np.nanmean(-p[:, 2, i]))
    hi = np.nanmax(p[:, 0, : IDX["72h"] + 1], axis=1)
    r["high >= +50% (72h)"] = float(np.mean(hi >= 0.5))
    r["high >= +100% (72h)"] = float(np.mean(hi >= 1.0))
    return r


if __name__ == "__main__":
    main()
