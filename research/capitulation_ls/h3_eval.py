"""H3 (other quotes, 2020-26): the long sleeve L, unchanged, on the USDT-converted BUSD / BTC /
FDUSD / USDC / TUSD / ETH / BNB pairs of coins without a USDT spot pair at the time (h3_build.py,
h3_score.py), and the 10-slot LMC+ book with these trades added.

- L: pct >= 0.995 (the fold of the spike's year; 2020-21: the 2022 fold applied backwards), entry
  at the next 5m open, TP +20%, time exit 72h, one position per coin (`h1_eval.trades`, which
  runs `sim.simulate` at 0.4% round trip; the meta's `symbol` is the coin, so two quote pairs of
  one coin share the lock). Wide cost: 1.2% round trip (alt leg + the quote/USDT conversion
  legs), subtracted from the 0.4% net as in h2_eval (`h2_eval.shifted`).
- New flush days: against the spot USDT L of the study (e25 `sleeve_long`, 2022-26; for 2020-21
  the same rule on the study's USDT spikes scored with the 2022 fold, h3_score.py) and against
  the perp-only L (`p2_lib.lperp`, p1, 2022-26).
- Calibration: per-spike fixed pct bins (`h1_eval.per_spike` / `fixed_bins`) and the diag
  deciles over the pct >= 0.95 pool, 2022-26 (walk-forward) and 2020-21 (backwards) apart.
- Books: LMC+ (`p2_lib.lmc()` + `p2_lib.to_book(lperp())`) and LMC+ with these trades added as
  L book rows (stake cap 0.5% of the pair's 24h quote volume in USDT: qv_base x 288 of the
  converted series), `e25_combined.book`, 10 slots, $10k fresh each year, 2022 .. 2026; costs
  0.4% and 1.2% for the h3 trades. 2020-21: L-only books (USDT L with the backward fold, with and
  without the h3 trades).

Writes OWN/h3/{eval_tables.md, l_trades.parquet} and prints the tables.
Run: uv run --group research python h3_eval.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date

import numpy as np
import polars as pl
import sim
from diag import annotate, calibration, scorecard, table, threshold_reach
from e25_combined import book, sleeve_long
from h1_eval import (
    YFMT,
    baselines,
    by_year,
    clustered,
    diag_frame,
    fixed_bins,
    md,
    per_spike,
    sleeve,
    trades,
)
from h2_eval import shifted
from h3_fetch import H3, split
from p2_lib import BOOK_COLS, lmc, lperp, to_book

WIDE = 0.012 - 0.004  # extra cost over the 0.4% net
ALL_Y = (2020, 2021, 2022, 2023, 2024, 2025, 2026)
WF_Y = (2022, 2023, 2024, 2025, 2026)
BACK_Y = (2020, 2021)
out: list[str] = []


def emit(s: str = "") -> None:
    out.append(s)
    print(s, flush=True)


def use_h3() -> pl.DataFrame:
    m = pl.read_parquet(H3 / "meta.parquet").with_columns(
        year=pl.col("open_time").dt.year(), day=pl.col("open_time").dt.date()
    )
    f2 = np.load(H3 / "fwd2.npy", mmap_mode="r")
    f72 = np.load(H3 / "fwd72.npy", mmap_mode="r")
    sim.meta, sim.fwd2, sim.fwd72 = (lambda: m), (lambda: f2), (lambda: f72)  # type: ignore[assignment,attr-defined]
    return m


def days_of(t: pl.DataFrame, col: str = "day") -> dict[int, set[date]]:
    d: dict[int, set[date]] = defaultdict(set)
    for x in t[col].to_list():
        d[x.year].add(x)
    return d


def h3_book_rows(t: pl.DataFrame) -> pl.DataFrame:
    return sleeve(t).with_columns(w=pl.lit(1.0)).select(BOOK_COLS)


def book_line(name: str, parts: list[pl.DataFrame], years: tuple[int, ...]) -> dict[str, object]:
    parts = [p for p in parts if p.height]
    if not parts:
        return {"book": name, "trades": 0}
    b = book(pl.concat(parts, how="vertical_relaxed"), 10).filter(pl.col("year").is_in(years))
    n = int(b["trades"].sum())
    row: dict[str, object] = {
        "book": name, "trades": n,
        "bp": round(float((b["bp"] * b["trades"]).sum() / n)) if n else 0,
    }  # fmt: skip
    yrs = {int(r["year"]): r for r in b.iter_rows(named=True)}
    for y in years:
        r = yrs.get(y)
        row[str(y)] = f"{r['ret'] * 100:+.1f}% (dd {r['dd'] * 100:.1f}, nL {r['nL']})" if r else "-"
    rets = [float(r["ret"]) for y, r in yrs.items() if y in years]
    row["worst yr"] = f"{min(rets) * 100:+.1f}%" if rets else "-"
    row["max dd"] = f"{min(float(r['dd']) for r in yrs.values()) * 100:.1f}%" if yrs else "-"
    return row


def main() -> None:
    pl.Config.set_tbl_rows(200)
    # ---- reference sleeves on the study's own data (default sim loaders) ----
    base = lmc()
    lp = lperp()
    lp_rows = to_book(lp, "L").select(BOOK_COLS)
    spot_l = sleeve_long()
    spot_days = days_of(spot_l.with_columns(day=pl.from_epoch(pl.col("t_in") - 300).dt.date()))
    perp_days = days_of(lp)
    um = sim.meta()  # type: ignore[attr-defined]
    u = pl.read_parquet(H3 / "usdt_2020_21.parquet")
    upct = np.full(um.height, np.nan)
    upct[u["row"].to_numpy()] = u["pct"].to_numpy()
    ul = trades(um, upct, BACK_Y, 0.995)
    for y, s in days_of(ul).items():
        spot_days[y] |= s
    ul_rows = sleeve(ul).with_columns(w=pl.lit(1.0)).select(BOOK_COLS)

    # ---- h3 data ----
    m = use_h3()
    sc = pl.read_parquet(H3 / "scores.parquet").sort("row")
    pct = sc["pct"].to_numpy().astype(np.float64)
    yr = m["year"].to_numpy()

    emit("## Coverage\n")
    plan = json.loads((H3 / "plan.json").read_text())
    stored = sorted(
        p.name.removeprefix("symbol=")
        for p in (H3 / "pairs/klines/spot/interval=5m").glob("symbol=*")
    )
    rows = []
    for name in stored:
        sp = split(name)
        if sp is None:  # USDTUSD (a stablecoin pair stored by mistake, not used)
            continue
        for f in (H3 / "pairs/klines/spot/interval=5m" / f"symbol={name}").glob("*.parquet"):
            rows.append({"pair": name, "quote": sp[1], "year": int(f.stem[:4]), "month": f.stem})
    st = pl.DataFrame(rows).filter(pl.col("year") >= 2020)
    emit(f"- pairs planned {len(plan)}, symbol-months planned {sum(map(len, plan.values()))},"
         f" pairs stored {len(stored)}, symbol-months stored {len(rows)}")  # fmt: skip
    cov = json.loads((H3 / "coverage.json").read_text())
    tot = {k: sum(v[k] for v in cov.values()) for k in ("liquid", "no_usdt", "kept")}
    emit(f"- liquid spikes on the stored months (2020-01-02 .. 2026-10-28) {tot['liquid']}, of which"  # noqa: E501
         f" while the coin had no USDT pair {tot['no_usdt']}, after dropping live p1 perps"
         f" (2022+) {tot['kept']}\n")  # fmt: skip
    pq = st.group_by("quote", "year").agg(pl.col("pair").n_unique()).sort("year")
    emit("Pairs with stored months, per quote and year:\n")
    emit(md(pq.pivot("year", index="quote", values="pair").fill_null(0)))
    emit("")
    sq = m.group_by("quote", "year").agg(pl.len()).sort("year")
    emit("Kept liquid spikes, per quote and year:\n")
    emit(md(sq.pivot("year", index="quote", values="len").fill_null(0)))
    emit("")
    cv = (
        m.group_by("year")
        .agg(spikes=pl.len(), coins=pl.col("symbol").n_unique(), pairs=pl.col("pair").n_unique(),
             med_qv_base=pl.col("qv_base").median())
        .sort("year")
    )  # fmt: skip
    sh = (
        pl.DataFrame({"year": yr, "pct": pct})
        .group_by("year")
        .agg(ge_0995=(pl.col("pct") >= 0.995).mean(), nan=pl.col("pct").is_nan().sum())
        .sort("year")
    )
    emit(md(cv.join(sh, on="year"), {"med_qv_base": ".0f", "ge_0995": ".4f"}))
    emit("")
    emit("## Baselines (L's exit, one position per coin, 0.4%)\n")
    emit("\n".join(baselines(m, WF_Y, "h3 2022-26")))
    emit("\n".join(baselines(m, BACK_Y, "h3 2020-21")) + "\n")

    # ---- L trades ----
    lt = trades(m, pct, ALL_Y, 0.995)
    lt = lt.join(m.with_row_index("row").select(pl.col("row").cast(lt["row"].dtype), "pair",
                                                "quote"), on="row", how="left")  # fmt: skip
    lt.write_parquet(H3 / "l_trades.parquet")
    emit("## L trades by year (pct >= 0.995, TP 20%, 72h; mean_bp at 0.4%)\n")
    yb = by_year(lt)
    rows = []
    for r in yb.iter_rows(named=True):
        g = lt.filter(pl.col("year") == r["year"])
        d = set(g["day"].to_list())
        sd, pdys = spot_days.get(r["year"], set()), perp_days.get(r["year"], set())
        rows.append({
            "year": r["year"], "n": r["n"], "mean_bp": r["mean_bp"],
            "mean_bp_1p2": r["mean_bp"] - WIDE * 1e4, "tp_rate": r["tp_rate"], "win": r["win"],
            "days": len(d), "new_vs_spot": len(d - sd), "new_vs_perp": len(d - pdys),
            "new_vs_both": len(d - sd - pdys), "spot_L_days": len(sd), "perp_L_days": len(pdys),
            "coins": r["coins"],
            "quotes": ",".join(f"{q}:{c}" for q, c in sorted(
                g.group_by("quote").agg(pl.len()).iter_rows(), key=lambda x: -x[1])),
        })  # fmt: skip
    yt = pl.DataFrame(rows)
    emit(md(yt, {**YFMT, "mean_bp_1p2": "+.0f"}))
    emit("")
    for label, ys in (
        ("2022-26 (walk-forward folds)", WF_Y),
        ("2020-21 (2022 fold backwards)", BACK_Y),
    ):
        g = lt.filter(pl.col("year").is_in(ys))
        if g.height == 0:
            emit(f"{label}: no trades\n")
            continue
        ex = g.group_by("exit_kind").agg(n=pl.len(), mean_bp=pl.col("net").mean() * 1e4)
        emit(f"**{label}**: n {g.height}, mean {g['net'].mean() * 1e4:+.0f} bp (1.2%:"
             f" {(g['net'].mean() - WIDE) * 1e4:+.0f}), median {g['net'].median() * 1e4:+.0f},"
             f" TP {(g['kind'] == 2).mean():.2f}, days {g['day'].n_unique()}; exits "
             + ", ".join(f"{r['exit_kind']} {r['n']} ({r['mean_bp']:+.0f} bp)" for r in ex.iter_rows(named=True)))  # fmt: skip  # noqa: E501
        emit("")
        emit(clustered(g, f"h3 {label} 0.4%"))
        emit(clustered(shifted(g, WIDE), f"h3 {label} 1.2%") + "\n")
    emit("### L trades (all)\n")
    emit(md(lt.select("day", "symbol", "pair", "signal", "net", "exit_kind", "qv").sort("day"),
            {"signal": ".4f", "net": "+.3f", "qv": ".0f"}))  # fmt: skip
    emit("")
    ul_y = by_year(ul)
    emit("Reference, spot USDT L 2020-21 with the 2022 fold backwards (study's USDT spikes):\n")
    emit(md(ul_y, YFMT))
    emit("")

    # ---- calibration ----
    for label, ys in (("2022-26", WF_Y), ("2020-21 (backwards)", BACK_Y)):
        emit(f"## Calibration {label} (0.4%)\n")
        hy = np.isin(yr, ys)
        ps = per_spike(m, pct, ys, 0.9)
        emit("Per spike, fixed pct bins (no per-coin lock):\n")
        emit(md(fixed_bins(ps, int((hy & np.isfinite(pct)).sum())),
                {"mean_bp": "+.0f", "tp_rate": ".2f", "share": ".4f"}))  # fmt: skip
        emit("")
        g = lt.filter(pl.col("year").is_in(ys))
        pool = trades(m, pct, ys, 0.95, 0.995)
        if g.height and pool.height:
            fe, fn = diag_frame(g), diag_frame(pool)
            emit(table({f"L h3 {label}": scorecard(annotate(fe))}))
            tr = threshold_reach(fe, fn)
            emit(f"Spearman(signal, outcome) over the pct >= 0.95 pool"
                 f" {tr['spearman_signal_outcome']:+.3f}\n")  # fmt: skip
            emit("By signal decile (pool pct >= 0.95, one per coin):\n")
            emit(md(calibration(fe, fn), {"bin": ".4f", "E_o": "+.3f"}))
            emit("")
        nb = []
        for q in (0.99, 0.995, 0.998):
            t = trades(m, pct, ys, q)
            nb.append({"rule": f"pct>={q}", "n": t.height,
                       "mean_bp": float(t["net"].mean() or 0) * 1e4 if t.height else 0.0,
                       "tp_rate": float((t["kind"] == 2).mean()) if t.height else 0.0,
                       "days": t["day"].n_unique() if t.height else 0})  # fmt: skip
        emit(md(pl.DataFrame(nb), {"mean_bp": "+.0f", "tp_rate": ".2f"}))
        emit("")

    # ---- books ----
    emit("## 10-slot book, 2022-26 ($10k fresh each year)\n")
    h3r = h3_book_rows(lt.filter(pl.col("year").is_in(WF_Y)))
    lines = [
        book_line("LMC+ (reference)", [*base, lp_rows], WF_Y),
        book_line("LMC+ + h3 L, 0.4%", [*base, lp_rows, h3r], WF_Y),
        book_line(
            "LMC+ + h3 L, 1.2%", [*base, lp_rows, h3r.with_columns(net=pl.col("net") - WIDE)], WF_Y
        ),
        book_line("L spot + L perp (reference)", [base[0], lp_rows], WF_Y),
        book_line(
            "L spot + L perp + h3 L, 1.2%",
            [base[0], lp_rows, h3r.with_columns(net=pl.col("net") - WIDE)],
            WF_Y,
        ),
        book_line("h3 L alone, 0.4%", [h3r], WF_Y),
    ]
    emit(md(pl.DataFrame(lines)))
    emit("")
    emit("## 10-slot L-only book, 2020-21 (2022 fold backwards)\n")
    h3b = h3_book_rows(lt.filter(pl.col("year").is_in(BACK_Y)))
    lines = [
        book_line("USDT L (backwards)", [ul_rows], BACK_Y),
        book_line("USDT L + h3 L, 0.4%", [ul_rows, h3b], BACK_Y),
        book_line(
            "USDT L + h3 L, 1.2%", [ul_rows, h3b.with_columns(net=pl.col("net") - WIDE)], BACK_Y
        ),
        book_line("h3 L alone, 0.4%", [h3b], BACK_Y),
    ]
    emit(md(pl.DataFrame(lines)))
    emit("")
    (H3 / "eval_tables.md").write_text("\n".join(out) + "\n")
    print("written", H3 / "eval_tables.md", file=sys.stderr)


if __name__ == "__main__":
    main()
