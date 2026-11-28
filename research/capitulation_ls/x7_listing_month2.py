# ruff: noqa: E501
"""X7b: listing momentum after the first month. New Binance spot listings (e12, 2021+), entered
30 / 60 / 90 days after listing, long (spot) or short (perp, isolated 1x) by their state at entry:
return since the first day (close at listing + 24 h), 30-day trend, or distance to the high since
listing. In-sample exploration; every config is listed.

Entry at the open of the hour listing + D days (signal bar = the hour before). Spot 24h volume
>= $500k; shorts need an active perp (>= $50k traded in each of the last two closed hours).
Costs 0.3% perp / 0.4% spot round trip, funding paid on shorts; exits on hourly bars, adverse first.

Writes results/explore/listing_month2_raw.md (+ x7_listing_month2_grid.csv, calibration CSVs).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import polars as pl
from common_ls import OWN
from diag import calibration, scorecard, table, threshold_reach
from x7_lib import (
    PERP_COST,
    SPOT_COST,
    T0,
    books,
    card,
    diag_frame,
    ep,
    fpaid,
    perp,
    perp_ok,
    simulate,
    split_mean,
    spot,
    yr_cells,
)

OUT = Path(__file__).resolve().parent / "results" / "explore"
MIN_QV24 = 500_000.0
S = spot()
P = perp()
SCOL = {s: j for j, s in enumerate(S["syms"])}
PCOL = {s: j for j, s in enumerate(P["syms"])}
NH = S["close"].shape[0]


def ffill_col(x: np.ndarray) -> np.ndarray:
    idx = np.where(np.isfinite(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


def candidates(day: int) -> pl.DataFrame:
    """One row per listing: features at the signal hour (listing + day days - 1 h)."""
    L = pl.read_parquet(OWN / "e12_listings.parquet")
    rows: list[dict] = []
    for r in L.iter_rows(named=True):
        col = SCOL.get(r["symbol"])
        if col is None:
            continue
        h0 = (int(r["listed"].timestamp()) - T0) // 3600
        j = h0 + day * 24 - 1
        if j + 2 >= NH or h0 < 0:
            continue
        c = ffill_col(S["close"][h0 : j + 1, col])
        if not np.isfinite(c[-1]) or len(c) < 24 * 30:
            continue
        d1 = c[23]
        qv24 = float(np.nansum(S["qv"][j - 23 : j + 1, col]))
        rows.append({
            "symbol": r["symbol"], "col": col, "h": j, "listed": r["listed"],
            "since_d1": float(c[-1] / d1 - 1), "r30": float(c[-1] / c[max(len(c) - 721, 0)] - 1),
            "off_high": float(c[-1] / np.nanmax(S["high"][h0 + 24 : j + 1, col]) - 1),
            "qv24": qv24,
        })  # fmt: skip
    d = pl.DataFrame(rows)
    return d.with_columns(year=pl.from_epoch(pl.col("h") * 3600 + T0 + 3600).dt.year()).filter(
        pl.col("qv24") >= MIN_QV24
    )


def trades(
    cand: pl.DataFrame, side: int, hold: int, tp: float | None, stop: float | None
) -> pl.DataFrame:
    out: list[dict] = []
    for e in cand.iter_rows(named=True):
        j, sym = e["h"], e["symbol"]
        t_in = ep(j + 1)
        if side < 0:
            pc = PCOL.get(sym, -1)
            if not perp_ok(pc, j):
                continue
            arr = (P["open"][:, pc], P["high"][:, pc], P["low"][:, pc], P["close"][:, pc])
            qv24 = float(np.nansum(P["quote_volume"][j - 23 : j + 1, pc]))
            cost = PERP_COST
        else:
            col = e["col"]
            arr = (S["open"][:, col], S["high"][:, col], S["low"][:, col], S["close"][:, col])
            qv24, cost = e["qv24"], SPOT_COST
        s = simulate(*arr, j + 1, hold, side, tp, stop)
        if s is None:
            continue
        t_out = ep(s["k_exit"] + 1)
        fund = fpaid(sym, t_in, t_out) if side < 0 else 0.0
        out.append({**e, "t_in": t_in, "t_out": t_out, "net": max(s["ret"] - cost - fund, -1.0),
                    "fund": fund, "kind": s["kind"], "mfe": s["mfe"], "mae": s["mae"], "qv24": qv24})  # fmt: skip
    return pl.DataFrame(out) if out else pl.DataFrame()


DAYS = (30, 60, 90)
HOLDS = (72, 168, 720)
SHORT_EXITS = {"liq": (None, None), "tp50+sl50": (0.5, 0.5)}
LONG_EXITS = {"none": (None, None), "tp50+sl30": (0.5, 0.3)}
# (feature, op, level); op "le" for weak (short), "ge" for strong (long); None = all listings
SHORT_F: list[tuple[str, str, float] | None] = [
    None,
    ("since_d1", "le", 0.0),
    ("since_d1", "le", -0.5),
    ("r30", "le", 0.0),
    ("r30", "le", -0.3),
    ("off_high", "le", -0.5),
    ("off_high", "le", -0.75),
]
LONG_F: list[tuple[str, str, float] | None] = [
    None,
    ("since_d1", "ge", 0.0),
    ("since_d1", "ge", 0.5),
    ("r30", "ge", 0.0),
    ("r30", "ge", 0.3),
    ("off_high", "ge", -0.2),
    ("off_high", "ge", -0.35),
]


def fname(f: tuple[str, str, float] | None) -> str:
    return "all" if f is None else f"{f[0]} {'<=' if f[1] == 'le' else '>='} {f[2]:+g}"


def apply(c: pl.DataFrame, f: tuple[str, str, float] | None) -> pl.DataFrame:
    if f is None:
        return c
    return c.filter(pl.col(f[0]) <= f[2] if f[1] == "le" else pl.col(f[0]) >= f[2])


def stop_of(side: int, stop: float | None) -> float | None:
    return stop if stop is not None else (1.0 if side < 0 else None)


def main() -> None:
    grid: list[dict] = []
    tcache: dict[tuple, pl.DataFrame] = {}
    cands = {d: candidates(d) for d in DAYS}
    for d in DAYS:
        print(
            f"day {d}: {cands[d].height} listings",
            cands[d].group_by("year").len().sort("year").rows(),
            flush=True,
        )
    for d, side, hold in itertools.product(DAYS, (-1, 1), HOLDS):
        exits = SHORT_EXITS if side < 0 else LONG_EXITS
        for ex, (tp, stop) in exits.items():
            full = trades(cands[d], side, hold, tp, stop)
            for f in SHORT_F if side < 0 else LONG_F:
                t = apply(full, f)
                if t.height == 0:
                    continue
                key = (d, side, hold, ex, fname(f))
                tcache[key] = t
                c = card(t, tp, stop_of(side, stop))
                n1, e1 = split_mean(t, 2022, 2024)
                n2, e2 = split_mean(t, 2025, 2026)
                grid.append({
                    "day": d, "side": "short" if side < 0 else "long", "filter": fname(f), "hold": hold,
                    "exit": ex, "n": t.height, "mean": float(t["net"].mean()), "win": c["win"],
                    "stopped": c["share_stopped"], "contrib_stopped": c["contrib_stopped"],
                    "giveback": c["giveback_med"], "n22_24": n1, "e22_24": e1, "n25_26": n2,
                    "e25_26": e2, "years": yr_cells(t),
                })  # fmt: skip
        print("done", d, side, hold, flush=True)
    g = pl.DataFrame(grid)
    g.drop("years").write_csv(OUT / "x7_listing_month2_grid.csv")

    picks: dict[str, dict] = {}
    for side in ("short", "long"):
        sub = g.filter((pl.col("side") == side) & (pl.col("n22_24") >= 20))
        picks[side] = sub.sort("e22_24", descending=True).row(0, named=True)

    md = [
        "# Listing momentum after the first month (days 30-90)",
        "",
        "Script: `x7_listing_month2.py` (helpers `x7_lib.py`). In-sample exploration; choose on 2022-24,",
        "read 2025-26. Grid CSV: `x7_listing_month2_grid.csv`.",
        "",
        "## Setup",
        "",
        "* Listings: `e12_listings.parquet` (397 spot listings 2021-01 .. 2026; first 5m candle = listing).",
        "  Hourly spot/perp prices from the x3 matrices; listings not in the 582-symbol matrix are skipped.",
        "* Entry at the open of the hour listing + D days (D = 30 / 60 / 90); features use closes up to the",
        "  hour before: `since_d1` = close / close at listing + 24 h - 1 (distance to the first-day price),",
        "  `r30` = 30-day return, `off_high` = close / highest high since listing + 24 h - 1.",
        "* Spot 24h volume >= $500k. Long = spot, 0.4% round trip. Short = perp, active (>= $50k in each of the",
        "  last two closed hours, >= 24 h of history), isolated 1x (+100% = stake lost), 0.3% + funding.",
        "* Exits on hourly bars, adverse first. Books: `e25_combined.book`, 10 slots, $10k each year,",
        "  stake <= 0.5% of 24h volume; LMC = L + M + C (2022+), new sleeve priority 2.0.",
        "",
        "## Grid (pre-declared, all listed)",
        "",
        f"D {DAYS} x side x filter (short: {', '.join(fname(f) for f in SHORT_F)}; long: {', '.join(fname(f) for f in LONG_F)})",
        f"x hold {HOLDS} h x exit (short {list(SHORT_EXITS)}, long {list(LONG_EXITS)}) = {g.height} rows.",
        "`e22_24` / `e25_26` = mean of per-year mean net per trade.",
        "",
        "| D | side | filter | hold | exit | n | mean | win | stopped | contrib_stopped | giveback | e22_24 (n) | e25_26 (n) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |",
        "|" + "---|" * 19,
    ]
    for r in g.iter_rows(named=True):
        md.append(
            f"| {r['day']} | {r['side']} | {r['filter']} | {r['hold']} | {r['exit']} | {r['n']} | {r['mean'] * 100:+.1f}% | {r['win']:.2f} | "
            f"{r['stopped']:.2f} | {r['contrib_stopped']:+.3f} | {r['giveback']:.2f} | {r['e22_24'] * 100:+.1f}% ({r['n22_24']}) | "
            f"{r['e25_26'] * 100:+.1f}% ({r['n25_26']}) | {r['years']} |"
        )
    cards: dict[str, dict[str, float]] = {}
    book_lines: list[str] = []
    cal_notes: list[str] = []
    lmc_line = ""
    for name, p in picks.items():
        side = -1 if p["side"] == "short" else 1
        tp, stop = (SHORT_EXITS if side < 0 else LONG_EXITS)[p["exit"]]
        t = tcache[(p["day"], side, p["hold"], p["exit"], p["filter"])]
        label = f"{name}: D{p['day']} {p['filter']} hold {p['hold']}h {p['exit']}"
        cards[label] = scorecard(
            diag_frame(t.with_columns(signal=pl.lit(1.0)), 0.0, tp, stop_of(side, stop))
        )
        pool = trades(cands[p["day"]], side, p["hold"], tp, stop)
        f = next(x for x in (SHORT_F if side < 0 else LONG_F) if fname(x) == p["filter"])
        feat, op, lvl = (
            f
            if f is not None
            else (("since_d1", "le", 1e9) if side < 0 else ("since_d1", "ge", -1e9))
        )
        sgn = -1.0 if op == "le" else 1.0
        pool = pool.with_columns(signal=(pl.col(feat) * sgn).rank() / pl.len())
        ent = pool.filter(pl.col(feat) <= lvl if op == "le" else pl.col(feat) >= lvl)
        theta = float(ent["signal"].min()) if ent.height else 1.0
        fr = diag_frame(pool, theta, tp, stop_of(side, stop))
        e_, n_ = fr.filter(pl.col("signal") >= theta), fr.filter(pl.col("signal") < theta)
        cal = calibration(e_, n_)
        cal.write_csv(OUT / f"x7_calibration_listing_month2_{name}.csv")
        tr = threshold_reach(e_, n_)
        cal_notes += [f"### {label}", "", f"Pool: all listings at D{p['day']} (n={fr.height}); signal = rank of {'-' if sgn < 0 else ''}{feat}; "
                      f"threshold {theta:.3f}; Spearman(signal, outcome) {tr['spearman_signal_outcome']:+.3f}.", "",
                      "| bin (signal >=) | n | E_o | win | p_tp | p_stopped | tp_progress | entered |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in cal.iter_rows(named=True):
            cal_notes.append(
                f"| {r['bin']:.3f} | {r['n']} | {r['E_o']:+.3f} | {r['win']:.2f} | {r['p_tp']:.2f} | {r['p_stopped']:.2f} | {r['tp_progress']:.2f} | {r['entered']:.2f} |"
            )
        cal_notes.append("")
        b = books(t)
        lmc_line = b["lmc"]
        book_lines += [f"* **{label}** (2022-24 {p['e22_24'] * 100:+.1f}%/trade, 2025-26 {p['e25_26'] * 100:+.1f}%/trade)",
                       f"  * alone: {b['alone']}", f"  * LMC + X: {b['lmc_plus']}"]  # fmt: skip
    # post-hoc (seen after the grid): the best cells that are positive in both periods, and their
    # neighbours across the entry day D, get books too
    post = [(30, -1, 720, "liq", "all"), (60, -1, 720, "liq", "all"), (90, -1, 720, "liq", "all"),
            (60, -1, 720, "liq", "r30 <= -0.3"), (30, 1, 168, "none", "r30 >= +0.3"), (90, 1, 168, "none", "r30 >= +0.3")]  # fmt: skip
    for key in post:
        t = tcache[key]
        b = books(t)
        n1, e1 = split_mean(t, 2022, 2024)
        n2, e2 = split_mean(t, 2025, 2026)
        book_lines += [f"* post-hoc D{key[0]} {'short' if key[1] < 0 else 'long'} {key[4]} hold {key[2]}h {key[3]} (2022-24 {e1 * 100:+.1f}%/trade n={n1}, 2025-26 {e2 * 100:+.1f}%/trade n={n2})",
                       f"  * alone: {b['alone']}", f"  * LMC + X: {b['lmc_plus']}"]  # fmt: skip
    # robustness of the post-hoc D60 short: sweep the entry day (all listings, 30 d, liquidation only)
    sweep = ["", "## Entry-day sweep (post-hoc robustness): short all listings, hold 720 h, liq only", "",
             "| D | n | mean | stopped | e22_24 (n) | e25_26 (n) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |", "|" + "---|" * 12]  # fmt: skip
    for dd in (35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85):
        t = trades(candidates(dd), -1, 720, None, None)
        n1, e1 = split_mean(t, 2022, 2024)
        n2, e2 = split_mean(t, 2025, 2026)
        st = float(t["kind"].is_in(["stop", "liq"]).mean())
        sweep.append(
            f"| {dd} | {t.height} | {t['net'].mean() * 100:+.1f}% | {st:.2f} | {e1 * 100:+.1f}% ({n1}) | {e2 * 100:+.1f}% ({n2}) | {yr_cells(t)} |"
        )
        print(sweep[-1], flush=True)
    md += ["", "## Picks (best e22_24 per side, n22_24 >= 20)", "", table(cards), "", "## Books (10 slots, each year from $10k)", "",
           f"* LMC baseline: {lmc_line}", *book_lines, *sweep, "", "## Calibration", "", *cal_notes]  # fmt: skip
    (OUT / "listing_month2_raw.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-60:]))


if __name__ == "__main__":
    main()
