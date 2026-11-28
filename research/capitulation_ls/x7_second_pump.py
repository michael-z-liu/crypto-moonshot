# ruff: noqa: E501
"""X7a: pump-then-dump chains. A coin pumps (spot w-hour return >= thr) and pumps AGAIN 1-14 days
later. Short the second pump on the perp (isolated 1x, liquidation at +100%, optional funding gate
"last settled rate >= 0") or long it on spot. In-sample exploration; every config is listed.

Events: hourly spot closes (x3 matrices), trigger when close/close[w hours ago] - 1 >= thr, one
event per coin per 24 h (first trigger hour, then locked). A repeat ("second") pump is an event
with another event of the same coin 24-336 h before it. Entry at the next hour's open (perp for
shorts, spot for longs); spot 24h quote volume >= $1M; shorts need an active perp (>= $50k in each
of the last two closed hours). Costs 0.3% perp / 0.4% spot round trip, funding paid on shorts.

Writes results/explore/second_pump.md (+ x7_second_pump_grid.csv, calibration CSVs).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import polars as pl
from diag import calibration, scorecard, table, threshold_reach
from x7_lib import (
    PERP_COST,
    SPOT_COST,
    books,
    card,
    diag_frame,
    ep,
    fpaid,
    last_rate,
    perp,
    perp_ok,
    simulate,
    split_mean,
    spot,
    yr_cells,
)

OUT = Path(__file__).resolve().parent / "results" / "explore"
LOCK, GAP_MIN, GAP_MAX = 24, 24, 336
MIN_QV24 = 1_000_000.0


def ffill(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    idx = np.where(np.isfinite(out), np.arange(len(out))[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return np.take_along_axis(out, idx, axis=0)


S = spot()
P = perp()
SYMS = list(S["syms"])
PCOL = {s: j for j, s in enumerate(P["syms"])}
C = S["close"]
CF = ffill(C)
QV = np.nan_to_num(S["qv"])
_cq = np.cumsum(QV, axis=0, dtype=np.float64)
QV24 = _cq - np.vstack([np.zeros((24, QV.shape[1])), _cq[:-24]])
FIRST = np.where(np.isfinite(C).any(0), np.isfinite(C).argmax(0), C.shape[0])
NH = C.shape[0]


def events(w: int, thr: float) -> pl.DataFrame:
    """All pump events (w-hour spot return >= thr, 24h lock), with the prior events of the coin."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.full_like(CF, np.nan)
        r[w:] = CF[w:] / CF[:-w] - 1
    ok = np.isfinite(C) & (QV > 0) & (np.arange(NH)[:, None] >= FIRST[None, :] + w)
    trig = ok & (r >= thr)
    hh, cc = np.nonzero(trig)
    order = np.lexsort((hh, cc))
    hh, cc = hh[order], cc[order]
    rows: list[dict] = []
    last_c, last_h = -1, -(10**9)
    hist: list[tuple[int, float]] = []
    for h, col in zip(hh.tolist(), cc.tolist(), strict=True):
        if col != last_c:
            last_c, last_h, hist = col, -(10**9), []
        if h - last_h < LOCK:
            continue
        last_h = h
        prior = [(ph, pr) for ph, pr in hist if GAP_MIN <= h - ph <= GAP_MAX]
        rv = float(r[h, col])
        rows.append({
            "col": col, "symbol": SYMS[col], "h": h, "r": rv,
            "n_prior": len(prior), "gap_h": h - prior[-1][0] if prior else -1,
            "prior_max": max(pr for _, pr in prior) if prior else float("nan"),
            "qv24": float(QV24[h, col]),
        })  # fmt: skip
        hist.append((h, rv))
    d = pl.DataFrame(rows)
    return d.with_columns(year=pl.from_epoch(pl.col("h") * 3600 + ep(0)).dt.year()).filter(
        (pl.col("year") >= 2021) & (pl.col("qv24") >= MIN_QV24) & (pl.col("h") + 2 < NH)
    )


def trades(
    ev: pl.DataFrame, side: int, hold: int, tp: float | None, stop: float | None, gate: bool
) -> pl.DataFrame:
    out: list[dict] = []
    for e in ev.iter_rows(named=True):
        j, sym = e["h"], e["symbol"]
        t_in = ep(j + 1)
        if side < 0:
            pc = PCOL.get(sym, -1)
            if not perp_ok(pc, j):
                continue
            rate = last_rate(sym, t_in)
            if gate and rate == rate and rate < 0:
                continue
            arr = (P["open"][:, pc], P["high"][:, pc], P["low"][:, pc], P["close"][:, pc])
            qv24 = float(np.nansum(P["quote_volume"][j - 23 : j + 1, pc]))
            cost = PERP_COST
        else:
            col = e["col"]
            arr = (S["open"][:, col], S["high"][:, col], S["low"][:, col], S["close"][:, col])
            qv24 = e["qv24"]
            rate = float("nan")
            cost = SPOT_COST
        s = simulate(*arr, j + 1, hold, side, tp, stop)
        if s is None:
            continue
        t_out = ep(s["k_exit"] + 1)
        fund = fpaid(sym, t_in, t_out) if side < 0 else 0.0
        net = max(s["ret"] - cost - fund, -1.0)
        out.append({**e, "t_in": t_in, "t_out": t_out, "net": net, "ret": s["ret"], "fund": fund,
                    "kind": s["kind"], "mfe": s["mfe"], "mae": s["mae"], "rate": rate,
                    "qv24": qv24})  # fmt: skip
    return pl.DataFrame(out) if out else pl.DataFrame()


def dedup(t: pl.DataFrame) -> pl.DataFrame:
    """One position per coin while held (a later event of a held coin is skipped)."""
    if t.height == 0:
        return t
    t = t.sort(["symbol", "t_in"])
    keep, last_s, last_out = [], None, -1
    for i, (s, a, b) in enumerate(zip(t["symbol"], t["t_in"], t["t_out"], strict=True)):
        if s != last_s or a >= last_out:
            keep.append(i)
            last_s, last_out = s, b
    return t[keep].sort("t_in")


GRID_W = ((4, 0.4), (4, 0.6), (24, 0.4), (24, 0.6))
HOLDS = (24, 72, 168)
SHORT_EXITS = {
    "liq": (None, None),
    "tp30": (0.3, None),
    "sl50": (None, 0.5),
    "tp30+sl50": (0.3, 0.5),
}
LONG_EXITS = {
    "none": (None, None),
    "tp30": (0.3, None),
    "sl20": (None, 0.2),
    "tp30+sl20": (0.3, 0.2),
}


def stop_of(side: int, stop: float | None) -> float | None:
    return stop if stop is not None else (1.0 if side < 0 else None)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid: list[dict] = []
    cache_t: dict[tuple, pl.DataFrame] = {}
    ev_cache: dict[tuple[int, float], pl.DataFrame] = {}
    for w, thr in GRID_W:
        ev = events(w, thr)
        ev_cache[(w, thr)] = ev
        rep = ev.filter(pl.col("n_prior") >= 1)
        first = ev.filter(pl.col("n_prior") == 0)
        print(
            f"w={w} thr={thr}: events {ev.height}, repeat {rep.height}, first {first.height}",
            flush=True,
        )
        specs: list[tuple[str, int, str, bool, int]] = []
        for hold, ex in itertools.product(HOLDS, SHORT_EXITS):
            specs += [("repeat", -1, ex, False, hold), ("repeat", -1, ex, True, hold)]
        for hold, ex in itertools.product(HOLDS, LONG_EXITS):
            specs.append(("repeat", 1, ex, False, hold))
        # references: first pumps (no prior pump in 14 days), base exits only
        for hold in HOLDS:
            specs += [
                ("first", -1, "liq", False, hold),
                ("first", -1, "liq", True, hold),
                ("first", 1, "none", False, hold),
            ]
        for kind, side, ex, gate, hold in specs:
            tp, stop = (SHORT_EXITS if side < 0 else LONG_EXITS)[ex]
            t = dedup(trades(rep if kind == "repeat" else first, side, hold, tp, stop, gate))
            key = (w, thr, kind, side, ex, gate, hold)
            cache_t[key] = t
            if t.height == 0:
                continue
            c = card(t, tp, stop_of(side, stop))
            n1, e1 = split_mean(t, 2022, 2024)
            n2, e2 = split_mean(t, 2025, 2026)
            grid.append({
                "w": w, "thr": thr, "set": kind, "side": "short" if side < 0 else "long", "exit": ex,
                "gate": gate, "hold": hold, "n": t.height, "mean": float(t["net"].mean()),
                "E_o": c["E_o"], "win": c["win"], "stopped": c["share_stopped"],
                "contrib_stopped": c["contrib_stopped"], "giveback": c["giveback_med"],
                "n22_24": n1, "e22_24": e1, "n25_26": n2, "e25_26": e2, "years": yr_cells(t),
            })  # fmt: skip
    g = pl.DataFrame(grid)
    g.drop("years").write_csv(OUT / "x7_second_pump_grid.csv")
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_cols(30)
    print(g.drop("years"))

    # choice on 2022-24 (n >= 30 there), per side; check on 2025-26
    picks: dict[str, dict] = {}
    for side in ("short", "long"):
        sub = g.filter(
            (pl.col("set") == "repeat") & (pl.col("side") == side) & (pl.col("n22_24") >= 30)
        )
        best = sub.sort("e22_24", descending=True).row(0, named=True)
        picks[side] = best
    sub = g.filter(
        (pl.col("set") == "repeat")
        & (pl.col("side") == "short")
        & pl.col("gate")
        & (pl.col("n22_24") >= 30)
    )
    picks["short_gated"] = sub.sort("e22_24", descending=True).row(0, named=True)

    md: list[str] = []
    md += [
        "# Second pumps (pump-then-pump chains): short the exhaustion or long the momentum?",
        "",
        "Script: `x7_second_pump.py` (helpers `x7_lib.py`). Everything is in-sample exploration; the only",
        "holdout logic is *choose on 2022-24, read 2025-26*. Grid CSV: `x7_second_pump_grid.csv`.",
        "",
        "## Setup",
        "",
        "* Pump event: hourly spot close / close w hours earlier - 1 >= thr (x3 hourly matrices, 582 spot",
        "  symbols); first trigger hour per coin, then a 24 h lock. **Repeat pump** = an event with another",
        "  event of the same coin 1-14 days (24-336 h) before it (that includes third and later pumps of a",
        "  chain). Reference set **first** = events with no prior event in the last 14 days.",
        "* Entry at the open of the next hour (signal bar closed). Spot 24h volume >= $1M at the signal.",
        "* Short: perp, active (>= $50k traded with a moving price in each of the last two closed hours, >= 24 h",
        "  of perp history), isolated 1x (liquidation at +100% = stake lost), 0.3% round trip, funding paid from",
        "  the funding history. Gate (when on): last settled funding rate >= 0 (shorts not paid/crowded).",
        "* Long: spot, 0.4% round trip.",
        "* Exits on hourly bars, adverse side first; gaps through a stop fill at the open. Holds 24 h / 72 h / 7 d.",
        "* One position per coin while held (later events of a held coin skipped), no slot limit for the grid.",
        "* Books: `e25_combined.book`, 10 slots, each year from $10k, stake <= 0.5% of the instrument's 24h",
        "  volume, one position per coin. LMC = L + M + C sleeves (2022+); the new sleeve has priority 2.0",
        "  (below L, above C and M).",
        "",
        "## Grid (pre-declared, all listed)",
        "",
        f"(w, thr) in {list(GRID_W)}; holds {HOLDS}; short exits {list(SHORT_EXITS)} x gate off/on;",
        f"long exits {list(LONG_EXITS)} (tp 30%, stop 20%). Reference: first pumps, base exits.",
        f"{g.height} rows. `mean` = mean net per trade; `e22_24` / `e25_26` = mean of per-year means.",
        "",
        "| w | thr | set | side | exit | gate | hold | n | mean | win | stopped | contrib_stopped | giveback | e22_24 (n) | e25_26 (n) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |",
        "|" + "---|" * 21,
    ]
    for r in g.iter_rows(named=True):
        md.append(
            f"| {r['w']} | {r['thr']} | {r['set']} | {r['side']} | {r['exit']} | {'on' if r['gate'] else 'off'} | {r['hold']} | {r['n']} | "
            f"{r['mean'] * 100:+.1f}% | {r['win']:.2f} | {r['stopped']:.2f} | {r['contrib_stopped']:+.3f} | {r['giveback']:.2f} | "
            f"{r['e22_24'] * 100:+.1f}% ({r['n22_24']}) | {r['e25_26'] * 100:+.1f}% ({r['n25_26']}) | {r['years']} |"
        )

    # picks: scorecards, calibration, books
    cards: dict[str, dict[str, float]] = {}
    book_lines: list[str] = []
    cal_notes: list[str] = []
    pool_ev = {w: events(w, 0.2) for w in sorted({w for w, _ in GRID_W})}
    for name, p in picks.items():
        side = -1 if p["side"] == "short" else 1
        tp, stop = (SHORT_EXITS if side < 0 else LONG_EXITS)[p["exit"]]
        key = (p["w"], p["thr"], "repeat", side, p["exit"], p["gate"], p["hold"])
        t = cache_t[key]
        label = f"{name}: w{p['w']} thr{p['thr']} {p['exit']} gate={'on' if p['gate'] else 'off'} hold {p['hold']}h"
        # calibration pool: repeat pumps at thr 0.2 (both pumps >= 20%), signal = percentile of
        # min(this pump, largest prior pump); threshold = percentile of p['thr']
        pool = (
            pool_ev[p["w"]]
            .filter(pl.col("n_prior") >= 1)
            .with_columns(sz=pl.min_horizontal("r", "prior_max"))
        )
        pool = pool.with_columns(signal=pl.col("sz").rank() / pl.len())
        theta = float(pool.filter(pl.col("sz") >= p["thr"])["signal"].min())
        pt = dedup(trades(pool, side, p["hold"], tp, stop, p["gate"]))
        fr = diag_frame(pt, theta, tp, stop_of(side, stop))
        ent, non = fr.filter(pl.col("signal") >= theta), fr.filter(pl.col("signal") < theta)
        cards[label] = scorecard(
            diag_frame(t.with_columns(signal=pl.lit(1.0)), 0.0, tp, stop_of(side, stop))
        )
        cal = calibration(ent, non)
        cal.write_csv(OUT / f"x7_calibration_second_pump_{name}.csv")
        tr = threshold_reach(ent, non)
        cal_notes += [f"### {label}", "", f"Pool: repeat pumps with both pumps >= 20% (w={p['w']}), n={fr.height}; "
                      f"threshold {theta:.3f} (= both >= {p['thr']:.0%}); Spearman(signal, outcome) {tr['spearman_signal_outcome']:+.3f}.", "",
                      "| bin (signal >=) | n | E_o | win | p_tp | p_stopped | tp_progress | entered |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in cal.iter_rows(named=True):
            cal_notes.append(
                f"| {r['bin']:.3f} | {r['n']} | {r['E_o']:+.3f} | {r['win']:.2f} | {r['p_tp']:.2f} | {r['p_stopped']:.2f} | {r['tp_progress']:.2f} | {r['entered']:.2f} |"
            )
        cal_notes.append("")
        b = books(t)
        book_lines += [f"* **{label}** (2022-24 {p['e22_24'] * 100:+.1f}%/trade, 2025-26 {p['e25_26'] * 100:+.1f}%/trade)",
                       f"  * alone: {b['alone']}", f"  * LMC + X: {b['lmc_plus']}"]  # fmt: skip
        lmc_line = b["lmc"]
        # breakdowns of the pick by gap and by chain length
        bd = t.with_columns(
            gap=pl.when(pl.col("gap_h") <= 72).then(pl.lit("1-3d")).otherwise(pl.lit("3-14d")),
            chain=pl.when(pl.col("n_prior") >= 2).then(pl.lit("3rd+")).otherwise(pl.lit("2nd")),
        )
        for col in ("gap", "chain"):
            agg = (
                bd.group_by(col)
                .agg(
                    n=pl.len(),
                    mean=pl.col("net").mean(),
                    stopped=pl.col("kind").is_in(["stop", "liq"]).mean(),
                )
                .sort(col)
            )
            cal_notes.append(
                f"By {col}: "
                + "; ".join(
                    f"{r[col]} n={r['n']} mean {r['mean'] * 100:+.1f}% stopped {r['stopped']:.0%}"
                    for r in agg.iter_rows(named=True)
                )
            )
        cal_notes.append("")
    md += ["", "## Picks (best e22_24 per side with n22_24 >= 30; plus the best gated short)", ""]
    md += [
        table(cards),
        "",
        "## Books (10 slots, each year from $10k)",
        "",
        f"* LMC baseline: {lmc_line}",
        *book_lines,
        "",
    ]
    md += [
        "## Calibration (outcome by signal decile, pool simulated with the pick's exits)",
        "",
        *cal_notes,
    ]
    (OUT / "second_pump_raw.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-80:]))


if __name__ == "__main__":
    main()
