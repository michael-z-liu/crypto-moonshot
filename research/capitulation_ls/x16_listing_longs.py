# ruff: noqa: E501
"""X16: LONG trades around new Binance SPOT listings (the opposite side of the M short sleeve).
In-sample exploration on the 397 spot listings of `e12_listings.parquet` (2021-01 .. 2026), dense
5m spot paths from `x16_paths.py`. Every configuration tried is listed in the grid outputs.

Common rules (fixed before looking at any outcome)
* Reference "listing price" = close of the first 5m spot candle (the first candle's open is a
  placeholder on Binance listings). Long spot, 0.4% round trip, net floored at -100%.
* Entry at the OPEN of the 5m bar after the signal bar; exits on 5m bars, adverse side first,
  gaps through a level fill at the bar open (`x7_lib.simulate`). Flat-filled bars where no candle
  printed; the path ends at the last candle (delisting).
* Stake cap in the books: 0.5% of the trailing 24h spot quote volume up to the signal bar
  (unscaled, so a 5m entry is capped by the first candle's volume alone).
* Year = entry year. Choose on 2021-24 (mean of per-year mean net, >= 20 trades and trades in
  >= 3 of the 4 years), read on 2025-26. No tokenised stocks / ETFs appear in the list.

Pre-declared grid
A. Listing-day longs. Entry after the first 5m candle (bar 1), the first 1 h (bar 12), 4 h
   (bar 48) or 24 h (bar 288) x hold {4 h, 24 h, 72 h, 7 d} x TP {20%, 50%, 100%, none} x stop
   {none, 30%} = 128 configs, each on 8 subsets: all; first-hour return r1h = close(bar 11) /
   reference - 1 in (<= -15%, -15..0, 0..+15%, > +15%); first-day quote volume qv_1d (bars
   0..287) < $100M, $100M-$400M, > $400M. A subset is "eligible" for the pick only if its split
   is known at entry (r1h: entries >= 1 h; qv_1d: the 24 h entry); the rest are ex-post
   diagnostics. 1024 rows.
B. Strong-listing longs. Check at 24 h / 48 h / 72 h (signal bar 287 / 575 / 863, entry at the
   next open); listings whose close is above the reference, by how far: all > 0, 0..20%,
   20..50%, > 50% x hold {24 h, 72 h, 7 d, 30 d} x TP {50%, 100%, none} x stop {30%, 50%, none}
   = 432 rows, all eligible.
C. Pair. M (e25 `sleeve_listing_weak`: perp short at listing + 72 h on listings at or below
   their perp listing price, 30 d, TP/stop 50%) plus a spot long on its complement (e27 rows
   with since_list > 0, i.e. the 58 "strong" listings at M's own clock and price), entered on
   the 5m grid at M's entry time, with (i) the mirror exit 30 d, TP 50%, stop 50% and (ii) the
   B pick's hold / TP / stop. Books: M alone, M + long, LMC+ + long.
Books: `e25_combined.book`, 10 slots, $10k each year, per-trade w = 1, one position per coin;
the new sleeve at the lowest priority (0.5). Alone 2021+; LMC+ (= L + M + C + L on perp-only
coins, `p2_lib`) is 2022+, so the addition enters it from 2022.
Robustness of the picks only: drop a declared list of renames / migrations that look like new
listings in e12 (BTTC LUNC USTC RENDER POL S A MANTRA KAIA T XEC BEAMX).

Writes results/explore/listing_longs_raw.md, x16_grid_a.csv, x16_grid_b.csv,
x16_calibration_{a,b}.csv.
"""

from __future__ import annotations

import itertools
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
from common_ls import OWN
from diag import calibration, table, threshold_reach
from x7_lib import SPOT_COST, card, diag_frame, simulate

OUT = Path(__file__).resolve().parent / "results" / "explore"
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
BARS_H = 12
RENAMES = {"BTTC", "LUNC", "USTC", "RENDER", "POL", "S", "A", "MANTRA", "KAIA", "T", "XEC", "BEAMX"}

A_ENTRY = {"5m": 1, "1h": 12, "4h": 48, "24h": 288}
A_HOLD = {"4h": 48, "24h": 288, "72h": 864, "7d": 2016}
A_TP: tuple[float | None, ...] = (0.2, 0.5, 1.0, None)
A_STOP: tuple[float | None, ...] = (None, 0.3)
R1H_BINS = (
    ("r1h<=-15%", -9.0, -0.15),
    ("r1h -15..0%", -0.15, 0.0),
    ("r1h 0..15%", 0.0, 0.15),
    ("r1h>15%", 0.15, 9e9),
)
QV_BINS = (("qv1d<100M", 0.0, 1e8), ("qv1d 100-400M", 1e8, 4e8), ("qv1d>400M", 4e8, 1e15))

B_CHECK = {"24h": 288, "48h": 576, "72h": 864}
B_BINS = (
    ("above>0", 0.0, 9e9),
    ("above 0-20%", 0.0, 0.2),
    ("above 20-50%", 0.2, 0.5),
    ("above>50%", 0.5, 9e9),
)
B_HOLD = {"24h": 288, "72h": 864, "7d": 2016, "30d": 8640}
B_TP: tuple[float | None, ...] = (0.5, 1.0, None)
B_STOP: tuple[float | None, ...] = (0.3, 0.5, None)


@cache
def paths() -> dict[str, np.ndarray]:
    z = np.load(OWN / "explore" / "x16_paths.npz")
    return {k: z[k] for k in z.files}


@cache
def listings() -> pl.DataFrame:
    """One row per listing: reference price, first-hour return, first-day volume, year."""
    P = paths()
    c, qv = P["close"], P["qv"]
    ref = c[:, 0]
    return pl.DataFrame({
        "i": np.arange(len(ref)), "symbol": P["symbols"], "t0": P["listed"],
        "year": pl.from_epoch(pl.Series(P["listed"]), "s").dt.year(),
        "ref": ref, "r1h": c[:, 11] / ref - 1, "qv_1d": np.nansum(qv[:, :288], axis=1),
    })  # fmt: skip


def fmt_tp(x: float | None) -> str:
    return "none" if x is None else f"{x * 100:.0f}%"


def trades(e_of: dict[int, int], hold: int, tp: float | None, stop: float | None) -> pl.DataFrame:
    """Long every listing i at bar e_of[i] (entry at that bar's open) for `hold` bars."""
    P = paths()
    L = listings()
    out: list[dict[str, object]] = []
    for i, e in e_of.items():
        s = simulate(P["open"][i], P["high"][i], P["low"][i], P["close"][i], e, hold, 1, tp, stop)
        if s is None:
            continue
        t0 = int(P["listed"][i])
        r = L.row(i, named=True)
        out.append({
            "i": i, "symbol": r["symbol"], "year": r["year"], "t_in": t0 + e * 300,
            "t_out": t0 + (int(s["k_exit"]) + 1) * 300, "net": max(s["ret"] - SPOT_COST, -1.0),
            "kind": s["kind"], "mfe": s["mfe"], "mae": s["mae"],
            "qv24": float(np.nansum(P["qv"][i, max(0, e - 288) : e])),
            "r1h": r["r1h"], "qv_1d": r["qv_1d"], "since": float(P["close"][i, e - 1] / r["ref"] - 1),
        })  # fmt: skip
    return pl.DataFrame(out)


def per_year(t: pl.DataFrame) -> dict[int, tuple[int, float]]:
    g = t.group_by("year").agg(n=pl.len(), e=pl.col("net").mean())
    return {int(r["year"]): (int(r["n"]), float(r["e"])) for r in g.iter_rows(named=True)}


def summary(t: pl.DataFrame) -> dict[str, object]:
    py = per_year(t)
    a = [py[y] for y in (2021, 2022, 2023, 2024) if y in py]
    b = [py[y] for y in (2025, 2026) if y in py]
    st = float(t["kind"].is_in(["stop", "liq"]).mean()) if t.height else float("nan")
    return {
        "n": t.height, "mean": float(t["net"].mean()) if t.height else float("nan"),
        "median": float(t["net"].median()) if t.height else float("nan"),
        "win": float((t["net"] > 0).mean()) if t.height else float("nan"), "stopped": st,
        "n21_24": sum(n for n, _ in a), "yrs21_24": len(a),
        "e21_24": float(np.mean([e for _, e in a])) if a else float("nan"),
        "n25_26": sum(n for n, _ in b), "e25_26": float(np.mean([e for _, e in b])) if b else float("nan"),
        **{f"y{y}": (py[y][1] if y in py else float("nan")) for y in YEARS},
        **{f"n{y}": (py[y][0] if y in py else 0) for y in YEARS},
    }  # fmt: skip


def subset(t: pl.DataFrame, col: str, lo: float, hi: float) -> pl.DataFrame:
    return t.filter((pl.col(col) > lo) & (pl.col(col) <= hi))


def grid_a() -> tuple[pl.DataFrame, dict[tuple[object, ...], pl.DataFrame]]:
    L = listings()
    rows: list[dict[str, object]] = []
    cache_t: dict[tuple[object, ...], pl.DataFrame] = {}
    for (en, e), (hn, h), tp, sl in itertools.product(
        A_ENTRY.items(), A_HOLD.items(), A_TP, A_STOP
    ):
        full = trades({int(i): e for i in L["i"]}, h, tp, sl)
        subs: list[tuple[str, pl.DataFrame, bool]] = [("all", full, True)]
        subs += [(nm, subset(full, "r1h", lo, hi), e >= 12) for nm, lo, hi in R1H_BINS]
        subs += [(nm, subset(full, "qv_1d", lo, hi), e >= 288) for nm, lo, hi in QV_BINS]
        for nm, t, elig in subs:
            key = ("A", en, hn, tp, sl, nm)
            cache_t[key] = t
            rows.append(
                {
                    "entry": en,
                    "hold": hn,
                    "tp": fmt_tp(tp),
                    "stop": fmt_tp(sl),
                    "subset": nm,
                    "eligible": elig,
                    **summary(t),
                }
            )
        print(
            "A", en, hn, fmt_tp(tp), fmt_tp(sl), f"all {full['net'].mean() * 100:+.1f}%", flush=True
        )
    return pl.DataFrame(rows), cache_t


def grid_b() -> tuple[pl.DataFrame, dict[tuple[object, ...], pl.DataFrame]]:
    L = listings()
    rows: list[dict[str, object]] = []
    cache_t: dict[tuple[object, ...], pl.DataFrame] = {}
    for (cn, e), (hn, h), tp, sl in itertools.product(
        B_CHECK.items(), B_HOLD.items(), B_TP, B_STOP
    ):
        full = trades({int(i): e for i in L["i"]}, h, tp, sl)  # every listing: the calibration pool
        cache_t[("Bpool", cn, hn, tp, sl)] = full
        for nm, lo, hi in B_BINS:
            t = subset(full, "since", lo, hi)
            cache_t[("B", cn, hn, tp, sl, nm)] = t
            rows.append(
                {
                    "check": cn,
                    "bin": nm,
                    "hold": hn,
                    "tp": fmt_tp(tp),
                    "stop": fmt_tp(sl),
                    **summary(t),
                }
            )
        print(
            "B",
            cn,
            hn,
            fmt_tp(tp),
            fmt_tp(sl),
            f">0 {subset(full, 'since', 0, 9e9)['net'].mean() * 100:+.1f}%",
            flush=True,
        )
    return pl.DataFrame(rows), cache_t


def pick(g: pl.DataFrame) -> dict[str, object]:
    sub = g.filter((pl.col("n21_24") >= 20) & (pl.col("yrs21_24") >= 3))
    if "eligible" in g.columns:
        sub = sub.filter(pl.col("eligible"))
    return sub.sort("e21_24", descending=True, nulls_last=True).row(0, named=True)


def pct(x: object) -> str:
    v = float(x)  # type: ignore[arg-type]
    return "-" if not np.isfinite(v) else f"{v * 100:+.1f}%"


def yr_cells(r: dict[str, object]) -> str:
    return " | ".join(f"{pct(r[f'y{y}'])} ({r[f'n{y}']})" for y in YEARS)


def as_book(t: pl.DataFrame, sleeve: str, prio: float = 0.5) -> pl.DataFrame:
    return t.select(
        "symbol", adv=pl.col("mae").clip(0, 1), t_in=pl.col("t_in").cast(pl.Int64), t_out=pl.col("t_out").cast(pl.Int64),
        net="net", qv="qv24", sleeve=pl.lit(sleeve), prio=pl.lit(prio), w=pl.lit(1.0),
    )  # fmt: skip


def book_line(name: str, frames: list[pl.DataFrame]) -> str:
    from e25_combined import book

    fr = [
        f.select(
            "symbol",
            "adv",
            "t_in",
            "t_out",
            "net",
            "qv",
            "sleeve",
            pl.col("prio").cast(pl.Float64),
            "w",
        )
        for f in frames
    ]
    b = book(pl.concat(fr, how="vertical_relaxed"), 10)
    if b.height == 0:
        return f"| {name} | 0 | - | - | - |" + " - |" * len(YEARS)
    n = int(b["trades"].sum())
    bp = float((b["bp"] * b["trades"]).sum() / n)
    yrs = {int(r["year"]): r for r in b.iter_rows(named=True)}
    cells = " | ".join(
        f"{yrs[y]['ret'] * 100:+.0f}% ({yrs[y]['trades']}t, dd {yrs[y]['dd'] * 100:.0f})"
        if y in yrs
        else "-"
        for y in YEARS
    )
    in21 = [float(yrs[y]["ret"]) for y in (2021, 2022, 2023, 2024) if y in yrs]
    return (
        f"| {name} | {n} | {bp:+.0f} | {min(float(r['ret']) for r in yrs.values()) * 100:+.0f}% | "
        f"{float(b['dd'].min()) * 100:+.0f}% | {min(in21) * 100 if in21 else float('nan'):+.0f}% | {cells} |"
    )


BOOK_HEAD = [
    "| book | trades | bp/trade | worst year | worst DD | worst 2021-24 | "
    + " | ".join(str(y) for y in YEARS)
    + " |",
    "|" + "---|" * (6 + len(YEARS)),
]


def cal_block(
    label: str,
    pool: pl.DataFrame,
    entered: pl.Series,
    sig_col: str,
    tp: float | None,
    stop: float | None,
) -> tuple[list[str], pl.DataFrame]:
    pool = pool.with_columns(signal=pl.col(sig_col).rank() / pl.len(), ent=entered)
    theta = float(pool.filter(pl.col("ent"))["signal"].min())
    fr = diag_frame(pool, theta, tp, stop).with_columns(ent=pool["ent"])
    e_, n_ = fr.filter(pl.col("ent")).drop("ent"), fr.filter(~pl.col("ent")).drop("ent")
    cal = calibration(e_, n_ if n_.height else None)
    tr = threshold_reach(e_, n_ if n_.height else None)
    lines = [f"### {label}", "", f"Pool n={pool.height}; signal = rank of {sig_col}; entered share {float(pool['ent'].mean()):.2f}; "
             f"Spearman(signal, outcome) {tr['spearman_signal_outcome']:+.3f}.", "",
             "| bin (signal >=) | n | E_o | win | p_tp | p_stopped | tp_progress | entered |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in cal.iter_rows(named=True):
        lines.append(
            f"| {r['bin']:.3f} | {r['n']} | {r['E_o']:+.3f} | {r['win']:.2f} | {r['p_tp']:.2f} | {r['p_stopped']:.2f} | {r['tp_progress']:.2f} | {r['entered']:.2f} |"
        )
    return [*lines, ""], cal


def grid_md(g: pl.DataFrame, keys: list[str]) -> list[str]:
    head = [
        *keys,
        "n",
        "mean",
        "median",
        "win",
        "stopped",
        "e21_24 (n)",
        "e25_26 (n)",
        *[str(y) for y in YEARS],
    ]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in g.iter_rows(named=True):
        out.append("| " + " | ".join(str(r[k]) for k in keys) + f" | {r['n']} | {pct(r['mean'])} | {pct(r['median'])} | {float(r['win']):.2f} | "
                   f"{float(r['stopped']):.2f} | {pct(r['e21_24'])} ({r['n21_24']}) | {pct(r['e25_26'])} ({r['n25_26']}) | {yr_cells(r)} |")  # fmt: skip
    return out


def selection_check(g: pl.DataFrame, name: str) -> list[str]:
    """How the in-sample winners read out of period (selection bias across the grid)."""
    sub = g.filter((pl.col("n21_24") >= 20) & (pl.col("yrs21_24") >= 3))
    if "eligible" in g.columns:
        sub = sub.filter(pl.col("eligible"))
    sub = sub.sort("e21_24", descending=True, nulls_last=True)
    k = max(1, sub.height // 10)
    top = sub.head(k)
    ge10 = sub.filter(pl.col("e21_24") >= 0.10)
    return [
        f"* {name}: {sub.height} pickable rows; e21_24 >= +10%: {ge10.height} rows (their mean e25_26 {pct(ge10['e25_26'].mean() if ge10.height else float('nan'))}, "
        f"share with e25_26 >= +10%: {float((ge10['e25_26'] >= 0.10).mean()) if ge10.height else float('nan'):.2f}); top decile ({k} rows) mean e21_24 "
        f"{pct(top['e21_24'].mean())} -> e25_26 {pct(top['e25_26'].mean())}; all pickable rows mean e21_24 {pct(sub['e21_24'].mean())}, e25_26 {pct(sub['e25_26'].mean())}."
    ]


def strong_paths() -> list[str]:
    """Why: forward spot path from the 72 h check for listings above / at-or-below the reference."""
    P = paths()
    c = P["close"]
    ref = c[:, 0]
    s72 = c[:, 863] / ref - 1
    out = [
        "| group (72 h vs reference) | n | median since | fwd 1d mean / median | 3d | 7d | 30d | share 30d max >= +50% | share 30d min <= -30% | share peak-so-far hour <= 24 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    hi_so_far = np.nanargmax(P["high"][:, :864], axis=1) / BARS_H
    for nm, m in (
        ("above > 50%", s72 > 0.5),
        ("above 20-50%", (s72 > 0.2) & (s72 <= 0.5)),
        ("above 0-20%", (s72 > 0) & (s72 <= 0.2)),
        ("at or below", s72 <= 0),
    ):
        e0 = P["open"][m, 864]
        cells = []
        for d in (1, 3, 7, 30):
            f = c[m, 863 + d * 288] / e0 - 1
            cells.append(f"{np.nanmean(f) * 100:+.1f}% / {np.nanmedian(f) * 100:+.1f}%")
        mx = np.nanmax(P["high"][m, 864 : 864 + 8640], axis=1) / e0 - 1
        mn = np.nanmin(P["low"][m, 864 : 864 + 8640], axis=1) / e0 - 1
        out.append(f"| {nm} | {int(m.sum())} | {np.nanmedian(s72[m]) * 100:+.0f}% | " + " | ".join(cells)
                   + f" | {np.mean(mx >= 0.5):.2f} | {np.mean(mn <= -0.3):.2f} | {np.mean(hi_so_far[m] <= 24):.2f} |")  # fmt: skip
    return out


def pair_c(b_pick: dict[str, object]) -> list[str]:
    from e25_combined import sleeve_listing_weak
    from p2_lib import lmc, lperp, to_book

    P = paths()
    L = listings()
    idx = {s: int(i) for s, i in zip(L["symbol"], L["i"], strict=True)}
    e27 = pl.read_parquet(OWN / "e27_listings.parquet")
    strong = e27.filter(pl.col("since_list") > 0)
    e_of = {}
    for r in strong.iter_rows(named=True):
        i = idx.get(r["symbol"])
        if i is None:
            continue
        e = (int(r["t_in"]) - int(P["listed"][i])) // 300
        if 1 <= e < 1500:
            e_of[i] = e
    tpm = {"none": None, "30%": 0.3, "50%": 0.5, "100%": 1.0}
    variants = {
        "mirror 30d TP50 stop50": (8640, 0.5, 0.5),
        f"B-pick exits ({b_pick['hold']}, TP {b_pick['tp']}, stop {b_pick['stop']})": (
            B_HOLD[str(b_pick["hold"])],
            tpm[str(b_pick["tp"])],
            tpm[str(b_pick["stop"])],
        ),
    }
    M = sleeve_listing_weak().with_columns(w=pl.lit(1.0))
    base = [*lmc(), to_book(lperp(), "LP")]
    lines = [
        "",
        f"M-complement listings (e27 since_list > 0): {strong.height}, matched to 5m paths: {len(e_of)}.",
        "",
    ]
    cards: dict[str, dict[str, float]] = {}
    book_rows = [*BOOK_HEAD, book_line("M alone (2022+)", [M]), book_line("LMC+ (2022+)", base)]
    for nm, (h, tp, sl) in variants.items():
        t = trades(e_of, h, tp, sl)
        s = summary(t)
        lines.append(f"* long {nm}: n={s['n']}, mean {pct(s['mean'])}, median {pct(s['median'])}, win {float(s['win']):.2f}, stopped {float(s['stopped']):.2f}; "
                     f"2021-24 {pct(s['e21_24'])} ({s['n21_24']}), 2025-26 {pct(s['e25_26'])} ({s['n25_26']}); by year {yr_cells(s)}")  # fmt: skip
        cards[f"C long {nm}"] = card(t.with_columns(signal=pl.lit(1.0)), tp, sl)
        x = as_book(t, "XC")
        x22 = x.filter(pl.col("t_in") >= 1640995200)
        book_rows += [
            book_line(f"C long {nm} alone", [x]),
            book_line(f"M + C long {nm}", [M, x22]),
            book_line(f"LMC+ + C long {nm}", [*base, x22]),
        ]
    return [*lines, "", table(cards), "", *book_rows]


def main() -> None:
    ga, ta = grid_a()
    gb, tb = grid_b()
    ga.write_csv(OUT / "x16_grid_a.csv")
    gb.write_csv(OUT / "x16_grid_b.csv")
    pa, pb = pick(ga), pick(gb)
    tpm = {"none": None, "20%": 0.2, "30%": 0.3, "50%": 0.5, "100%": 1.0}
    ka = ("A", pa["entry"], pa["hold"], tpm[str(pa["tp"])], tpm[str(pa["stop"])], pa["subset"])
    kb = ("B", pb["check"], pb["hold"], tpm[str(pb["tp"])], tpm[str(pb["stop"])], pb["bin"])
    t_a, t_b = ta[ka], tb[kb]
    la = f"A pick: entry {pa['entry']}, hold {pa['hold']}, TP {pa['tp']}, stop {pa['stop']}, {pa['subset']}"
    lb = f"B pick: check {pb['check']}, {pb['bin']}, hold {pb['hold']}, TP {pb['tp']}, stop {pb['stop']}"
    cards = {
        la: card(t_a.with_columns(signal=pl.lit(1.0)), ka[3], ka[4]),
        lb: card(t_b.with_columns(signal=pl.lit(1.0)), kb[3], kb[4]),
    }
    # A "all listings" reference rows: buy-and-hold for each entry / hold, no exits
    for en in A_ENTRY:
        t = ta[("A", en, "7d", None, None, "all")]
        cards[f"A ref: entry {en}, 7d, no exits, all"] = card(
            t.with_columns(signal=pl.lit(1.0)), None, None
        )
    cards["B ref: 72h, above>0, 30d, no exits"] = card(
        tb[("B", "72h", "30d", None, None, "above>0")].with_columns(signal=pl.lit(1.0)), None, None
    )

    # calibration by the split variable
    cal_lines: list[str] = []
    a_pool = ta[("A", pa["entry"], pa["hold"], ka[3], ka[4], "all")]
    var = (
        "qv_1d"
        if str(pa["subset"]).startswith("qv") or (pa["subset"] == "all" and pa["entry"] == "24h")
        else "r1h"
    )
    ent_a = a_pool["i"].is_in(t_a["i"].implode())
    lines, cal = cal_block(la, a_pool, ent_a, var, ka[3], ka[4])
    cal.write_csv(OUT / "x16_calibration_a.csv")
    cal_lines += lines
    for v in ("r1h", "qv_1d"):  # the other split on the same config, all entered
        if v != var:
            lines, _ = cal_block(
                f"{la}: by {v} (all listings entered)",
                a_pool,
                pl.Series([True] * a_pool.height),
                v,
                ka[3],
                ka[4],
            )
            cal_lines += lines
    b_pool = tb[("Bpool", pb["check"], pb["hold"], kb[3], kb[4])]
    ent_b = b_pool["i"].is_in(t_b["i"].implode())
    lines, cal = cal_block(lb, b_pool, ent_b, "since", kb[3], kb[4])
    cal.write_csv(OUT / "x16_calibration_b.csv")
    cal_lines += lines

    # books
    from p2_lib import lmc, lperp, to_book

    base = [*lmc(), to_book(lperp(), "LP")]
    book_rows = [*BOOK_HEAD, book_line("LMC+ (2022+)", base)]
    for nm, t in (("A pick", t_a), ("B pick", t_b)):
        x = as_book(t, "X")
        x22 = x.filter(pl.col("t_in") >= 1640995200)
        book_rows += [book_line(f"{nm} alone", [x]), book_line(f"LMC+ + {nm}", [*base, x22])]
        xr = as_book(t.filter(~pl.col("symbol").str.replace("USDT$", "").is_in(list(RENAMES))), "X")
        book_rows.append(book_line(f"{nm} alone, renames dropped", [xr]))
    ren = [f"* {nm}: renames dropped -> n={summary(t.filter(~pl.col('symbol').str.replace('USDT$', '').is_in(list(RENAMES))))['n']}, "
           f"mean {pct(summary(t.filter(~pl.col('symbol').str.replace('USDT$', '').is_in(list(RENAMES))))['mean'])} (was {pct(t['net'].mean())})" for nm, t in (("A pick", t_a), ("B pick", t_b))]  # fmt: skip

    c_lines = pair_c(pb)
    md = [
        "# X16 raw output: longs around new spot listings", "", "Script `x16_listing_longs.py` (grid declared in its docstring).", "",
        "## Picks", "", f"* {la}: {yr_cells(pa)}", f"* {lb}: {yr_cells(pb)}", *ren, "", table(cards), "",
        "## Selection check", "", *selection_check(ga, "A"), *selection_check(gb, "B"), "",
        "## Books (10 slots, $10k each year)", "", *book_rows, "",
        "## C: M + long on M's strong complement", *c_lines, "",
        "## Forward spot paths from the 72 h check (all 397 listings)", "", *strong_paths(), "",
        "## Calibration", "", *cal_lines,
        "## Grid B (432 rows)", "", *grid_md(gb, ["check", "bin", "hold", "tp", "stop"]), "",
        "## Grid A (1024 rows; eligible = split known at entry)", "", *grid_md(ga, ["entry", "hold", "tp", "stop", "subset", "eligible"]), "",
    ]  # fmt: skip
    (OUT / "listing_longs_raw.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:120]))


if __name__ == "__main__":
    main()
