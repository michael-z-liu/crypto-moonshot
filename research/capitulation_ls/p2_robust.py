"""P2.1 (perp-only follow-ups): robustness of LMC+ (LMC + L on perp-only coins, 10 slots).

1. Day-block bootstrap (flush days resampled): per-trade mean of L-perp-only and of all L trades
   (spot + perp-only), and the LMC / LMC+ book with the L entry days of each year resampled with
   replacement (a day drawn k times enters its trades at k stakes; M and C fixed). Paired: the same
   draws for both books.
2. Starting equity $10k / $50k / $250k (the 0.5%-of-daily-volume stake cap binds).
3. Entries one 5m candle late (L, L-perp-only and C; M enters hourly and is unchanged).
4. LMCF (x8_lib.sleeves(): L, M, C, F) with and without L-perp-only.

Writes results/explore/p2_robust.md (pasted under "Follow-ups" in perp_only.md).
Run: uv run --group research python p2_robust.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from e25_combined import book
from p2_lib import BOOK_COLS, YEARS, book_row, lmc, lperp, md, to_book

OUT = Path(__file__).resolve().parent / "results" / "explore"
REPS = 400


def day_boot(
    net: np.ndarray, day: np.ndarray, reps: int = 4000, seed: int = 1
) -> tuple[float, float]:
    """95% CI of the per-trade mean with days resampled with replacement."""
    rng = np.random.default_rng(seed)
    ud, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, weights=net)
    n = np.bincount(inv).astype(float)
    w = rng.multinomial(len(ud), np.full(len(ud), 1 / len(ud)), size=reps).astype(float)
    m = (w @ s) / (w @ n)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def with_day(t: pl.DataFrame) -> pl.DataFrame:
    return t.with_columns(
        day=pl.from_epoch("t_in", "s").dt.date(), year=pl.from_epoch("t_in", "s").dt.year()
    )


def stats(b: pl.DataFrame) -> tuple[float, float, dict[int, float]]:
    n = int(b["trades"].sum())
    bp = float((b["bp"] * b["trades"]).sum() / n)
    yrs = {int(r["year"]): float(r["ret"]) for r in b.iter_rows(named=True)}
    return bp, min(yrs.values()), yrs


def main() -> None:
    lines = ["### F1. Robustness of LMC+", ""]
    base = lmc()
    lp = lperp()
    xl = to_book(lp, "L").select(BOOK_COLS)
    plus = [*base, xl]

    # ------------------------------------------------ 1. bootstrap
    spot_l = with_day(base[0])
    rows = []
    for name, t in (
        ("L-perp-only", with_day(lp.select("t_in", "net"))),
        ("spot L", spot_l.select("t_in", "net", "day", "year")),
        (
            "all L (spot + perp-only)",
            pl.concat([spot_l.select("t_in", "net"), lp.select("t_in", "net")]),
        ),
    ):
        t = with_day(t.select("t_in", "net"))
        lo, hi = day_boot(t["net"].to_numpy(), t["day"].to_numpy())
        rows.append(
            {
                "trades": name,
                "n": t.height,
                "days": t["day"].n_unique(),
                "mean net": float(t["net"].mean()),
                "95% CI (days resampled)": f"{lo:+.3f} .. {hi:+.3f}",
            }
        )
    lines += ["Per-trade mean, day-block bootstrap (4,000 draws):", "", md(pl.DataFrame(rows)), ""]

    cand_b = with_day(pl.concat(base, how="vertical_relaxed"))
    cand_p = with_day(pl.concat(plus, how="vertical_relaxed"))
    l_days = cand_p.filter(pl.col("sleeve") == "L").select("year", "day").unique().sort("day")
    bp0, mn0, y0 = stats(book(cand_b.drop("day", "year"), 10))
    bp1, mn1, y1 = stats(book(cand_p.drop("day", "year"), 10))
    rng = np.random.default_rng(7)
    res: dict[str, list[tuple[float, float, dict[int, float]]]] = {"LMC": [], "LMC+": []}
    for k in range(REPS):
        cnt = []
        for y in YEARS:
            dd = l_days.filter(pl.col("year") == y)["day"]
            c = rng.multinomial(len(dd), np.full(len(dd), 1 / len(dd)))
            cnt.append(pl.DataFrame({"day": dd, "k": c}))
        cnt_df = pl.concat(cnt)
        for name, cand in (("LMC", cand_b), ("LMC+", cand_p)):
            c = cand.join(cnt_df, on="day", how="left").with_columns(
                k=pl.when(pl.col("sleeve") == "L").then(pl.col("k").fill_null(0)).otherwise(1)
            )
            c = c.filter(pl.col("k") > 0).with_columns(w=pl.col("k").cast(pl.Float64))
            res[name].append(stats(book(c.drop("day", "year", "k"), 10)))
        if k % 50 == 0:
            print("boot", k, file=sys.stderr, flush=True)
    brow = []
    for name, (bp, mn, _yr) in (("LMC", (bp0, mn0, y0)), ("LMC+", (bp1, mn1, y1))):
        r = res[name]
        bps = np.array([x[0] for x in r])
        mins = np.array([x[1] for x in r])
        row: dict[str, object] = {"book": name, "bp (actual)": round(bp),
                                  "bp 5-95%": f"{np.percentile(bps, 5):.0f} .. {np.percentile(bps, 95):.0f}",  # noqa: E501
                                  "worst yr (actual)": f"{mn * 100:+.0f}%",
                                  "worst yr 5% / 50%": f"{np.percentile(mins, 5) * 100:+.0f}% / {np.median(mins) * 100:+.0f}%",  # noqa: E501
                                  "P(worst yr < 0)": f"{np.mean(mins < 0):.3f}"}  # fmt: skip
        for y in YEARS:
            v = np.array([x[2].get(y, 0.0) for x in r])
            row[f"{y} median (5%)"] = (
                f"{np.median(v) * 100:+.0f}% ({np.percentile(v, 5) * 100:+.0f}%)"
            )
        brow.append(row)
    diff = np.array([b[1] - a[1] for a, b in zip(res["LMC"], res["LMC+"], strict=True)])
    lines += [
        f"Book with the L entry days of each year resampled ({REPS} paired draws; M and C fixed):",
        "",
        md(pl.DataFrame(brow)),
        "",
        f"Worst year LMC+ minus LMC over the same draws: median {np.median(diff) * 100:+.0f} pts, "
        f"5% {np.percentile(diff, 5) * 100:+.0f} pts, share of draws where LMC+ is worse {np.mean(diff < 0):.2f}.",  # noqa: E501
        "",
    ]

    # ------------------------------------------------ 2. starting equity
    rows2 = []
    for start in (10_000.0, 50_000.0, 250_000.0):
        for name, parts in (("LMC", base), ("LMC+", plus)):
            r = book_row(
                f"{name} ${start / 1000:.0f}k",
                book(pl.concat(parts, how="vertical_relaxed"), 10, start=start),
            )
            rows2.append(r)
    lines += ["Starting equity (each year restarts at this amount; stake <= 0.5% of the coin's daily volume):", "",  # noqa: E501
              md(pl.DataFrame(rows2)), ""]  # fmt: skip

    # ------------------------------------------------ 3. one candle late
    base_d = lmc(delay=1)
    lp_d = lperp(delay=1)
    rows3 = [
        book_row("LMC, on time", book(pl.concat(base, how="vertical_relaxed"), 10)),
        book_row("LMC+, on time", book(pl.concat(plus, how="vertical_relaxed"), 10)),
        book_row("LMC, 1 candle late", book(pl.concat(base_d, how="vertical_relaxed"), 10)),
        book_row(
            "LMC+, 1 candle late",
            book(
                pl.concat([*base_d, to_book(lp_d, "L").select(BOOK_COLS)], how="vertical_relaxed"),
                10,
            ),
        ),
    ]
    lines += ["Entries one 5m candle late (L, L-perp-only, C):", "", md(pl.DataFrame(rows3)), "",
              f"L-perp-only alone: on time n {lp.height}, mean net {lp['net'].mean():+.4f}; "
              f"1 candle late n {lp_d.height}, mean net {lp_d['net'].mean():+.4f}, "
              f"TP rate {(lp_d['exit_kind'] == 'tp').mean():.2f}.", ""]  # fmt: skip

    # ------------------------------------------------ 4. LMCF
    import x8_lib

    s = x8_lib.sleeves()
    fr = [x8_lib.as_book(s.filter(pl.col("sleeve") == k)).with_columns(w=pl.lit(1.0)).select(BOOK_COLS)  # noqa: E501
          for k in ("L", "M", "C", "F")]  # fmt: skip
    rows4 = [
        book_row("LMCF", book(pl.concat(fr, how="vertical_relaxed"), 10)),
        book_row("LMCF + L-perp-only", book(pl.concat([*fr, xl], how="vertical_relaxed"), 10)),
    ]
    lines += ["LMCF (x8_lib.sleeves()) with L-perp-only:", "", md(pl.DataFrame(rows4)), ""]
    (OUT / "p2_robust.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
