"""X15: LONG pumps on perp-only coins (in-sample exploration; the mirror of f1's failed shorts).

f1 found that 19% of perp-only pumps (ret_4h >= 40%) reach +100% within 72h (5% on spot coins).
Pre-declared grid: long the perp at the next 5m open, 1x, TP 30/50/100%, stop 20/30/50% or none,
hold 24h/72h/7d, all pumps or only those with funding < 0 (crowded shorts). Cost 0.3% + funding
(by perp symbol). One position per coin. Split 2025 / 2026 (perp-only coins barely exist before).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import polars as pl
from f1_lib import funding_paid_perp, perp_only_pool
from p2_lib import sim_perp
from sim import Rule

OUT = Path(__file__).resolve().parent / "results" / "explore"


def run(c: pl.DataFrame, tp: float, sl: float | None, hold_h: int) -> pl.DataFrame:
    rule = Rule(hold_min=hold_h * 60, tp=tp, sl=sl, slots=100_000, cost=0.003, liq_frac=None)
    t = sim_perp(c, 1, rule)
    fund = funding_paid_perp(t["symbol"].to_numpy(), t["t_in"].to_numpy(), t["t_out"].to_numpy(),
                             np.ones(t.height))  # fmt: skip
    return t.with_columns(net=pl.Series(t["ret"].to_numpy() - 0.003 - fund))


def grid() -> None:
    pool = perp_only_pool()
    lines = [
        f"# X15: long pumps on perp-only coins (pool n={pool.height}, ret_4h >= 40%)",
        "",
        "| pool | TP | stop | hold h | n | mean net bp | 2025 bp | 2026 bp | win | TP hit | stopped |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    res = []
    for gname, tp, sl, hold in itertools.product(
        ("all", "funding<0"), (0.3, 0.5, 1.0), (0.2, 0.3, 0.5, None), (24, 72, 168)
    ):
        c = pool if gname == "all" else pool.filter(pl.col("rate").fill_nan(None).fill_null(0) < 0)
        t = run(c, tp, sl, hold)
        y = {
            int(r["year"]): float(r["m"])
            for r in t.group_by("year").agg(m=pl.col("net").mean()).iter_rows(named=True)
        }
        row = (f"| {gname} | {tp:.0%} | {'-' if sl is None else f'{sl:.0%}'} | {hold} | {t.height} | "  # noqa: E501
               f"{t['net'].mean() * 1e4:+.0f} | {y.get(2025, float('nan')) * 1e4:+.0f} | "
               f"{y.get(2026, float('nan')) * 1e4:+.0f} | {(t['net'] > 0).mean():.2f} | "
               f"{(t['exit_kind'] == 'tp').mean():.2f} | {(t['exit_kind'] == 'stop').mean():.2f} |")  # fmt: skip  # noqa: E501
        res.append((t["net"].mean(), row))
        lines.append(row)
        print(row, flush=True)
    pos = sum(1 for m, _ in res if m > 0)
    lines += ["", f"{pos} of {len(res)} configurations have a positive mean."]
    (OUT / "perp_only_pump_longs.md").write_text("\n".join(lines) + "\n")


def day_ci(t: pl.DataFrame, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of the mean net with calendar days resampled."""
    g = (
        t.with_columns(d=pl.from_epoch("t", "s").dt.date())
        .group_by("d")
        .agg(s=pl.col("net").sum(), k=pl.len())
    )
    s, k = g["s"].to_numpy(), g["k"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n, len(s)))
    m = s[idx].sum(1) / k[idx].sum(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def books() -> None:
    from e25_combined import book
    from p2_lib import book_row, lmc, lperp, to_book

    pool = perp_only_pool()
    base = [*lmc(), to_book(lperp(), "LP")]
    rows = [book_row("LMC+", book(pl.concat(base, how="vertical_relaxed"), 10))]
    out = ["", "## Two configurations in the LMC+ book (pump longs at lowest priority)", ""]
    for name, c, tp, sl in (
        ("all, TP 100%, no stop, 72h", pool, 1.0, None),
        ("funding<0, TP 100%, stop 30%, 72h", pool.filter(pl.col("rate").fill_nan(None).fill_null(0) < 0), 1.0, 0.3),  # noqa: E501
    ):  # fmt: skip
        t = run(c, tp, sl, 72)
        lo, hi = day_ci(t)
        out.append(f"- {name}: n={t.height}, mean {t['net'].mean() * 1e4:+.0f} bp, "
                   f"day-bootstrap 95% CI {lo * 1e4:+.0f}..{hi * 1e4:+.0f} bp, days {t['day'].n_unique()}, "  # noqa: E501
                   f"median {t['net'].median() * 1e4:+.0f} bp")  # fmt: skip
        pb = t.select(
            "symbol", adv=pl.col("adv").clip(0, 1), t_in="t_in", t_out="t_out", net="net",
            qv=pl.col("qv_base") * 288, sleeve=pl.lit("PL"), prio=pl.lit(0.5), w=pl.lit(1.0),
        )  # fmt: skip
        rows.append(
            book_row(f"LMC+ + {name}", book(pl.concat([*base, pb], how="vertical_relaxed"), 10))
        )
    df = pl.DataFrame(rows)
    out += ["", "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.iter_rows()]
    txt = "\n".join(out) + "\n"
    print(txt)
    with (OUT / "perp_only_pump_longs.md").open("a") as f:
        f.write(txt)


if __name__ == "__main__":
    import sys

    if "--books" not in sys.argv:
        grid()
    books()
