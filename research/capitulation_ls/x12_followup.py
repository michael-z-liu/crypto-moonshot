"""X12 follow-up (post hoc, after the pre-declared tables of x12_book_depth.py were read).

1. L within-day IC by period, per-day sign count, and controls (coin size qv, the CNN signal).
2. Why the R1 tie-break lost in the book: L trades that base and R1 take on each flush day.
3. Post-hoc book rows: L tie-break on dep5_qv / on qv (control); depth gate on F only, and the
   same gate on qv (control). These are flagged as post hoc in book_depth.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from x12_book_depth import COLS, features, ic_day, l_rules, num, run_book

OUT = Path(__file__).resolve().parent / "results" / "explore"


def per_day_ic(x: pl.DataFrame, f: str) -> tuple[int, int]:
    neg = pos = 0
    for _, g in x.drop_nulls(f).group_by("day"):
        if g.height < 10:
            continue
        r = g.select(pl.corr(pl.col(f).rank(), pl.col("net").rank())).item()
        if r is None or np.isnan(r):
            continue
        neg += int(r < 0)
        pos += int(r > 0)
    return neg, pos


def taken_l(t: pl.DataFrame, combo: str) -> pl.DataFrame:
    """L trades the book takes (book()'s trades_out: year, t_out, pnl, sleeve)."""
    from e25_combined import book

    out: list[dict[str, object]] = []
    book(t.filter(pl.col("sleeve").is_in(list(combo))).select(COLS), 10, trades_out=out)
    o = pl.DataFrame(out).filter(pl.col("sleeve") == "L")
    return o


def main() -> None:
    x = features()
    lx = x.filter((pl.col("sleeve") == "L") & pl.col("has_depth")).with_columns(
        lqv=pl.col("qv").log10(), cnn=pl.col("signal")
    )
    lines = [
        "## Follow-up (post hoc)",
        "",
        "### L within-day IC by period and controls",
        "",
        "| feature | IC_day 23-24 (days) | IC_day 25-26 (days) | days IC<0 / IC>0 (>= 10 cands) |",
        "|---|---|---|---|",
    ]
    for f in ("dep5_qv", "dep1_qv", "bid2_qv", "ask2_qv", "ch5", "bid5", "lqv", "cnn"):
        a, na = ic_day(lx.filter(pl.col("per") == "23-24"), f)
        b, nb = ic_day(lx.filter(pl.col("per") == "25-26"), f)
        n, p = per_day_ic(lx, f)
        lines.append(f"| {f} | {a:+.3f} ({na}) | {b:+.3f} ({nb}) | {n} / {p} |")
    # partial: dep5_qv residual on log qv within day
    z = lx.drop_nulls("dep5_qv").with_columns(ld=pl.col("dep5_qv").log10())
    z = z.with_columns(
        ld_c=pl.col("ld") - pl.col("ld").mean().over("day"),
        q_c=pl.col("lqv") - pl.col("lqv").mean().over("day"),
    )
    beta = num(z["ld_c"].dot(z["q_c"])) / num(z["q_c"].dot(z["q_c"]))
    z = z.with_columns(dep5_resid=pl.col("ld_c") - beta * pl.col("q_c"))
    r, nr = ic_day(z, "dep5_resid")
    rho = float(z.select(pl.corr(pl.col("ld_c"), pl.col("q_c"))).item())
    lines += [
        "",
        f"Within-day correlation of log dep5_qv with log qv: {rho:+.2f}; within-day IC of "
        f"dep5_qv with log qv partialled out: {r:+.3f} ({nr} days).",
        "",
    ]
    # partial vs the CNN: residual of the within-day depth rank on the within-day CNN rank
    lines += [
        "| feature | within-day IC after removing the CNN rank (days) | top-10 by CNN: mean net bp | top-10 by CNN+depth composite |",  # noqa: E501
        "|---|---|---|---|",
    ]
    zc = lx.filter(pl.len().over("day") >= 3)
    for f in ("dep5_qv", "ch5", "ask2_qv", "bid2_qv"):
        w = (
            zc.drop_nulls(f)
            .with_columns(
                rf=(pl.col(f).rank() / pl.len()).over("day"),
                rc=(pl.col("cnn").rank() / pl.len()).over("day"),
            )
            .with_columns(
                rf_c=pl.col("rf") - pl.col("rf").mean().over("day"),
                rc_c=pl.col("rc") - pl.col("rc").mean().over("day"),
            )
        )
        bb = num(w["rf_c"].dot(w["rc_c"])) / num(w["rc_c"].dot(w["rc_c"]))
        w = w.with_columns(
            res=pl.col("rf_c") - bb * pl.col("rc_c"), comp=pl.col("rc") + 0.5 * (1 - pl.col("rf"))
        )
        pr, npr = ic_day(w, "res")
        big = w.filter(pl.len().over("day") >= 20)
        t1 = big.sort("cnn", descending=True).group_by("day").head(10)["net"].mean()
        t2 = big.sort("comp", descending=True).group_by("day").head(10)["net"].mean()
        lines.append(f"| {f} | {pr:+.3f} ({npr}) | {num(t1) * 1e4:+.0f} | {num(t2) * 1e4:+.0f} |")
    lines.append("")
    # 2. why R1 lost
    rules = l_rules(x)
    rows = []
    for name in ("base", "R1 tie bid2_qv asc"):
        o = taken_l(rules[name], "LMC")
        rows.append(o.with_columns(rule=pl.lit(name)))
    tk = pl.concat(rows, how="diagonal_relaxed")
    lines += [
        "### L trades taken in the LMC book: base vs R1 (L pnl in $ of the yearly $10k book)",
        "",
        "| year | base n | R1 n | base L pnl $ | R1 L pnl $ |",
        "|---|---|---|---|---|",
    ]
    for y in range(2023, 2027):
        s = [
            tk.filter((pl.col("rule") == r) & (pl.col("year") == y))
            for r in ("base", "R1 tie bid2_qv asc")
        ]
        lines.append(
            f"| {y} | {s[0].height} | {s[1].height} | {float(s[0]['pnl'].sum()):+.0f} | {float(s[1]['pnl'].sum()):+.0f} |"  # noqa: E501
        )
    # 3. post-hoc book rows
    base = rules["base"]
    isl = pl.col("sleeve") == "L"

    def tie(f: str, asc: bool) -> pl.DataFrame:
        r = (pl.col(f).rank() / pl.col(f).count()).over("t_in")
        t = ((1.0 - r) if asc else r).fill_null(0.5) * 0.999
        return base.with_columns(prio=pl.when(isl).then(3.0 + t).otherwise(pl.col("prio")))

    a33 = num(
        x.filter(pl.col("sleeve").is_in(["F", "M", "C"]) & (pl.col("year") <= 2024))[
            "ask2_qv"
        ].quantile(1 / 3)
    )
    f33 = num(
        x.filter((pl.col("sleeve") == "F") & (pl.col("year") <= 2024))["ask1"].quantile(1 / 3)
    )
    q33 = num(x.filter((pl.col("sleeve") == "F") & (pl.col("year") <= 2024))["qv"].quantile(1 / 3))
    isf = pl.col("sleeve") == "F"
    comp = base.with_columns(
        rc=(pl.col("signal").rank() / pl.col("signal").count()).over("t_in"),
        rf=(pl.col("dep5_qv").rank() / pl.col("dep5_qv").count()).over("t_in"),
    ).with_columns(c=(pl.col("rc") + 0.5 * (1 - pl.col("rf").fill_null(0.5))) / 1.5)
    post = {
        "P0 L tie CNN rank + 0.5 x low-dep5_qv rank": comp.with_columns(
            prio=pl.when(isl).then(3.0 + pl.col("c") * 0.999).otherwise(pl.col("prio"))
        ),
        "P1 L tie dep5_qv asc": tie("dep5_qv", True),
        "P2 L tie qv desc (control)": base.with_columns(lq=pl.col("qv")).pipe(
            lambda d: d.with_columns(
                prio=pl.when(isl)
                .then(3.0 + ((pl.col("lq").rank() / pl.len()).over("t_in")) * 0.999)
                .otherwise(pl.col("prio"))
            )
        ),
        f"P3 F only: ask2_qv >= {a33:.4f}": base.with_columns(
            w=pl.when(isf & (pl.col("ask2_qv") < a33).fill_null(False)).then(0.0).otherwise(1.0)
        ),
        f"P4 F only: ask1 >= 10^{f33:.2f} (2023-24 F 33rd pct)": base.with_columns(
            w=pl.when(isf & (pl.col("ask1") < f33).fill_null(False)).then(0.0).otherwise(1.0)
        ),
        f"P5 F only: qv >= {q33 / 1e6:.1f}M (control)": base.with_columns(
            w=pl.when(isf & (pl.col("qv") < q33)).then(0.0).otherwise(1.0)
        ),
    }
    # F stops by qv tercile (control for "thin asks" = small coin)
    fx = x.filter(isf & pl.col("has_depth"))
    q1, q2 = fx["qv"].quantile(1 / 3), fx["qv"].quantile(2 / 3)
    st = [
        num(fx.filter((pl.col("qv") > lo) & (pl.col("qv") <= hi))["stopped"].mean()) * 100
        for lo, hi in ((-np.inf, q1), (q1, q2), (q2, np.inf))
    ]
    fs = fx.filter(pl.col("stopped")).select(
        "symbol", "year", "net", "ask1", "ask2_qv", "qv", "imb1"
    )
    lines += [
        "",
        f"F stopped % by qv tercile low/mid/high: {st[0]:.0f}/{st[1]:.0f}/{st[2]:.0f}.",
        "",
        "F stop-outs (ask1 = log10 USDT within 1% above mid):",
        "",
        "| symbol | year | net | ask1 | ask2_qv | qv $M | imb1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in fs.iter_rows(named=True):
        lines.append(
            f"| {row['symbol']} | {row['year']} | {row['net']:+.2f} | {row['ask1']:.2f} | {row['ask2_qv']:.4f} | {row['qv'] / 1e6:.1f} | {row['imb1']:+.2f} |"  # noqa: E501
        )
    lines += [
        "",
        "### Post-hoc book rows (10 slots, 2023-26)",
        "",
        "| book | rule | trades | bp/trade | L trades | L bp | min year % | worst DD % | years 23/24/25/26 % |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for combo in ("LMC", "LMCF"):
        for name, t in {"base": base, **post}.items():
            if name.startswith(("P3", "P4", "P5")) and combo == "LMC":
                continue
            rb = run_book(t.filter(pl.col("sleeve").is_in(list(combo))))
            lines.append(
                f"| {combo} | {name} | {rb['n']} | {rb['bp']:+.0f} | {rb['nL']} | {rb['bpL']:+.0f} | "  # noqa: E501
                f"{rb['min'] * 100:+.0f} | {rb['dd'] * 100:+.0f} | {rb['years']} |"
            )
    txt = "\n".join(lines) + "\n"
    print(txt)
    (OUT / "book_depth_followup.md").write_text(txt)


if __name__ == "__main__":
    main()
