"""X12: does perp order-book depth at entry predict the four sleeves' outcomes and tails?

Data: x12_fetch.py (Binance USD-M bookDepth, 5-minute last snapshots, 2023+). Trades: the cached
sleeve frame x8_sleeves.parquet (L, C, F, M; every candidate, no slot limit), 2023-26 only.
In-sample exploration. Everything below was declared before any outcome was looked at.

Snapshot: the last one at or before t_in (stale by <= 15 min, else missing); the "24h earlier"
snapshot is the last one at or before t_in - 24h (same tolerance). Perp = the spot name, else the
1000-prefixed perp. qv = the coin's 24h spot quote volume from the frame.

Features (notional in USDT; b = bids below mid, a = asks above mid, k = within k%):
  bid1/bid2/bid5, ask1/ask2/ask5   log10 notional
  imb1, imb5                      (b - a) / (b + a) at 1% and 5%
  dep1_qv, dep5_qv                (b + a) / qv at 1% and 5%
  bid2_qv, ask2_qv                b2 / qv, a2 / qv
  ch1, ch5                        log(depth now / depth 24h earlier), b + a at 1% and 5%

Hypotheses and rules:
  L (longs): thin bids / ask-heavy books at the flush give bigger bounces: low bid2_qv and low
    imb1 are better. Book rules in the 10-slot LMC / LMCF books (2023-26):
      R1 tie-break: among L candidates of the same 5-minute bar, take the lowest bid2_qv first
      R2 tie-break: same with imb1
      R3 skip: no L trade when bid2_qv >= the 2023-24 L 67th percentile (thick bids)
      R4 skip: no L trade when imb1 >= the 2023-24 L 67th percentile (bid-heavy book)
    (missing depth: tie-break rank 0.5, never skipped)
  F, M, C (shorts): thin asks predict squeezes / stop-outs: low ask2_qv and high imb1 are worse.
    Gates, applied to all three short sleeves at once (missing depth passes):
      G1 ask2_qv >= the 2023-24 pooled-short 33rd percentile (drop the thinnest-ask tercile)
      G2 imb1 <= +0.25 (drop bid-heavy books)
      G3 ch5 >= log(0.5) (drop books that halved in 24h)
      G4 dep1_qv >= the 2023-24 pooled-short 33rd percentile
  Squeeze = stopped (diag mode) or mae >= 30%.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from common_ls import OWN
from diag import annotate
from e25_combined import book
from futures_ls import spot_name

OUT = Path(__file__).resolve().parent / "results" / "explore"
BD = OWN / "bookdepth"
COLS = ["symbol", "adv", "t_in", "t_out", "net", "qv", "sleeve", "prio", "w"]
FEATS = [
    "bid1",
    "bid2",
    "bid5",
    "ask1",
    "ask2",
    "ask5",
    "imb1",
    "imb5",
    "dep1_qv",
    "dep5_qv",
    "bid2_qv",
    "ask2_qv",
    "ch1",
    "ch5",
]
TOL = 900


def num(v: Any) -> float:
    """A polars scalar as a float (None -> nan)."""
    return float("nan") if v is None else float(v)


def perp_of() -> dict[str, str]:
    f = pl.read_parquet(OWN / "futures" / "funding.parquet", columns=["symbol"])
    perps = sorted(set(f["symbol"].unique().to_list()))
    out = {p: p for p in perps}
    for p in perps:
        s = spot_name(p)
        if s != p and s not in out:
            out[s] = p
    return out


def features(refresh: bool = False) -> pl.DataFrame:
    p = BD / "x12_feats.parquet"
    if p.exists() and not refresh:
        return pl.read_parquet(p)
    s = pl.read_parquet(OWN / "explore" / "x8_sleeves.parquet").filter(pl.col("year") >= 2023)
    pm = perp_of()
    s = s.with_row_index("i").with_columns(
        perp=pl.col("symbol").replace_strict(pm, default=None),
        day=pl.from_epoch("t_in").dt.date(),
    )
    d = pl.read_parquet(BD / "depth5m.parquet").sort("perp", "ts")
    now = s.filter(pl.col("perp").is_not_null()).sort("t_in")
    j = now.join_asof(
        d, left_on="t_in", right_on="ts", by="perp", strategy="backward", tolerance=TOL
    )
    prev = (
        now.select("i", "perp", t24=pl.col("t_in") - 86400)
        .sort("t24")
        .join_asof(
            d.select("perp", "ts", p1=pl.col("b1") + pl.col("a1"), p5=pl.col("b5") + pl.col("a5")),
            left_on="t24",
            right_on="ts",
            by="perp",
            strategy="backward",
            tolerance=TOL,
        )
        .select("i", "p1", "p5")
    )
    j = j.join(prev, on="i", how="left")
    lg = pl.col
    f = j.select(
        "i",
        has_depth=lg("b1").is_not_null() & lg("a1").is_not_null(),
        bid1=lg("b1").log10(),
        bid2=lg("b2").log10(),
        bid5=lg("b5").log10(),
        ask1=lg("a1").log10(),
        ask2=lg("a2").log10(),
        ask5=lg("a5").log10(),
        imb1=(lg("b1") - lg("a1")) / (lg("b1") + lg("a1")),
        imb5=(lg("b5") - lg("a5")) / (lg("b5") + lg("a5")),
        dep1_qv=(lg("b1") + lg("a1")) / lg("qv"),
        dep5_qv=(lg("b5") + lg("a5")) / lg("qv"),
        bid2_qv=lg("b2") / lg("qv"),
        ask2_qv=lg("a2") / lg("qv"),
        ch1=((lg("b1") + lg("a1")) / lg("p1")).log(),
        ch5=((lg("b5") + lg("a5")) / lg("p5")).log(),
    )
    out = s.join(f, on="i", how="left").with_columns(pl.col("has_depth").fill_null(False))
    out = out.with_columns([pl.col(c).fill_nan(None) for c in FEATS]).with_columns(
        [pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c) for c in FEATS]
    )
    out = annotate(out).with_columns(
        squeeze=(pl.col("mode") == "stopped") | (pl.col("mae") >= 0.30),
        stopped=pl.col("mode") == "stopped",
        per=pl.when(pl.col("year") <= 2024).then(pl.lit("23-24")).otherwise(pl.lit("25-26")),
    )
    out.write_parquet(p)
    return out


# ---------------------------------------------------------------- tables ----------------------
def terciles(x: pl.DataFrame, f: str, col: str = "net") -> list[float]:
    z = x.drop_nulls(f)
    if z.height < 15:
        return [float("nan")] * 3
    q1, q2 = z[f].quantile(1 / 3), z[f].quantile(2 / 3)
    out = []
    for lo, hi in ((-np.inf, q1), (q1, q2), (q2, np.inf)):
        s = z.filter((pl.col(f) > lo) & (pl.col(f) <= hi))
        v = s[col].cast(pl.Float64).mean() if s.height else None
        out.append(num(v) if v is not None else num("nan"))
    return out


def ic(x: pl.DataFrame, f: str) -> float:
    z = x.drop_nulls(f)
    if z.height < 20:
        return float("nan")
    return float(z.select(pl.corr(pl.col(f).rank(), pl.col("net").rank())).item())


def ic_day(x: pl.DataFrame, f: str) -> tuple[float, int]:
    z = x.drop_nulls(f).filter(pl.len().over("day") >= 3)
    if z.height < 30:
        return float("nan"), 0
    z = z.with_columns(
        rf=(pl.col(f).rank() / pl.len()).over("day"),
        rn=(pl.col("net").rank() / pl.len()).over("day"),
    )
    v = z.select(
        pl.corr(
            pl.col("rf") - pl.col("rf").mean().over("day"),
            pl.col("rn") - pl.col("rn").mean().over("day"),
        )
    ).item()
    return float(v), int(z["day"].n_unique())


def fmt3(v: list[float], scale: float = 1e4, pct: bool = False) -> str:
    if pct:
        return "/".join("nan" if np.isnan(a) else f"{a * 100:.0f}" for a in v)
    return "/".join("nan" if np.isnan(a) else f"{a * scale:+.0f}" for a in v)


def tercile_table(x: pl.DataFrame, sl: str) -> list[str]:
    a, b = x.filter(pl.col("per") == "23-24"), x.filter(pl.col("per") == "25-26")
    short = sl != "L"
    head = "| feature | 23-24 bp low/mid/high | 25-26 bp low/mid/high | IC | IC_day (days) |"
    sep = "|---|---|---|---|---|"
    if short:
        head += " stopped % low/mid/high | squeeze % low/mid/high |"
        sep += "---|---|"
    lines = [head, sep]
    for f in FEATS:
        d, nd = ic_day(x, f)
        row = (
            f"| {f} | {fmt3(terciles(a, f))} | {fmt3(terciles(b, f))} | {ic(x, f):+.3f} | "
            f"{d:+.3f} ({nd}) |"
        )
        if short:
            row += f" {fmt3(terciles(x, f, 'stopped'), pct=True)} | {fmt3(terciles(x, f, 'squeeze'), pct=True)} |"  # noqa: E501
        lines.append(row)
    return lines


# ---------------------------------------------------------------- books -----------------------
def run_book(t: pl.DataFrame) -> dict[str, Any]:
    b = book(t.select(COLS), 10)
    n = int(b["trades"].sum())
    ys = {int(r["year"]): r for r in b.iter_rows(named=True)}
    return {
        "n": n,
        "bp": float((b["bp"] * b["trades"]).sum() / n),
        "min": num(b["ret"].min()),
        "dd": num(b["dd"].min()),
        "years": "/".join(
            f"{ys[y]['ret'] * 100:+.0f}" if y in ys else "na" for y in range(2023, 2027)
        ),
        "nL": int(b["nL"].sum()),
        "bpL": float((b["bpL"] * b["nL"]).sum() / max(1, int(b["nL"].sum()))),
    }


def l_rules(x: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Each rule returns the full 2023-26 frame with prio / w changed for L only."""
    base = x.with_columns(w=pl.lit(1.0))
    ref = x.filter((pl.col("sleeve") == "L") & (pl.col("year") <= 2024))
    out = {"base": base}
    isl = pl.col("sleeve") == "L"
    for name, f in (("R1 tie bid2_qv asc", "bid2_qv"), ("R2 tie imb1 asc", "imb1")):
        r = (pl.col(f).rank() / pl.col(f).count()).over("t_in")  # 0..1 within the bar
        tie = (1.0 - r).fill_null(0.5) * 0.999  # low feature first
        out[name] = base.with_columns(prio=pl.when(isl).then(3.0 + tie).otherwise(pl.col("prio")))
    for name, f in (
        ("R3 skip bid2_qv top tercile", "bid2_qv"),
        ("R4 skip imb1 top tercile", "imb1"),
    ):
        cut = num(ref[f].quantile(2 / 3))
        out[name] = base.with_columns(
            w=pl.when(isl & (pl.col(f) >= cut).fill_null(False)).then(0.0).otherwise(1.0)
        )
    return out


def short_gates(x: pl.DataFrame) -> dict[str, pl.Expr]:
    ref = x.filter(pl.col("sleeve").is_in(["F", "M", "C"]) & (pl.col("year") <= 2024))
    a33 = num(ref["ask2_qv"].quantile(1 / 3))
    d33 = num(ref["dep1_qv"].quantile(1 / 3))
    return {
        "base": pl.lit(True),
        f"G1 ask2_qv >= {a33:.4f}": pl.col("ask2_qv") >= a33,
        "G2 imb1 <= +0.25": pl.col("imb1") <= 0.25,
        "G3 ch5 >= log(0.5)": pl.col("ch5") >= np.log(0.5),
        f"G4 dep1_qv >= {d33:.4f}": pl.col("dep1_qv") >= d33,
    }


def oracle_l(x: pl.DataFrame) -> list[str]:
    """Look-ahead diagnostic: first 10 L candidates of a day vs the 10 best/worst by a feature."""
    lx = x.filter((pl.col("sleeve") == "L") & pl.col("has_depth"))
    lines = [
        "| feature | days | first-10 bp | 10 lowest bp | 10 highest bp | all bp |",
        "|---|---|---|---|---|---|",
    ]
    days = lx.group_by("day").len().filter(pl.col("len") >= 20)["day"]
    z = lx.filter(pl.col("day").is_in(days.implode()))
    first = z.sort("t_in", "prio", descending=[False, True]).group_by("day").head(10)["net"].mean()
    for f in ("bid2_qv", "imb1", "dep1_qv", "ask2_qv", "ch5", "bid2"):
        zz = z.drop_nulls(f)
        lo = zz.sort(f).group_by("day").head(10)["net"].mean()
        hi = zz.sort(f, descending=True).group_by("day").head(10)["net"].mean()
        lines.append(
            f"| {f} | {days.len()} | {num(first) * 1e4:+.0f} | {num(lo) * 1e4:+.0f} | {num(hi) * 1e4:+.0f} | {num(zz['net'].mean()) * 1e4:+.0f} |"  # noqa: E501
        )
    return lines


def main() -> None:
    x = features()
    L: list[str] = [
        "# X12: perp order-book depth at entry vs the four sleeves (2023-26, in-sample)",
        "",
    ]
    L += [
        "See the docstring of `x12_book_depth.py` for the pre-declared features, rules and gates.",
        "",
    ]
    L += [
        "## Coverage (candidates with a depth snapshot <= 15 min before entry)",
        "",
        "| sleeve | candidates | with perp | with depth | with 24h-earlier depth | 2023 | 2024 | 2025 | 2026 |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for sl in "LFMC":
        s = x.filter(pl.col("sleeve") == sl)
        yrs = " | ".join(
            f"{int(g['has_depth'].sum())}/{g.height}"
            for y in range(2023, 2027)
            for g in [s.filter(pl.col("year") == y)]
        )
        L.append(
            f"| {sl} | {s.height} | {int(s['perp'].is_not_null().sum())} | {int(s['has_depth'].sum())} | {int(s['ch5'].is_not_null().sum())} | {yrs} |"  # noqa: E501
        )
    L += ["", "Mean net of candidates with vs without depth (bp):", ""]
    for sl in "LFMC":
        s = x.filter(pl.col("sleeve") == sl)
        w, wo = (
            s.filter(pl.col("has_depth"))["net"].mean(),
            s.filter(~pl.col("has_depth"))["net"].mean(),
        )
        L.append(
            f"- {sl}: with {num(w or 0) * 1e4:+.0f} (n={int(s['has_depth'].sum())}), without {num(wo or 0) * 1e4:+.0f} (n={int((~s['has_depth']).sum())})"  # noqa: E501
        )
    for sl in "LFMC":
        s = x.filter((pl.col("sleeve") == sl) & pl.col("has_depth"))
        L += [
            "",
            f"## Sleeve {sl}: feature terciles (n={s.height}; mean net bp; terciles within each period)",  # noqa: E501
            "",
        ]
        L += tercile_table(s, sl)
        print("\n".join(L[-len(FEATS) - 2 :]), flush=True)
    sh = x.filter(pl.col("sleeve").is_in(["F", "M", "C"]) & pl.col("has_depth"))
    L += ["", f"## Pooled shorts F+M+C (n={sh.height})", ""]
    L += tercile_table(sh, "S")
    L += ["", "## L within-day look-ahead diagnostic (days with >= 20 L candidates with depth)", ""]
    L += oracle_l(x)
    # ---- short gates
    L += [
        "",
        "## Short gates (applied to F, M, C at once; missing depth passes)",
        "",
        "| gate | sleeve | n kept | n dropped | E net kept bp | E net dropped bp | stopped % kept | stopped % dropped | squeeze % kept | squeeze % dropped |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    gates = short_gates(x)
    for name, g in gates.items():
        if name == "base":
            continue
        for sl in ("F", "M", "C"):
            s = x.filter(pl.col("sleeve") == sl).with_columns(keep=g.fill_null(True))
            k, dr = s.filter(pl.col("keep")), s.filter(~pl.col("keep"))

            def mm(d: pl.DataFrame, c: str, sc: float) -> str:
                return "na" if d.height == 0 else f"{num(d[c].cast(pl.Float64).mean()) * sc:+.0f}"

            L.append(
                f"| {name} | {sl} | {k.height} | {dr.height} | {mm(k, 'net', 1e4)} | {mm(dr, 'net', 1e4)} | "  # noqa: E501
                f"{mm(k, 'stopped', 100)} | {mm(dr, 'stopped', 100)} | {mm(k, 'squeeze', 100)} | {mm(dr, 'squeeze', 100)} |"  # noqa: E501
            )
    # ---- books
    L += [
        "",
        "## 10-slot books, $10k a year, 2023-26 (book() from e25)",
        "",
        "| book | rule | trades | bp/trade | L trades | L bp | min year % | worst DD % | years 23/24/25/26 % |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rules = l_rules(x)
    for combo in ("LMC", "LMCF"):
        for name, t in rules.items():
            r = run_book(t.filter(pl.col("sleeve").is_in(list(combo))))
            row = (
                f"| {combo} | {name} | {r['n']} | {r['bp']:+.0f} | {r['nL']} | {r['bpL']:+.0f} | {r['min'] * 100:+.0f} | "  # noqa: E501
                f"{r['dd'] * 100:+.0f} | {r['years']} |"
            )
            L.append(row)
            print(row, flush=True)
        base = rules["base"]
        for name, g in gates.items():
            if name == "base":
                continue
            t = base.with_columns(
                w=pl.when(pl.col("sleeve").is_in(["F", "M", "C"]) & ~g.fill_null(True))
                .then(0.0)
                .otherwise(pl.col("w"))
            )
            r = run_book(t.filter(pl.col("sleeve").is_in(list(combo))))
            row = (
                f"| {combo} | {name} | {r['n']} | {r['bp']:+.0f} | {r['nL']} | {r['bpL']:+.0f} | {r['min'] * 100:+.0f} | "  # noqa: E501
                f"{r['dd'] * 100:+.0f} | {r['years']} |"
            )
            L.append(row)
            print(row, flush=True)
    (OUT / "book_depth_tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
