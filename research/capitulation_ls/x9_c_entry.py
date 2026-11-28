"""X9: entry features for sleeve C, idiosyncratic crash shorts (in-sample exploration).

why/c_exits.md showed that C cannot be improved by exits, only by entry or sizing. This script
takes every C candidate (2022-26, no slot limit), joins the spike and futures-metric features,
and reports for each feature the mean net by tercile in 2022-24 and 2025-26, the raw and
within-day rank IC, and a few pre-declared gates in the 10-slot LMC book.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from common_ls import OWN
from diag import annotate, scorecard
from e25_combined import book, scores, sleeve_listing_weak, sleeve_long
from sim import Rule, simulate

OUT = Path(__file__).resolve().parent / "results" / "explore"
FEATS = ["ret_4h", "ret_24h", "r1h", "r15m", "vol_mult", "off_low4h", "off_high24h", "dlo30d",
         "dhi30d", "r7d", "r30d", "vol30d", "qv7_30", "listed_days", "taker1h", "wick_dn",
         "n_prior_pumps30", "rate", "rate24", "mkt_r4h", "mkt_r24h", "btc_r24h", "spike_share",
         "oi_ch1h", "oi_ch4h", "oi_ch24h", "oi_to_vol", "ls_acc", "top_ls_pos", "taker_ls_1h",
         "cnn"]  # fmt: skip


def candidates() -> pl.DataFrame:
    d = pl.read_parquet(OWN / "feats.parquet").filter(
        pl.col("perp")
        & (pl.col("qv_base") >= 20_000)
        & (pl.col("open_time").dt.year() >= 2022)
        & (pl.col("ret_4h") <= -0.15)
        & (pl.col("btc_ret_4h").abs() <= 0.01)
    )
    pct = scores("day5")
    d = d.with_columns(cnn=pl.Series(pct[d["row"].to_numpy()])).filter(pl.col("cnn") <= 0.5)
    b = simulate(
        d.select("row", side=pl.lit(-1), prio=-pl.col("ret_4h")),
        Rule(hold_min=1440, slots=100_000, cost=0.003, sl=1.0, liq_frac=None),
        start_equity=1e12,
        perp_funding=True,
        source="perp",
    )
    t = b.trades.select("row", "symbol", "adv", "mfe", "net", "t", "t_out", "kind").with_columns(
        exit=pl.col("kind").replace_strict({0: "time", 1: "stop", 2: "tp", 3: "liq", 4: "trail"})
    )
    mf = pl.read_parquet(OWN / "metric_feats.parquet")
    return (
        t.join(d, on=["row", "symbol"], how="left", suffix="_f")
        .join(mf, on="row", how="left")
        .with_columns(day=pl.col("open_time").dt.date(), yr=pl.col("open_time").dt.year())
    )


def ic_within_day(x: pl.DataFrame, f: str) -> float:
    z = x.drop_nulls(f).with_columns(
        rf=(pl.col(f).rank() / pl.len()).over("day"),
        rn=(pl.col("net").rank() / pl.len()).over("day"),
    )
    z = z.filter(pl.len().over("day") >= 3)
    if z.height < 30:
        return float("nan")
    return float(z.select(pl.corr(pl.col("rf") - pl.col("rf").mean().over("day"),
                                  pl.col("rn") - pl.col("rn").mean().over("day"))).item())  # fmt: skip  # noqa: E501


def terciles(x: pl.DataFrame, f: str) -> list[float]:
    z = x.drop_nulls(f).filter(pl.col(f).is_not_nan())
    if z.height < 30:
        return [float("nan")] * 3
    q1, q2 = z[f].quantile(1 / 3), z[f].quantile(2 / 3)
    out = []
    for lo, hi in ((-np.inf, q1), (q1, q2), (q2, np.inf)):
        s = z.filter((pl.col(f) > lo) & (pl.col(f) <= hi))
        out.append(float(s["net"].mean()) * 1e4 if s.height else float("nan"))
    return out


def lmc_with(c: pl.DataFrame, lg: pl.DataFrame, mm: pl.DataFrame) -> tuple[float, float, str, int]:
    cols = ["symbol", "adv", "t_in", "t_out", "net", "qv", "sleeve", "prio"]
    cc = c.select(
        "symbol", "adv", t_in=pl.col("t") + 300, t_out="t_out", net="net",
        qv=pl.col("qv_base") * 288, sleeve=pl.lit("C"), prio=pl.lit(1.5),
    )  # fmt: skip
    b = book(
        pl.concat([lg.select(cols), mm.select(cols), cc.select(cols)], how="vertical_relaxed"), 10
    )
    n = int(b["trades"].sum())
    bp = float((b["bp"] * b["trades"]).sum() / n)
    yrs = "/".join(f"{r['ret'] * 100:+.0f}" for r in b.iter_rows(named=True))
    return bp, float(b["ret"].min()), yrs, n


if __name__ == "__main__":
    x = candidates()
    print(
        "C candidates", x.height, "mean net bp", x["net"].mean() * 1e4, "days", x["day"].n_unique()
    )
    a, b_ = x.filter(pl.col("yr") <= 2024), x.filter(pl.col("yr") >= 2025)
    lines = [
        f"# X9: entry features for crash shorts C (candidates n={x.height}, {x['day'].n_unique()} days, "  # noqa: E501
        f"mean {x['net'].mean() * 1e4:+.0f} bp; 2022-24 n={a.height}, 2025-26 n={b_.height})",
        "",
        "Mean net (bp) by feature tercile (low/mid/high), within each period; IC = Spearman rank IC",  # noqa: E501
        "with net over all candidates; IC_day = within-day IC (days with >= 3 candidates).",
        "",
        "| feature | missing | 22-24 low/mid/high | 25-26 low/mid/high | IC | IC_day |",
        "|---|---|---|---|---|---|",
    ]
    for f in FEATS:
        if f not in x.columns:
            continue
        miss = float(x[f].is_null().mean())
        ta, tb = terciles(a, f), terciles(b_, f)
        z = x.drop_nulls(f)
        ic = float(z.select(pl.corr(pl.col(f).rank(), pl.col("net").rank())).item())
        row = (f"| {f} | {miss:.0%} | {ta[0]:+.0f}/{ta[1]:+.0f}/{ta[2]:+.0f} | "
               f"{tb[0]:+.0f}/{tb[1]:+.0f}/{tb[2]:+.0f} | {ic:+.3f} | {ic_within_day(x, f):+.3f} |")  # fmt: skip  # noqa: E501
        lines.append(row)
        print(row, flush=True)
    # Pre-declared gates (written before the tercile table was read): the crash is fresh and
    # not yet bounced, the coin was already weak, and shorts are not crowded.
    gates = {
        "base": pl.lit(True),
        "off_low4h <= 3% (no bounce yet)": pl.col("off_low4h") <= 0.03,
        "r30d <= 0 (already weak)": pl.col("r30d") <= 0,
        "dhi30d <= -40% (deep under 30d high)": pl.col("dhi30d") <= -0.40,
        "rate >= 0 (shorts not paid)": pl.col("rate") >= 0,
        "ls_acc >= 1 (shorts not crowded)": pl.col("ls_acc") >= 1.0,
        "oi_ch4h >= 0 (OI rising into crash)": pl.col("oi_ch4h") >= 0,
        "vol_mult top half": pl.col("vol_mult") >= pl.col("vol_mult").median(),
        "listed_days >= 365": pl.col("listed_days") >= 365,
    }
    lg = sleeve_long()
    mm = sleeve_listing_weak()
    lines += ["", "## Pre-declared gates (C candidates, then the LMC book with C gated)", "",
              "| gate | n | E_o | win | share_stopped | giveback_med | 22-24 bp | 25-26 bp | LMC bp | LMC trades | LMC min yr | LMC years |",  # noqa: E501
              "|" + "---|" * 12]  # fmt: skip
    for name, g in gates.items():
        s = x.filter(g.fill_null(False)) if name != "base" else x
        f = s.select(
            signal=pl.lit(1.0), threshold=pl.lit(0.0), net="net", mfe="mfe", mae="adv",
            tp=pl.lit(None, dtype=pl.Float64), stop=pl.lit(1.0), exit_kind="exit", year="yr",
        )  # fmt: skip
        sc = scorecard(annotate(f))
        pa = s.filter(pl.col("yr") <= 2024)["net"].mean()
        pb = s.filter(pl.col("yr") >= 2025)["net"].mean()
        bp, mn, yrs, n = lmc_with(s, lg, mm)
        row = (f"| {name} | {s.height} | {sc['E_o']:+.3f} | {sc['win']:.2f} | {sc['share_stopped']:.2f} | "  # noqa: E501
               f"{sc['giveback_med']:.2f} | {(pa or 0) * 1e4:+.0f} | {(pb or 0) * 1e4:+.0f} | {bp:+.0f} | {n} | "  # noqa: E501
               f"{mn * 100:+.0f} | {yrs} |")  # fmt: skip
        lines.append(row)
        print(row, flush=True)
    (OUT / "c_entry.md").write_text("\n".join(lines) + "\n")
