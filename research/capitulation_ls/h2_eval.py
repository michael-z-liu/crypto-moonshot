"""H2 (holdout, BTC pairs): the pre-registered long sleeve L, unchanged, on the USDT-converted BTC
pairs of 2017-08 .. 2019-12 (h2_build.py, h2_score.py), with h1_eval.py's functions:

- L: pct >= 0.995, entry at the next 5m open, take profit +20%, time exit 72h, one position per
  coin (`h1_eval.trades`, which runs `sim.simulate` at 0.4% round trip);
- costs: 0.4% (as pre-registered) and a wide scenario: 1.0% round trip on the alt/BTC pair plus
  the BTCUSDT legs (0.1% each side), 1.2% in all, for BTC-pair trades (1.0% for h1's USDT trades
  in the combined numbers); costs are subtracted from the 0.4% net (one position per coin and the
  exits do not depend on cost);
- calibration: per-spike fixed pct bins and the diag deciles (pool pct >= 0.95);
- combined holdout: h1's USDT-pair L trades (day5, holdout/l_trades.parquet) plus these, with the
  day-bootstrap CI (`h1_eval.clustered`) and the 10-slot book (`h1_eval.book_table`).

The simulator reads holdout/btc by swapping sim's `meta` / `fwd2` / `fwd72` loaders.
Writes holdout/btc/{eval_tables.md, l_trades.parquet} and prints the tables.
Run: uv run --group research python h2_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import sim
from diag import annotate, calibration, scorecard, table, threshold_reach
from h1_eval import (
    BFMT,
    HY,
    YFMT,
    baselines,
    book_table,
    by_year,
    clustered,
    diag_frame,
    fixed_bins,
    md,
    per_spike,
    trades,
)

HOLDOUT = Path("/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/holdout")
OUT = HOLDOUT / "btc"
WIDE_BTC, WIDE_USDT = 0.012 - 0.004, 0.010 - 0.004  # extra cost over the 0.4% net
out: list[str] = []


def emit(s: str = "") -> None:
    out.append(s)
    print(s)


def use_btc() -> pl.DataFrame:
    m = pl.read_parquet(OUT / "seq2_meta.parquet").with_columns(
        year=pl.col("open_time").dt.year(), day=pl.col("open_time").dt.date()
    )
    f2 = np.load(OUT / "fwd2.npy", mmap_mode="r")
    f72 = np.load(OUT / "fwd72.npy", mmap_mode="r")
    sim.meta, sim.fwd2, sim.fwd72 = (lambda: m), (lambda: f2), (lambda: f72)  # type: ignore[assignment]
    return m


def shifted(t: pl.DataFrame, extra: float) -> pl.DataFrame:
    return t.with_columns(net=pl.col("net") - extra)


def year_table(t: pl.DataFrame, usdt_days: set[object]) -> pl.DataFrame:
    """by_year at 0.4% plus the wide-cost mean and the flush days not in the USDT holdout."""
    y = by_year(t)
    extra = (
        t.group_by("year")
        .agg(
            mean_bp_1p0=(pl.col("net") - 0.006).mean() * 1e4,
            mean_bp_1p2=(pl.col("net") - WIDE_BTC).mean() * 1e4,
            new_days=pl.col("day").filter(~pl.col("day").is_in(list(usdt_days))).n_unique(),
        )
        .sort("year")
    )
    return y.join(extra, on="year")


def combined(h1: pl.DataFrame, bt: pl.DataFrame) -> pl.DataFrame:
    cols = ["symbol", "row", "t", "t_out", "net", "adv", "mfe", "kind", "qv", "signal", "year",
            "day", "exit_kind"]  # fmt: skip
    a = h1.select(cols).with_columns(src=pl.lit("USDT"))
    b = bt.select(cols).with_columns(src=pl.lit("BTC"))
    return pl.concat([a, b], how="vertical_relaxed").sort("t")


def wide(c: pl.DataFrame) -> pl.DataFrame:
    return c.with_columns(
        net=pl.col("net") - pl.when(pl.col("src") == "BTC").then(WIDE_BTC).otherwise(WIDE_USDT)
    )


def main() -> None:
    pl.Config.set_tbl_rows(200)
    m = use_btc()
    sc = pl.read_parquet(OUT / "scores.parquet")
    h1 = pl.read_parquet(HOLDOUT / "l_trades.parquet").filter(pl.col("model") == "day5")
    usdt_days = set(h1["day"].to_list())
    yr = m["year"].to_numpy()
    hy = np.isin(yr, HY)

    emit("## Coverage\n")
    plan = json.loads((HOLDOUT / "plan_btc.json").read_text())
    n_store = len(list((HOLDOUT / "klines_btc/klines/spot/interval=5m").glob("symbol=*")))
    emit(f"- BTC pairs planned {len(plan)}, symbol-months planned {sum(map(len, plan.values()))},"
         f" pairs stored {n_store}")  # fmt: skip
    cov = (
        m.filter(pl.col("year").is_in(HY))
        .group_by("year")
        .agg(spikes=pl.len(), coins=pl.col("symbol").n_unique(),
             med_qv_base=pl.col("qv_base").median(),
             first=pl.col("open_time").min(), last=pl.col("open_time").max())
        .sort("year")
    )  # fmt: skip
    emit(md(cov, {"med_qv_base": ".0f"}))
    emit("")
    emit("## Baselines (L's exit, one position per coin, 0.4%)\n")
    emit("\n".join(baselines(m, HY, "BTC pairs 2017-19")) + "\n")

    models = [c.removeprefix("pct_") for c in sc.columns if c.startswith("pct_")]
    all_l: dict[str, pl.DataFrame] = {}
    for name in models:
        pct = sc[f"pct_{name}"].to_numpy()
        emit(f"## Model {name}\n")
        sh = (
            pl.DataFrame({"year": yr[hy], "pct": pct[hy]})
            .group_by("year")
            .agg(n=pl.len(), ge_095=(pl.col("pct") >= 0.95).mean(),
                 ge_099=(pl.col("pct") >= 0.99).mean(), ge_0995=(pl.col("pct") >= 0.995).mean(),
                 ge_0998=(pl.col("pct") >= 0.998).mean())
            .sort("year")
        )  # fmt: skip
        emit("Share of liquid spikes at each pct threshold:\n")
        emit(md(sh, {k: ".4f" for k in sh.columns if k.startswith("ge")}))
        emit("")
        lt = trades(m, pct, HY, 0.995)
        all_l[name] = lt
        emit("### L trades by year (pct >= 0.995, TP 20%, 72h; mean_bp at 0.4%)\n")
        if lt.height == 0:
            emit("no trades\n")
            continue
        yfmt = {**YFMT, "mean_bp_1p0": "+.0f", "mean_bp_1p2": "+.0f"}
        emit(md(year_table(lt, usdt_days), yfmt))
        emit("")
        ex = lt.group_by("exit_kind").agg(n=pl.len(), mean_bp=pl.col("net").mean() * 1e4)
        emit(
            f"All years: n {lt.height}, mean {lt['net'].mean() * 1e4:+.0f} bp (1.0%:"
            f" {(lt['net'].mean() - 0.006) * 1e4:+.0f}, 1.2%: {(lt['net'].mean() - WIDE_BTC) * 1e4:+.0f}),"  # noqa: E501
            f" median {lt['net'].median() * 1e4:+.0f} bp, TP rate {(lt['kind'] == 2).mean():.2f},"
            f" win {(lt['net'] > 0).mean():.2f}, days {lt['day'].n_unique()}, new days"
            f" {len(set(lt['day'].to_list()) - usdt_days)}; exits "
            + ", ".join(
                f"{r['exit_kind']} {r['n']} ({r['mean_bp']:+.0f} bp)"
                for r in ex.iter_rows(named=True)
            )
            + "\n"
        )
        emit(clustered(lt, f"BTC pairs 0.4% ({name})"))
        emit(clustered(shifted(lt, WIDE_BTC), f"BTC pairs 1.2% ({name})") + "\n")
        emit("### 10-slot book, BTC pairs alone ($10k fresh each year)\n")
        emit("0.4%:\n")
        emit(md(book_table(lt), BFMT))
        emit("\n1.2% (1.0% alt leg + BTCUSDT legs):\n")
        emit(md(book_table(shifted(lt, WIDE_BTC)), BFMT))
        emit("")
        pool = trades(m, pct, HY, 0.95, 0.995)
        fe, fn = diag_frame(lt), diag_frame(pool) if pool.height else None
        emit("### diag scorecard (0.4%)\n")
        emit(table({f"L BTC pairs ({name})": scorecard(annotate(fe))}))
        tr = threshold_reach(fe, fn)
        emit(f"threshold {tr['threshold']:.3f}: {tr['share_reaching']:.1%} of the pool reach it;"
             f" Spearman(signal, outcome) {tr['spearman_signal_outcome']:+.3f}\n")  # fmt: skip
        emit("### calibration by signal decile (pool pct >= 0.95, 0.4%)\n")
        emit(md(calibration(fe, fn), {"bin": ".4f", "E_o": "+.3f"}))
        emit("")
        ps = per_spike(m, pct, HY, 0.9)
        emit("Per-spike outcome by fixed pct bins (no per-coin lock, 0.4%):\n")
        emit(md(fixed_bins(ps, int((hy & np.isfinite(pct)).sum())),
                {"mean_bp": "+.0f", "tp_rate": ".2f", "share": ".4f"}))  # fmt: skip
        emit("")
        rows = []
        for q in (0.99, 0.995, 0.998):
            for tp in (0.15, 0.2, 0.25):
                t = trades(m, pct, HY, q, tp=tp)
                rows.append({
                    "rule": f"pct>={q} TP{int(tp * 100)}" + (" (TEST)" if (q, tp) == (0.995, 0.2) else ""),  # noqa: E501
                    "n": t.height,
                    "mean_bp": float(t["net"].mean() or 0) * 1e4 if t.height else 0.0,
                    "tp_rate": float((t["kind"] == 2).mean()) if t.height else 0.0,
                    "days": t["day"].n_unique() if t.height else 0,
                })  # fmt: skip
        emit("### neighbours (0.4%)\n")
        emit(md(pl.DataFrame(rows), {"mean_bp": "+.0f", "tp_rate": ".2f"}))
        emit("")
        fd = (
            lt.join(m.with_row_index("row").select(pl.col("row").cast(pl.Int64), "btc_ret_24h",
                    "ret_24h", "qv_base"), on="row", how="left")
            .group_by("day")
            .agg(n=pl.len(), coins=pl.col("symbol").unique().sort().str.join(","),
                 mean_bp=pl.col("net").mean() * 1e4, tp=(pl.col("kind") == 2).sum(),
                 coin_ret_24h=pl.col("ret_24h").median(), btc_ret_24h=pl.col("btc_ret_24h").first(),
                 med_qv_day=(pl.col("qv_base") * 288).median(), pct_max=pl.col("signal").max())
            .with_columns(in_usdt_holdout=pl.col("day").is_in(list(usdt_days)))
            .sort("day")
        )  # fmt: skip
        emit("### flush days\n")
        emit(md(fd, {"mean_bp": "+.0f", "coin_ret_24h": "+.3f", "btc_ret_24h": "+.3f",
                     "med_qv_day": ".0f", "pct_max": ".4f"}))  # fmt: skip
        emit("")
        if name in ("day5", "day5_2022"):
            h1m = pl.read_parquet(HOLDOUT / "l_trades.parquet").filter(pl.col("model") == name)
            c = combined(h1m, lt)
            emit(f"### Combined holdout: USDT pairs (h1) + BTC pairs ({name})\n")
            emit(md(by_year(c), YFMT))
            emit("")
            emit(clustered(c, f"combined 0.4% ({name})"))
            emit(clustered(wide(c), f"combined wide cost, USDT 1.0% / BTC 1.2% ({name})") + "\n")
            emit("10-slot book, combined, 0.4%:\n")
            emit(md(book_table(c), BFMT))
            emit("\n10-slot book, combined, USDT 1.0% / BTC 1.2%:\n")
            emit(md(book_table(wide(c)), BFMT))
            emit("")
    pl.concat(
        [t.with_columns(model=pl.lit(k)) for k, t in all_l.items() if t.height],
        how="diagonal_relaxed",
    ).write_parquet(OUT / "l_trades.parquet")
    (OUT / "eval_tables.md").write_text("\n".join(out) + "\n")
    print("written", OUT / "eval_tables.md", file=sys.stderr)


if __name__ == "__main__":
    main()
