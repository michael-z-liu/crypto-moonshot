"""Common numerical diagnostics for every strategy (user request 2026-11-27 23:20Z).

Every trade is put on the same scales:

- signal s in [0, 1]: the entry signal's strength (a percentile within the candidate pool), with
  the strategy's entry threshold theta on the same scale;
- outcome o in [-1, +1]: the net return on the stake after costs, clipped (-1 = stake lost);
- tp_progress in [0, 1]: how far the best price move during the hold got towards the take profit
  (mfe / tp; strategies without a take profit use REF_TP = 20%);
- stop_progress in [0, 1]: how far the worst move got towards the stop or liquidation
  (mae / stop; spot longs and shorts without a stop use 1.0 = a 100% move);
- giveback in [0, 1]: the share of the best move given back by the exit.

Each trade gets one failure mode (first match): "stopped" (stop/liquidation or o <= -0.95),
"tp_win" (exited at the take profit), "time_win" (o > 0 otherwise), "near_miss" (o <= 0 but
tp_progress >= 0.5), "never_went" (o <= 0, tp_progress < 0.5; split by stop_progress >= 0.5 into
"went_against"). The expectancy E[o] is decomposed into the modes' contributions (share x mean
outcome). With candidates that did not reach the threshold, `calibration` shows the outcome by
signal decile and `threshold_reach` how often the signal got close.

Trade frame columns: signal, threshold, net, mfe, mae, tp (null = none), stop (null = 1.0),
exit_kind ("time"/"tp"/"stop"/"liq"), year; optional strategy, config.
"""

from __future__ import annotations

import numpy as np
import polars as pl

REF_TP = 0.20
MODES = ("tp_win", "time_win", "near_miss", "never_went", "went_against", "stopped")


def annotate(t: pl.DataFrame) -> pl.DataFrame:
    """Add the common scales and the failure mode to a trade frame."""
    tp = pl.col("tp").fill_null(REF_TP).cast(pl.Float64)
    stop = pl.col("stop").fill_null(1.0).cast(pl.Float64)
    t = t.with_columns(
        o=pl.col("net").clip(-1.0, 1.0),
        tp_progress=(pl.col("mfe") / tp).clip(0.0, 1.0),
        stop_progress=(pl.col("mae") / stop).clip(0.0, 1.0),
        giveback=pl.when(pl.col("mfe") > 0)
        .then(((pl.col("mfe") - pl.col("net").clip(-1.0, None)) / pl.col("mfe")).clip(0.0, 1.0))
        .otherwise(0.0),
    )
    mode = (
        pl.when(pl.col("exit_kind").is_in(["stop", "liq"]) | (pl.col("o") <= -0.95))
        .then(pl.lit("stopped"))
        .when(pl.col("exit_kind") == "tp")
        .then(pl.lit("tp_win"))
        .when(pl.col("o") > 0)
        .then(pl.lit("time_win"))
        .when(pl.col("tp_progress") >= 0.5)
        .then(pl.lit("near_miss"))
        .when(pl.col("stop_progress") >= 0.5)
        .then(pl.lit("went_against"))
        .otherwise(pl.lit("never_went"))
    )
    return t.with_columns(mode=mode)


def scorecard(t: pl.DataFrame) -> dict[str, float]:
    """One row of numbers per strategy (trades must be annotated or raw)."""
    if "mode" not in t.columns:
        t = annotate(t)
    n = t.height
    out: dict[str, float] = {
        "n": n,
        "E_o": float(t["o"].mean()),
        "median_o": float(t["o"].median()),
        "win": float((t["o"] > 0).mean()),
        "tp_progress_med": float(t["tp_progress"].median()),
        "stop_progress_med": float(t["stop_progress"].median()),
        "giveback_med": float(t["giveback"].median()),
        "signal_med": float(t["signal"].median()),
    }
    for m in MODES:
        sub = t.filter(pl.col("mode") == m)
        out[f"share_{m}"] = sub.height / n if n else 0.0
        out[f"contrib_{m}"] = float(sub["o"].sum()) / n if n else 0.0  # sums to E_o
    yr = t.group_by("year").agg(e=pl.col("o").mean()).sort("year")
    out["worst_year_E_o"] = float(yr["e"].min())
    out["years_positive"] = float((yr["e"] > 0).mean())
    return out


def calibration(
    entries: pl.DataFrame, nonentries: pl.DataFrame | None, bins: int = 10
) -> pl.DataFrame:
    """Outcome by signal decile over entries and non-entries (as if entered)."""
    frames = [annotate(entries).with_columns(entered=pl.lit(True))]
    if nonentries is not None and nonentries.height:
        frames.append(annotate(nonentries).with_columns(entered=pl.lit(False)))
    a = pl.concat(
        [f.select("signal", "o", "mode", "tp_progress", "stop_progress", "entered") for f in frames]
    )
    a = a.with_columns(bin=(pl.col("signal").clip(0, 0.999999) * bins).floor() / bins)
    return (
        a.group_by("bin")
        .agg(
            n=pl.len(),
            E_o=pl.col("o").mean(),
            win=(pl.col("o") > 0).mean(),
            p_tp=(pl.col("mode") == "tp_win").mean(),
            p_stopped=(pl.col("mode") == "stopped").mean(),
            tp_progress=pl.col("tp_progress").median(),
            entered=pl.col("entered").mean(),
        )
        .sort("bin")
    )


def threshold_reach(entries: pl.DataFrame, nonentries: pl.DataFrame | None) -> dict[str, float]:
    """How often candidates reach the threshold, and how many fall just short (within 0.1)."""
    th = float(entries["threshold"].median())
    sig = entries["signal"].to_numpy()
    if nonentries is not None and nonentries.height:
        sig = np.concatenate([sig, nonentries["signal"].to_numpy()])
    return {
        "threshold": th,
        "share_reaching": float(np.mean(sig >= th)),
        "share_within_0.1_below": float(np.mean((sig < th) & (sig >= th - 0.1))),
        "spearman_signal_outcome": _spearman(entries, nonentries),
    }


def _spearman(entries: pl.DataFrame, nonentries: pl.DataFrame | None) -> float:
    a = annotate(entries).select("signal", "o")
    if nonentries is not None and nonentries.height:
        a = pl.concat([a, annotate(nonentries).select("signal", "o")])
    return float(a.select(pl.corr(pl.col("signal").rank(), pl.col("o").rank())).item())


def table(cards: dict[str, dict[str, float]]) -> str:
    """Markdown table of scorecards (one row per strategy)."""
    cols = ["n", "E_o", "median_o", "win", "share_tp_win", "share_time_win", "share_near_miss",
            "share_never_went", "share_went_against", "share_stopped", "contrib_stopped",
            "tp_progress_med", "stop_progress_med", "giveback_med", "worst_year_E_o", "years_positive"]  # fmt: skip  # noqa: E501
    head = "| strategy | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    rows = []
    for name, c in cards.items():
        vals = [
            f"{c[k]:.0f}"
            if k == "n"
            else f"{c[k]:+.3f}"
            if k.startswith(("E_", "median", "contrib", "worst"))
            else f"{c[k]:.2f}"
            for k in cols
        ]
        rows.append(f"| {name} | " + " | ".join(vals) + " |")
    return head + "\n".join(rows) + "\n"
