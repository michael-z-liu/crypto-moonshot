"""Summary charts for the write-up (reads the cached tables and results/)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import RESULTS
from evalx import COST, load_events
from m3_rules import non_overlapping


def deciles() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, feat, title in zip(
        axes,
        ["ret_4h", "ret_bar"],
        ["Coin's return over the previous 4h", "Spike bar's own return"],
        strict=True,
    ):
        for group, color in (("spike", "#c0392b"), ("random", "#7f8c8d")):
            d = pl.read_csv(RESULTS / f"m3_deciles_{group}.csv").filter(pl.col("feature") == feat)
            ax.plot(
                d["q"].cast(pl.Int32) + 1,
                d["f4h"],
                marker="o",
                color=color,
                label="volume spike (>=10x)" if group == "spike" else "random 5m bar",
            )
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(30, color="k", lw=0.8, ls=":", label="round-trip cost (30 bp)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("decile (1 = most negative, 10 = most positive)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean gross return over next 4h (bp)")
    axes[0].legend(fontsize=8)
    fig.suptitle("2020-23: spikes after a crash bounce, spikes after a pump fade", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "deciles.png", dpi=130)


def yearly() -> None:
    ev, fwd = load_events()
    rules = {
        "F2 deep liquid flush (1h)": (
            (pl.col("ret_4h") <= -0.10)
            & (pl.col("qv_base") >= 20_000)
            & (pl.col("btc_vol_mult") >= 6),
            12,
        ),
        "F3 deep crash, any market (4h)": (
            (pl.col("ret_4h") <= -0.15) & (pl.col("qv_base") >= 5_000),
            48,
        ),
        "User 2019 rule (4h)": (
            (pl.col("ret_bar") > 0) & (pl.col("tbr") > 0.6) & (pl.col("runup_1h").abs() < 0.01),
            48,
        ),
    }
    fig, ax = plt.subplots(figsize=(10, 4))
    years = np.arange(2020, 2027)
    w = 0.27
    for i, (name, (cond, h)) in enumerate(rules.items()):
        idx = np.flatnonzero(
            ev.select((pl.col("is_spike") & cond).fill_null(False))["is_spike"].to_numpy()
        )
        d = (
            ev[idx]
            .select("symbol", "open_time")
            .with_columns(r=pl.Series(np.asarray(fwd[idx, 2, h - 1])) - COST)
        )
        d = d.filter(
            pl.Series(non_overlapping(d["symbol"].to_numpy(), d["open_time"].to_numpy(), h))
        )
        days = d.group_by(pl.col("open_time").dt.date().alias("day")).agg(pl.col("r").mean())
        y = (
            days.group_by(pl.col("day").dt.year().alias("y"))
            .agg(pl.col("r").mean() * 1e4)
            .sort("y")
        )
        vals = [y.filter(pl.col("y") == yr)["r"].sum() for yr in years]
        ax.bar(years + (i - 1) * w, vals, w, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvspan(2024.5, 2026.6, color="#fde2e2", zorder=0, label="test period")
    ax.set_ylabel("mean net return per signal day (bp)")
    ax.set_title("Net return per signal day (equal-weight basket of that day's signals), 0.3% cost")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS / "yearly.png", dpi=130)


if __name__ == "__main__":
    deciles()
    yearly()
