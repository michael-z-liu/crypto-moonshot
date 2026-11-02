"""M6: the pre-registered finalists on every split, including the untouched test period."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import RESULTS, TRAIN_END, VALID_END
from evalx import load_events
from portfolio import Exit, simulate

SPIKE = pl.col("is_spike")
FINALISTS: dict[str, tuple[pl.Expr, Exit]] = {
    "F1 flush bounce": (
        SPIKE & (pl.col("ret_4h") <= -0.08) & (pl.col("qv_base") >= 5_000)
        & (pl.col("btc_vol_mult") >= 6), Exit(bars=12),
    ),
    "F2 deep liquid flush": (
        SPIKE & (pl.col("ret_4h") <= -0.10) & (pl.col("qv_base") >= 20_000)
        & (pl.col("btc_vol_mult") >= 6), Exit(bars=12),
    ),
    "F3 deep crash, no BTC filter": (
        SPIKE & (pl.col("ret_4h") <= -0.15) & (pl.col("qv_base") >= 5_000), Exit(bars=48),
    ),
    "ref: user 2019 rule +1%/-5%": (
        SPIKE & (pl.col("ret_bar") > 0) & (pl.col("tbr") > 0.6)
        & (pl.col("runup_1h").abs() < 0.01), Exit(tp=0.01, sl=0.05, bars=48),
    ),
}  # fmt: skip
PERIODS = {
    "train": (None, TRAIN_END),
    "valid": (TRAIN_END, VALID_END),
    "test": (VALID_END, None),
    "all": (None, None),
}


def main() -> None:
    ev, fwd = load_events()
    rows = []
    curves: dict[str, pl.DataFrame] = {}
    for name, (cond, ex) in FINALISTS.items():
        mask = ev.select(cond.fill_null(False).alias("m"))["m"]
        for period, (lo, hi) in PERIODS.items():
            m = mask
            if lo is not None:
                m = m & (ev["open_time"] >= lo)
            if hi is not None:
                m = m & (ev["open_time"] < hi)
            idx = np.flatnonzero(m.to_numpy())
            res = simulate(ev[idx], np.asarray(fwd[idx]), ex, slots=5)
            rows.append({"strategy": name, "period": period, **res.stats()})
            if period == "all":
                curves[name] = res.equity
                yearly = res.trades.group_by(pl.col("entry_time").dt.year().alias("year")).agg(
                    pl.len().alias("trades"), (pl.col("ret").mean() * 1e4).alias("mean_bp"),
                    (pl.col("ret") > 0).mean().alias("win"),
                )  # fmt: skip
                print(name)
                print(yearly.sort("year"))
                if not name.startswith("ref"):
                    res.trades.write_csv(RESULTS / f"trades_{name.split()[0]}.csv")
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m6_test.csv")
    with pl.Config(tbl_rows=30, tbl_cols=12, tbl_width_chars=220, float_precision=3):
        print(df.drop("days_in_market"))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, eq in curves.items():
        if eq.height and not name.startswith("ref"):
            ax.plot(eq["day"], eq["equity"], label=name, lw=1.3)
    ax.set_yscale("log")
    ax.axvspan(TRAIN_END.date(), VALID_END.date(), color="0.92", label="validation (2024)")
    ax.axvspan(VALID_END.date(), curves[next(iter(curves))]["day"].max(), color="#fde2e2",
               label="test (2025-26, untouched)")  # fmt: skip
    ax.set_ylabel("equity (log), start = 1")
    ax.set_title("Capitulation finalists, 5 slots, 0.3% round-trip cost (the 2019 rule goes to ~0)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "equity.png", dpi=130)


if __name__ == "__main__":
    main()
