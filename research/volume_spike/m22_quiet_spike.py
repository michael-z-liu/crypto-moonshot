"""The spike that ends a quiet spell (B3 round 2). Protocol: docs/plans/cnn-replication.md.

Every 10x volume spike (cache_all/events.parquet, >= 1k USDT median per 5m candle) gets the
quietness of its coin from the latest 4-hourly snapshot that closed before the spike candle (at
most 4 hours earlier), so the spike itself is not part of the quiet measure. Entry at the next 5m
open; outcomes from long30.parquet; the $50 no-stop book of m18_longshot.

Run: MOONSHOT_UNIVERSE=all uv run python research/volume_spike/m22_quiet_spike.py
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl
from common import CACHE, RESULTS
from m18_longshot import LAST_ENTRY, run_book, show
from m19_quiet import EX_CASCADE
from m19_quiet import load as load_quiet


def spikes() -> pl.DataFrame:
    cols = [
        "row",
        "symbol",
        "open_time",
        "is_spike",
        "qv_base",
        "ret_bar",
        "tbr",
        "runup_1h",
        "vol_mult",
        "ret_4h",
        "ret_24h",
        "btc_vol_mult",
    ]
    ev = pl.read_parquet(CACHE / "events.parquet", columns=cols).filter(
        pl.col("is_spike") & (pl.col("qv_base") >= 1_000) & (pl.col("open_time") < LAST_ENTRY)
    )
    lo = pl.read_parquet(CACHE / "long30.parquet")
    ev = ev.join(lo, on="row", how="inner")
    fwd = np.load(CACHE / "fwd.npy", mmap_mode="r")
    rows = ev["row"].to_numpy()
    ev = ev.with_columns(
        r1h=pl.Series(np.asarray(fwd[rows, 2, 11], dtype=np.float64)),
        r4h=pl.Series(np.asarray(fwd[rows, 2, 47], dtype=np.float64)),
    )
    q = (
        load_quiet()
        .select("symbol", "open_time", "q1", "q7", "q7_dec", "q1_dec", "days_since_spike")
        .with_columns(snap_end=pl.col("open_time") + timedelta(minutes=5))
        .drop("open_time")
        .sort("snap_end")
    )
    out = ev.sort("open_time").join_asof(
        q,
        left_on="open_time",
        right_on="snap_end",
        by="symbol",
        strategy="backward",
        tolerance=timedelta(hours=4),
    )
    return out.filter(pl.col("q7_dec").is_not_null()).with_columns(
        year=pl.col("open_time").dt.year(), vol_mult_neg=-pl.col("vol_mult")
    )


RULES: dict[str, pl.Expr] = {
    "S1 quiet before + up candle": (pl.col("q7_dec") <= 2) & (pl.col("ret_bar") > 0),
    "S2 quiet + 2019 rule": (pl.col("q7_dec") <= 2)
    & (pl.col("ret_bar") > 0)
    & (pl.col("tbr") > 0.6)
    & (pl.col("runup_1h") <= 0.02),
    "S3 quiet + first spike in 3 days": (pl.col("q7_dec") <= 2)
    & (pl.col("ret_bar") > 0)
    & (pl.col("days_since_spike") >= 3),
    "S4 quiet before, any candle": pl.col("q7_dec") <= 2,
    "B busy before + up candle": (pl.col("q7_dec") >= 5) & (pl.col("ret_bar") > 0),
    "B every up-candle spike": pl.col("ret_bar") > 0,
}


def main() -> None:
    d = spikes()
    print(d.height, "spikes with a quiet measure", flush=True)
    out, ctx = [], []
    for name, mask in RULES.items():
        s = d.filter(mask.fill_null(False))
        print(name, s.height, "signals", flush=True)
        ctx.append(
            s.group_by("year")
            .agg(
                pl.len().alias("signals"),
                (pl.col("r1h").mean() * 100).alias("mean_1h_pct"),
                (pl.col("r4h").mean() * 100).alias("mean_4h_pct"),
                (pl.col("ret_bar").median() * 100).alias("spike_candle_pct"),
            )
            .with_columns(entry=pl.lit(name))
        )
        out += run_book(s, name, "vol_mult_neg", range(2020, 2027))
        dx = s.filter(~pl.col("open_time").is_between(*EX_CASCADE, closed="left"))
        out += run_book(dx, name + " ex-cascade", "vol_mult_neg", range(2025, 2026))
    # secondary: the B2 long-shot CNN scores, only for spikes in a quiet context
    from m12_cnn_wf import WF

    z = np.load(WF / "longshot.npz")
    sc = pl.DataFrame({"row": z["event"], "pct": z["pct"]}).drop_nans()
    qs = d.filter(pl.col("q7_dec") <= 2).join(sc, on="row", how="inner")
    qs = qs.with_columns(neg_pct=-pl.col("pct"))
    for q in (0.9, 0.95):
        out += run_book(
            qs.filter(pl.col("pct") >= q),
            f"CNN (B2) top {(1 - q) * 100:g}%, quiet",
            "neg_pct",
            range(2022, 2027),
        )
    df = pl.DataFrame(out)
    df.write_csv(RESULTS / "m22_quiet_spike.csv")
    c = pl.concat(ctx)
    c.write_csv(RESULTS / "m22_quiet_spike_context.csv")
    with pl.Config(tbl_rows=60, float_precision=2):
        print(c.sort("entry", "year"))
    show(df.filter(~pl.col("entry").str.ends_with("ex-cascade")))


if __name__ == "__main__":
    main()
