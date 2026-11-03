"""Build the event table: every 5m volume spike plus a random sample of ordinary bars.

For each selected bar t (a closed candle) it stores features computed from candles up to and
including t, and the forward path of the next HORIZON candles relative to the entry price, which
is the open of candle t+1. Output (in cache/):

- events.parquet: one row per selected bar, features and flags
- fwd.npy: float32 array [rows, 3, HORIZON] of (high, low, close) / entry - 1, same row order
- fwd_long.parquet is folded into events.parquet as close returns at 12h and 24h

Run: uv run python research/volume_spike/build_events.py
"""

from __future__ import annotations

import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import numpy as np
import polars as pl
from common import BARS_PER_DAY, BARS_PER_HOUR, CACHE, HORIZON, load, universe

SPIKE_MULT = 10.0  # volume vs. its median over the previous day (top ~3% of bars)
RANDOM_RATE = 0.005  # share of ordinary bars kept as a baseline sample
WARMUP = BARS_PER_DAY * 7  # skip a symbol's first week


def full_grid(df: pl.DataFrame) -> pl.DataFrame:
    """Reindex to a gap-free 5m grid; missing candles become null rows."""
    grid = pl.DataFrame(
        {
            "open_time": pl.datetime_range(
                df["open_time"][0], df["open_time"][-1], timedelta(minutes=5), eager=True
            )
        }
    )
    return grid.join(df, on="open_time", how="left")


def features(df: pl.DataFrame) -> pl.DataFrame:
    d = BARS_PER_DAY
    h = BARS_PER_HOUR
    v, qv, c = pl.col("volume"), pl.col("quote_volume"), pl.col("close")
    hi, lo, op = pl.col("high"), pl.col("low"), pl.col("open")
    logret = (c / c.shift(1)).log()
    df = df.with_columns(
        vol_base=v.shift(1).rolling_median(d, min_samples=d // 2),
        qv_base=qv.shift(1).rolling_median(d, min_samples=d // 2),
        vol_mean_3h=v.shift(1).rolling_mean(36, min_samples=24),
        tbr=pl.col("taker_buy_base_volume") / v,
        tbr_base=pl.col("taker_buy_base_volume").shift(1).rolling_sum(d, min_samples=d // 2)
        / v.shift(1).rolling_sum(d, min_samples=d // 2),
        ret_bar=c / op - 1,
        range_bar=hi / lo - 1,
        clv=(c - lo) / (hi - lo),
        upper_wick=(hi - pl.max_horizontal(op, c)) / (hi - lo),
        runup_1h=c.shift(1) / c.shift(1 + h) - 1,
        ret_4h=c / c.shift(4 * h) - 1,
        ret_24h=c / c.shift(d) - 1,
        ret_7d=c / c.shift(7 * d) - 1,
        dist_hi_24h=c / hi.shift(1).rolling_max(d, min_samples=d // 2) - 1,
        dist_hi_7d=c / hi.shift(1).rolling_max(7 * d, min_samples=d) - 1,
        dist_lo_24h=c / lo.shift(1).rolling_min(d, min_samples=d // 2) - 1,
        vola_24h=logret.shift(1).rolling_std(d, min_samples=d // 2),
        range_base=(hi / lo - 1).shift(1).rolling_median(d, min_samples=d // 2),
        trades_base=pl.col("trade_count").shift(1).rolling_median(d, min_samples=d // 2),
        avg_trade=qv / pl.col("trade_count"),
        avg_trade_base=(qv / pl.col("trade_count")).shift(1).rolling_median(d, min_samples=d // 2),
        # price step as a share of price: the smallest non-zero close-to-close move in a day,
        # a proxy for the tick size (a coarse tick makes small targets cost a large spread)
        tick_rel=(c - c.shift(1)).abs().replace(0.0, None).rolling_min(d, min_samples=12) / c,
        flat_share=((c == c.shift(1)).cast(pl.Float64)).shift(1).rolling_mean(d, min_samples=12),
    )
    df = df.with_columns(
        vol_mult=v / pl.col("vol_base"),
        vol_mult_3h=v / pl.col("vol_mean_3h"),
        range_mult=pl.col("range_bar") / pl.col("range_base"),
        trades_mult=pl.col("trade_count") / pl.col("trades_base"),
        avg_trade_mult=pl.col("avg_trade") / pl.col("avg_trade_base"),
        tbr_excess=pl.col("tbr") - pl.col("tbr_base"),
        qv_usdt=qv,
    )
    spike3 = (pl.col("vol_mult") >= 3).cast(pl.Int32)
    df = df.with_columns(
        prev_spikes_1h=spike3.shift(1).rolling_sum(h, min_samples=1),
        prev_spikes_24h=spike3.shift(1).rolling_sum(d, min_samples=1),
        # forward close returns from the entry (open of t+1) at longer horizons
        entry=op.shift(-1),
    )
    df = df.with_columns(
        fwd_12h=c.shift(-12 * h) / pl.col("entry") - 1,
        fwd_24h=c.shift(-d) / pl.col("entry") - 1,
    )
    return df


def process(symbol: str) -> tuple[pl.DataFrame, np.ndarray] | None:
    raw = load(symbol)
    if raw.height < WARMUP + BARS_PER_DAY:
        return None
    df = features(full_grid(raw)).with_row_index("i")
    n = df.height
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    is_rand = rng.random(n) < RANDOM_RATE
    df = df.with_columns(is_random=pl.Series(is_rand))
    sel = df.filter(
        (pl.col("i") >= WARMUP)
        & (pl.col("i") < n - HORIZON - 1)
        & pl.col("volume").is_not_null()
        & pl.col("entry").is_not_null()
        & (pl.col("vol_base") > 0)
        & ((pl.col("vol_mult") >= SPIKE_MULT) | pl.col("is_random"))
    ).with_columns(is_spike=pl.col("vol_mult") >= SPIKE_MULT)
    if sel.height == 0:
        return None
    idx = sel["i"].to_numpy()
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    cl = df["close"].to_numpy()
    entry = sel["entry"].to_numpy()
    k = idx[:, None] + 1 + np.arange(HORIZON)[None, :]
    fwd = np.stack(
        [hi[k] / entry[:, None] - 1, lo[k] / entry[:, None] - 1, cl[k] / entry[:, None] - 1],
        axis=1,
    ).astype(np.float32)
    ok = ~np.isnan(fwd).any(axis=(1, 2))  # drop windows that cross an exchange gap
    sel = sel.filter(pl.Series(ok)).with_columns(
        symbol=pl.lit(symbol), listed_days=(pl.col("i") / BARS_PER_DAY).cast(pl.Float32)
    )
    return sel.drop("i"), fwd[ok]


def btc_context() -> pl.DataFrame:
    df = features(full_grid(load("BTCUSDT")))
    return df.select(
        "open_time",
        btc_ret_bar=pl.col("ret_bar"),
        btc_runup_1h=pl.col("runup_1h"),
        btc_ret_4h=pl.col("ret_4h"),
        btc_ret_24h=pl.col("ret_24h"),
        btc_ret_7d=pl.col("ret_7d"),
        btc_vol_mult=pl.col("vol_mult"),
        btc_vola_24h=pl.col("vola_24h"),
    )


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    symbols = universe()
    frames: list[pl.DataFrame] = []
    fwds: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for symbol, res in zip(symbols, pool.map(process, symbols), strict=True):
            if res is None:
                print(f"{symbol}: skipped", file=sys.stderr)
                continue
            frame, fwd = res
            frames.append(frame)
            fwds.append(fwd)
            print(
                f"{symbol}: {frame['is_spike'].sum()} spikes, {frame.height} rows", file=sys.stderr
            )
    events = (
        pl.concat(frames, how="vertical_relaxed")
        .with_row_index("row")
        .join(btc_context(), on="open_time", how="left", maintain_order="left")
        .sort("row")
    )
    events = events.with_columns(
        hour=pl.col("open_time").dt.hour(),
        dow=pl.col("open_time").dt.weekday(),
    )
    events.write_parquet(CACHE / "events.parquet")
    np.save(CACHE / "fwd.npy", np.concatenate(fwds))
    print(events.height, "rows;", events["is_spike"].sum(), "spikes")


if __name__ == "__main__":
    main()
