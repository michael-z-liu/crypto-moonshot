"""Volume-spike features, ported from the research (`research/volume_spike/build_events.py`).

Same formulas, windows and `min_samples` as the research, computed the same way: on a gap-free
grid of the signal interval (`full_grid`), where a missing candle is a row of nulls. Polars'
rolling functions skip nulls inside a window and give null when fewer than `min_samples`
non-null values remain, so a hole thins the windows that span it instead of shifting them.
Every feature of candle *t* uses candles up to and including *t* only.

Windows are given in time and converted to candles of the grid's interval; on 5m candles they
are the research's: a day is 288 candles, 4 hours 48, the warm-up 7 days (2016 candles).
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from crypto_moonshot.models import Interval

HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
WARMUP = 7 * DAY
"""A symbol's first week of grid is skipped: its rolling features are still filling."""
MIN_HISTORY = WARMUP + DAY
"""Symbols with fewer stored candles than this span are skipped."""

FEATURES = ("vol_base", "vol_mult", "qv_base", "ret_bar", "ret_4h", "ret_24h", "dist_hi_7d")


def bars(span: timedelta, interval: Interval) -> int:
    """Candles of `interval` in `span` (which must be a whole number of them)."""
    count, rest = divmod(span, interval.length)
    if rest:
        msg = f"{span} is not a whole number of {interval.value} candles"
        raise ValueError(msg)
    return count


def full_grid(candles: pl.DataFrame, interval: Interval = Interval.M5) -> pl.DataFrame:
    """Reindex to a gap-free grid from the first to the last candle; missing candles become
    rows of nulls (research `full_grid`)."""
    if candles.height == 0:
        return candles
    candles = candles.sort("open_time")
    dtype = candles.schema["open_time"]
    grid = pl.DataFrame(
        {
            "open_time": pl.datetime_range(
                candles["open_time"][0],
                candles["open_time"][-1],
                interval.length,
                eager=True,
                time_zone="UTC",
            ).cast(dtype)
        }
    )
    return grid.join(candles, on="open_time", how="left")


def features(grid: pl.DataFrame, interval: Interval = Interval.M5) -> pl.DataFrame:
    """Add the ported feature columns (`FEATURES`) to a gap-free grid.

    - `vol_base`: median volume of the previous day (min half a day of candles); `qv_base` the
      same for quote volume; `vol_mult = volume / vol_base`.
    - `ret_bar = close / open - 1`; `ret_4h`, `ret_24h`: close over the close 4h / 24h earlier.
    - `dist_hi_7d`: close over the highest high of the previous 7 days (min one day) - 1.
    """
    d = bars(DAY, interval)
    h = bars(HOUR, interval)
    v, qv, c = pl.col("volume"), pl.col("quote_volume"), pl.col("close")
    hi, op = pl.col("high"), pl.col("open")
    grid = grid.with_columns(
        vol_base=v.shift(1).rolling_median(d, min_samples=d // 2),
        qv_base=qv.shift(1).rolling_median(d, min_samples=d // 2),
        ret_bar=c / op - 1,
        ret_4h=c / c.shift(4 * h) - 1,
        ret_24h=c / c.shift(d) - 1,
        dist_hi_7d=c / hi.shift(1).rolling_max(7 * d, min_samples=d) - 1,
    )
    return grid.with_columns(vol_mult=v / pl.col("vol_base"))


def spikes(
    candles: pl.DataFrame, spike_mult: float = 10.0, interval: Interval = Interval.M5
) -> pl.DataFrame:
    """Volume spikes of one symbol's whole stored series, with their features.

    As the research: a series shorter than `MIN_HISTORY` (in stored candles) gives nothing;
    otherwise features are computed on the gap-free grid, the first `WARMUP` of grid rows are
    skipped, and a spike is a candle with `volume` present, `vol_base > 0` and
    `vol_mult >= spike_mult`. Returns `open_time` and `FEATURES`.
    """
    columns = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]
    if candles.height < bars(MIN_HISTORY, interval):
        return pl.DataFrame(schema={"open_time": candles.schema["open_time"], **_feature_schema()})
    grid = features(full_grid(candles.select(columns), interval), interval)
    return (
        grid.with_row_index("_i")
        .filter(
            (pl.col("_i") >= bars(WARMUP, interval))
            & pl.col("volume").is_not_null()
            & (pl.col("vol_base") > 0)
            & (pl.col("vol_mult") >= spike_mult)
        )
        .drop("_i")
        .select("open_time", *FEATURES)
    )


def btc_context(btc: pl.DataFrame, interval: Interval = Interval.M5) -> pl.DataFrame:
    """BTC's `vol_mult` by open time (`btc_vol_mult`), computed on BTC's own gap-free grid
    without a warm-up skip (research `btc_context`)."""
    if btc.height == 0:
        return pl.DataFrame(
            schema={"open_time": btc.schema["open_time"], "btc_vol_mult": pl.Float64}
        )
    grid = features(full_grid(btc, interval), interval)
    return grid.select("open_time", btc_vol_mult=pl.col("vol_mult"))


def _feature_schema() -> dict[str, pl.DataType]:
    return {name: pl.Float64() for name in FEATURES}
