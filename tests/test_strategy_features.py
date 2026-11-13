"""Parity of the ported features with a direct loop over the research formulas.

The reference below builds the gap-free 5m grid by hand and evaluates each research formula
(`research/volume_spike/build_events.py`) one candle at a time: a rolling window covers the
previous `w` grid rows, skips missing values and is null with fewer than `min_samples` present.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from store_fakes import kline_frame

from crypto_moonshot.strategy.features import btc_context, features, full_grid, spikes

M5 = timedelta(minutes=5)
T0 = datetime(2023, 3, 1, tzinfo=UTC)
DAY = 288
WARMUP = 7 * DAY

type Column = list[float | None]


def series(n: int, holes: list[range], seed: int = 1) -> pl.DataFrame:
    """`n` grid candles from T0 with random prices and volumes; `holes` are left out."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate(([100.0], close[:-1]))
    volume = rng.lognormal(3, 1, n)
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    keep = np.ones(n, dtype=bool)
    for hole in holes:
        keep[list(hole)] = False
    times = [T0 + k * M5 for k in range(n) if keep[k]]
    return kline_frame(
        times,
        close[keep],
        volume[keep],
        open_=open_[keep],
        high=high[keep],
        low=low[keep],
        quote_volume=volume[keep] * close[keep] * 1.5,
    )


def grid_columns(frame: pl.DataFrame) -> tuple[list[datetime], dict[str, Column]]:
    """The gap-free grid from the first to the last candle, built by hand."""
    rows = {r["open_time"]: r for r in frame.iter_rows(named=True)}
    first, last = min(rows), max(rows)
    times = [first + k * M5 for k in range((last - first) // M5 + 1)]
    names = ("open", "high", "low", "close", "volume", "quote_volume")
    cols: dict[str, Column] = {
        name: [rows[t][name] if t in rows else None for t in times] for name in names
    }
    return times, cols


def shifted(x: Column, k: int) -> Column:
    return [None] * k + x[: len(x) - k]


def rolling(x: Column, w: int, min_samples: int, fn: str) -> Column:
    out: Column = []
    for i in range(len(x)):
        window = [v for v in x[max(0, i - w + 1) : i + 1] if v is not None]
        if len(window) < min_samples:
            out.append(None)
        else:
            out.append(float(np.median(window)) if fn == "median" else max(window))
    return out


def ratio(a: Column, b: Column) -> Column:
    return [None if x is None or y is None else x / y - 1 for x, y in zip(a, b, strict=True)]


def reference(frame: pl.DataFrame) -> dict[str, Column]:
    _, g = grid_columns(frame)
    v, qv, c, hi, op = g["volume"], g["quote_volume"], g["close"], g["high"], g["open"]
    vol_base = rolling(shifted(v, 1), DAY, DAY // 2, "median")
    return {
        "vol_base": vol_base,
        "vol_mult": [
            None if x is None or b is None else x / b for x, b in zip(v, vol_base, strict=True)
        ],
        "qv_base": rolling(shifted(qv, 1), DAY, DAY // 2, "median"),
        "ret_bar": ratio(c, op),
        "ret_4h": ratio(c, shifted(c, 48)),
        "ret_24h": ratio(c, shifted(c, DAY)),
        "dist_hi_7d": ratio(c, rolling(shifted(hi, 1), 7 * DAY, DAY, "max")),
    }


def assert_same(actual: pl.Series, expected: Column) -> None:
    got = actual.to_list()
    assert len(got) == len(expected)
    assert [x is None for x in got] == [x is None for x in expected], actual.name
    a = np.array([x for x in got if x is not None])
    e = np.array([x for x in expected if x is not None])
    np.testing.assert_allclose(a, e, rtol=1e-12, err_msg=str(actual.name))


# A short exchange gap, and one longer than a day (the rolling windows lose their samples).
HOLES = [range(700, 704), range(1200, 1200 + DAY + 50)]
N = WARMUP + 2 * DAY + 400


@pytest.fixture(scope="module")
def candles() -> pl.DataFrame:
    return series(N, HOLES)


def test_full_grid_fills_holes_with_nulls(candles: pl.DataFrame) -> None:
    grid = full_grid(candles)
    assert grid.height == N
    assert grid["open_time"].diff().drop_nulls().unique().to_list() == [M5]
    missing = grid.filter(pl.col("close").is_null())
    assert missing.height == sum(len(h) for h in HOLES)
    assert grid.schema["open_time"] == pl.Datetime("us", "UTC")


@pytest.mark.parametrize(
    "name", ["vol_base", "vol_mult", "qv_base", "ret_bar", "ret_4h", "ret_24h", "dist_hi_7d"]
)
def test_features_match_the_research_formulas(candles: pl.DataFrame, name: str) -> None:
    got = features(full_grid(candles))
    assert_same(got[name], reference(candles)[name])


def test_min_samples_across_a_long_hole(candles: pl.DataFrame) -> None:
    got = features(full_grid(candles))
    after = HOLES[1].stop  # first candle after the long hole
    base = got["vol_base"]
    # Not enough of the previous day is present until half a day of candles has followed.
    assert base[after + DAY // 2 - 1] is None
    assert base[after + DAY // 2] is not None
    # The short hole only thins the window.
    assert base[704] is not None
    # Before the first half day nothing is known; ret_4h needs the candle 4h earlier.
    assert base[DAY // 2 - 1] is None
    assert base[DAY // 2] is not None
    assert got["ret_4h"][after + 47] is None
    assert got["ret_4h"][after + 48] is not None


def test_spikes_skip_the_warmup_and_need_a_volume_base() -> None:
    n = WARMUP + DAY + 100
    frame = series(n, [], seed=3)
    volume = np.full(n, 10.0)
    for i in (WARMUP - 1, WARMUP, WARMUP + 50):
        volume[i] = 500.0  # 50x the base
    frame = frame.with_columns(volume=pl.Series(volume))
    got = spikes(frame)
    assert got["open_time"].to_list() == [T0 + WARMUP * M5, T0 + (WARMUP + 50) * M5]
    assert got["vol_mult"].to_list() == [50.0, 50.0]
    assert got.columns[0] == "open_time"
    # A zero volume base is never a spike, however large the volume.
    zero = frame.with_columns(volume=pl.Series(np.where(volume > 10, 500.0, 0.0)))
    assert spikes(zero).height == 0


def test_spikes_skip_short_series_by_stored_candles() -> None:
    # The warm-up and minimum length count rows of the grid and stored candles respectively.
    n = WARMUP + DAY
    frame = series(n + 10, [range(100, 110)], seed=4)
    assert frame.height == n
    volume = np.full(n, 10.0)
    volume[-1] = 500.0
    frame = frame.with_columns(volume=pl.Series(volume))
    assert spikes(frame).height == 1
    assert spikes(frame.slice(1)).height == 0  # one stored candle short


def test_btc_context_is_btc_vol_mult_on_its_own_grid(candles: pl.DataFrame) -> None:
    ctx = btc_context(candles)
    assert ctx.columns == ["open_time", "btc_vol_mult"]
    assert ctx.height == N
    assert_same(ctx["btc_vol_mult"], reference(candles)["vol_mult"])
