from datetime import timedelta

import polars as pl
from broker_fakes import T0

from crypto_moonshot.backtest.source import InMemoryCandleSource
from crypto_moonshot.models import Interval

M5 = timedelta(minutes=5)


def frame(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time": [T0 + k * M5 for k in range(n)],
            **{c: [100.0 + k for k in range(n)] for c in ("open", "high", "low", "close")},
        },
        schema_overrides={"open_time": pl.Datetime("us", "UTC")},
    )


def test_windows_report_the_end_of_an_ended_series() -> None:
    src = InMemoryCandleSource.from_frames({("AUSDT", Interval.M5): frame(10)})
    inner = src.window("AUSDT", Interval.M5, T0 + 2 * M5, T0 + 5 * M5)
    assert inner is not None
    assert (len(inner), inner.ended, float(inner.open[0])) == (3, False, 102.0)
    tail = src.window("AUSDT", Interval.M5, T0 + 8 * M5, T0 + 50 * M5)
    assert tail is not None
    assert (len(tail), tail.ended) == (2, True)
    assert src.window("AUSDT", Interval.M1, T0, T0 + M5) is None
    assert src.window("BUSDT", Interval.M5, T0, T0 + M5) is None


def test_series_that_continue_never_report_ended() -> None:
    src = InMemoryCandleSource.from_frames({("AUSDT", Interval.M5): frame(10)}, ended=False)
    tail = src.window("AUSDT", Interval.M5, T0 + 8 * M5, T0 + 50 * M5)
    assert tail is not None
    assert not tail.ended
