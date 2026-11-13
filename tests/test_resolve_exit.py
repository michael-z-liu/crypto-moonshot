from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from broker_fakes import T0, candles, flat

from crypto_moonshot.broker.base import Costs, ExitPlan, ExitReason
from crypto_moonshot.broker.candles import Candles, from_us, to_us
from crypto_moonshot.broker.simulated import ExitFill, fill_path, resolve_exit
from crypto_moonshot.models import KLINE_SCHEMA, Interval

FREE = Costs(fee=0.0, slippage=0.0, latency=0.0)
M5 = timedelta(minutes=5)
M1 = timedelta(minutes=1)


def resolve(
    bars: Candles, plan: ExitPlan, costs: Costs = FREE, fine: Candles | None = None
) -> ExitFill:
    fill = resolve_exit(bars, plan, costs, fine=fine)
    assert fill is not None
    return fill


# 1. Entry ------------------------------------------------------------------------------------


def test_levels_are_relative_to_the_raw_entry_open() -> None:
    bars = candles([(100, 101, 99, 100), (100, 120, 99, 110)])
    fill = resolve(bars, ExitPlan(take_profit=0.2), Costs())
    assert fill.entry_ref == 100
    assert fill.raw_price == pytest.approx(120)  # not 120 * (1 + slippage)


def test_entry_candle_itself_can_exit() -> None:
    bars = candles([(100, 125, 99, 110), (110, 130, 105, 120)])
    fill = resolve(bars, ExitPlan(take_profit=0.2))
    assert (fill.reason, fill.raw_price, fill.time) == (ExitReason.TAKE_PROFIT, 120, T0 + M5)


def test_empty_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="entry candle"):
        resolve_exit(candles([]), ExitPlan(take_profit=0.1))


# 2. Take profit ------------------------------------------------------------------------------


def test_take_profit_touch_fills_at_target_at_candle_close() -> None:
    bars = candles([*flat(2), (100, 121, 99, 105), *flat(2)])
    fill = resolve(bars, ExitPlan(take_profit=0.2))
    assert fill.reason is ExitReason.TAKE_PROFIT
    assert fill.raw_price == pytest.approx(120)
    assert fill.time == T0 + 3 * M5


def test_take_profit_gap_open_fills_at_open() -> None:
    bars = candles([*flat(2), (130, 135, 125, 128)])
    fill = resolve(bars, ExitPlan(take_profit=0.2))
    assert (fill.raw_price, fill.time) == (130, T0 + 2 * M5)


def test_take_profit_close_rule_needs_a_close_at_or_above_target() -> None:
    bars = candles([*flat(1), (100, 125, 99, 110), (110, 121, 108, 120), *flat(1, 115)])
    fill = resolve(bars, ExitPlan(take_profit=0.2, tp_fill="close"))
    assert fill.reason is ExitReason.TAKE_PROFIT
    assert fill.raw_price == pytest.approx(120)
    assert fill.time == T0 + 3 * M5  # close time of the third candle, not the touch in the second


def test_take_profit_close_rule_gap_open_fills_at_open() -> None:
    bars = candles([*flat(1), (124, 126, 110, 112)])
    fill = resolve(bars, ExitPlan(take_profit=0.2, tp_fill="close"))
    assert (fill.reason, fill.raw_price, fill.time) == (ExitReason.TAKE_PROFIT, 124, T0 + M5)


def test_take_profit_touch_only_never_fills_under_close_rule() -> None:
    bars = candles([*flat(1), (100, 125, 99, 110), *flat(2, 110)])
    fill = resolve(bars, ExitPlan(take_profit=0.2, tp_fill="close"))
    assert fill.reason is ExitReason.SERIES_END


# 3. Stop -------------------------------------------------------------------------------------


def test_stop_fills_at_level_at_candle_close() -> None:
    bars = candles([*flat(2), (100, 101, 90, 97), *flat(1)])
    fill = resolve(bars, ExitPlan(stop=0.05))
    assert (fill.reason, fill.raw_price, fill.time) == (ExitReason.STOP, 95, T0 + 3 * M5)


def test_stop_gap_open_fills_at_open() -> None:
    bars = candles([*flat(2), (90, 92, 85, 91)])
    fill = resolve(bars, ExitPlan(stop=0.05))
    assert (fill.reason, fill.raw_price, fill.time) == (ExitReason.STOP, 90, T0 + 2 * M5)


def test_stop_exactly_touched_fills() -> None:
    fill = resolve(candles([*flat(1), (100, 100, 95, 96)]), ExitPlan(stop=0.05))
    assert fill.reason is ExitReason.STOP


# 4. Trailing stop ----------------------------------------------------------------------------


def test_trailing_stop_trails_the_highest_earlier_high() -> None:
    bars = candles(
        [
            (100, 101, 99, 100),
            (100, 120, 105, 118),  # new peak 120; its own low 105 is above 101 * 0.9
            (118, 119, 107, 108),  # level 120 * 0.9 = 108, low 107 -> fills at 108
            *flat(1),
        ]
    )
    fill = resolve(bars, ExitPlan(trailing=0.1))
    assert fill.reason is ExitReason.TRAILING_STOP
    assert fill.raw_price == pytest.approx(108)
    assert fill.time == T0 + 3 * M5


def test_trailing_stop_starts_from_the_entry_price() -> None:
    bars = candles([(100, 100, 92, 95), *flat(1, 95)])
    fill = resolve(bars, ExitPlan(trailing=0.05))
    assert (fill.reason, fill.raw_price, fill.time) == (ExitReason.TRAILING_STOP, 95, T0 + M5)


def test_trailing_stop_is_not_lifted_by_the_candle_that_triggers_it() -> None:
    # The second candle's high of 200 would put the level at 180, above its low; but the level
    # in that candle comes from earlier highs only (101 * 0.9 = 90.9).
    bars = candles([(100, 101, 99, 100), (100, 200, 95, 190), (190, 195, 170, 175)])
    fill = resolve(bars, ExitPlan(trailing=0.1))
    assert fill.raw_price == pytest.approx(180)
    assert fill.time == T0 + 3 * M5


def test_trailing_stop_gap_open_fills_at_open() -> None:
    bars = candles([(100, 120, 100, 118), (100, 101, 95, 96)])
    fill = resolve(bars, ExitPlan(trailing=0.1))
    assert (fill.reason, fill.raw_price, fill.time) == (ExitReason.TRAILING_STOP, 100, T0 + M5)


def test_higher_of_stop_and_trailing_level_fills() -> None:
    # After the peak of 120 the trailing level (108) is above the fixed stop (95).
    bars = candles([(100, 120, 100, 118), (118, 118, 90, 92)])
    fill = resolve(bars, ExitPlan(stop=0.05, trailing=0.1))
    assert fill.reason is ExitReason.TRAILING_STOP
    assert fill.raw_price == pytest.approx(108)
    # Before any rise the fixed stop (95) is above the trailing level (90).
    bars = candles([(100, 100, 85, 88)])
    fill = resolve(bars, ExitPlan(stop=0.05, trailing=0.1))
    assert fill.reason is ExitReason.STOP
    assert fill.raw_price == pytest.approx(95)


def test_equal_stop_and_trailing_levels_count_as_stop() -> None:
    fill = resolve(candles([(100, 100, 85, 88)]), ExitPlan(stop=0.1, trailing=0.1))
    assert fill.reason is ExitReason.STOP
    assert fill.raw_price == pytest.approx(90)


# 5. Stop and take profit in one candle -------------------------------------------------------


def test_stop_fills_first_when_one_candle_touches_both() -> None:
    bars = candles([*flat(1), (100, 125, 90, 110)])
    fill = resolve(bars, ExitPlan(take_profit=0.2, stop=0.05))
    assert (fill.reason, fill.raw_price) == (ExitReason.STOP, 95)


def test_trailing_stop_fills_first_when_one_candle_touches_both() -> None:
    bars = candles([(100, 115, 100, 112), (112, 125, 100, 110)])
    fill = resolve(bars, ExitPlan(take_profit=0.2, trailing=0.1))
    assert fill.reason is ExitReason.TRAILING_STOP
    assert fill.raw_price == pytest.approx(103.5)


def test_gap_open_beyond_target_beats_a_stop_in_the_same_candle() -> None:
    bars = candles([*flat(1), (122, 125, 90, 95)])
    fill = resolve(bars, ExitPlan(take_profit=0.2, stop=0.05))
    assert (fill.reason, fill.raw_price, fill.time) == (ExitReason.TAKE_PROFIT, 122, T0 + M5)


def test_gap_through_stop_beats_a_target_touch_in_the_same_candle() -> None:
    bars = candles([*flat(1), (90, 125, 88, 120)])
    fill = resolve(bars, ExitPlan(take_profit=0.2, stop=0.05))
    assert (fill.reason, fill.raw_price, fill.time) == (ExitReason.STOP, 90, T0 + M5)


def test_close_rule_gap_through_a_trailing_stop_above_the_target_fills_the_stop() -> None:
    # With `close` fills the trailing level can sit above the target: peak 150 -> level 135.
    bars = candles([(100, 150, 100, 110), (125, 126, 120, 121)])
    fill = resolve(bars, ExitPlan(take_profit=0.2, trailing=0.1, tp_fill="close"))
    assert (fill.reason, fill.raw_price) == (ExitReason.TRAILING_STOP, 125)


# 6. Time exit --------------------------------------------------------------------------------


def test_one_hour_hold_on_5m_exits_at_the_close_of_the_12th_candle() -> None:
    bars = candles([(100, 101, 99, 100 + k) for k in range(20)])
    fill = resolve(bars, ExitPlan(take_profit=0.5, max_hold=timedelta(hours=1)))
    assert fill.reason is ExitReason.TIME
    assert fill.raw_price == 111  # close of candle 11 (0-based), the 12th
    assert fill.time == T0 + timedelta(hours=1)


def test_candles_after_the_deadline_are_ignored() -> None:
    bars = candles([*flat(12), (100, 200, 50, 100)])
    fill = resolve(bars, ExitPlan(take_profit=0.2, stop=0.05, max_hold=timedelta(hours=1)))
    assert (fill.reason, fill.raw_price) == (ExitReason.TIME, 100)


def test_window_ending_at_the_deadline_is_a_time_exit() -> None:
    fill = resolve(candles(flat(12), ended=False), ExitPlan(max_hold=timedelta(hours=1)))
    assert (fill.reason, fill.time) == (ExitReason.TIME, T0 + timedelta(hours=1))


# 7. Multi-resolution -------------------------------------------------------------------------


def _fine(bars: list[tuple[float, float, float, float] | None], start_minute: int) -> Candles:
    return candles(bars, interval=Interval.M1, start=T0 + start_minute * M1)


BOTH = (100.0, 125.0, 90.0, 110.0)  # 5m candle touching a +20% target and a -5% stop


def test_fine_candles_resolve_target_first() -> None:
    bars = candles([*flat(1), BOTH, *flat(1, 110)])
    fine = _fine([(100, 101, 99, 100), (100, 125, 100, 121), (121, 121, 90, 95), *flat(2, 110)], 5)
    plan = ExitPlan(take_profit=0.2, stop=0.05)
    assert resolve(bars, plan).reason is ExitReason.STOP
    fill = resolve(bars, plan, fine=fine)
    assert (fill.reason, fill.raw_price) == (ExitReason.TAKE_PROFIT, 120)
    assert fill.time == T0 + 7 * M1  # close of the second 1m candle


def test_fine_candles_resolve_stop_first() -> None:
    bars = candles([*flat(1), BOTH, *flat(1, 110)])
    fine = _fine([(100, 101, 99, 100), (100, 100, 90, 95), (95, 125, 95, 121), *flat(2, 110)], 5)
    fill = resolve(bars, ExitPlan(take_profit=0.2, stop=0.05), fine=fine)
    assert (fill.reason, fill.raw_price, fill.time) == (ExitReason.STOP, 95, T0 + 7 * M1)


def test_both_touched_in_one_fine_candle_is_still_stop_first() -> None:
    bars = candles([*flat(1), BOTH])
    fine = _fine([*flat(2), BOTH, *flat(2)], 5)
    fill = resolve(bars, ExitPlan(take_profit=0.2, stop=0.05), fine=fine)
    assert (fill.reason, fill.time) == (ExitReason.STOP, T0 + 8 * M1)


def test_partly_covered_coarse_candle_is_kept() -> None:
    bars = candles([*flat(1), BOTH])
    # Only four of the five 1m candles: the target-first order cannot be trusted.
    fine = _fine([(100, 125, 100, 121), (121, 121, 90, 95), *flat(1), None, *flat(1)], 5)
    fill = resolve(bars, ExitPlan(take_profit=0.2, stop=0.05), fine=fine)
    assert (fill.reason, fill.time) == (ExitReason.STOP, T0 + 10 * M1)


def test_fill_path_mixes_resolutions_generically() -> None:
    # 15m candles, 5m fine candles covering only the second one.
    coarse = candles(flat(3), interval=Interval.M15)
    fine = candles([(1, 2, 0.5, 1.5)] * 3, interval=Interval.M5, start=T0 + timedelta(minutes=15))
    path = fill_path(coarse, fine)
    minutes = [(from_us(int(t)) - T0) // M1 for t in path.open_time]
    lengths = [
        (int(c) - int(o)) // 60_000_000
        for o, c in zip(path.open_time, path.close_time, strict=True)
    ]
    assert minutes == [0, 15, 20, 25, 30]
    assert lengths == [15, 5, 5, 5, 15]
    assert path.open.tolist() == [100, 1, 1, 1, 100]


def test_fine_interval_must_divide_coarse() -> None:
    with pytest.raises(ValueError, match="evenly divide"):
        fill_path(candles(flat(1)), candles(flat(1), interval=Interval.M3))


def test_time_exit_on_fine_path_uses_the_fine_close() -> None:
    bars = candles(flat(3))
    fine = _fine([(100, 100, 100, 100 + k) for k in range(15)], 0)
    fill = resolve(bars, ExitPlan(max_hold=timedelta(minutes=12)), fine=fine)
    assert (fill.reason, fill.raw_price, fill.time) == (ExitReason.TIME, 111, T0 + 12 * M1)


# 8. Holes and ended series -------------------------------------------------------------------


def test_no_fill_inside_a_hole_and_gap_rule_after_it() -> None:
    bars = candles([*flat(2), None, None, (88, 99, 87, 98)])
    fill = resolve(bars, ExitPlan(stop=0.05))
    assert (fill.reason, fill.raw_price, fill.time) == (ExitReason.STOP, 88, T0 + 4 * M5)


def test_deadline_inside_a_hole_fills_at_the_last_close_before_it() -> None:
    bars = candles([*flat(10), (100, 101, 99, 97), None, None, None, *flat(3, 50)])
    fill = resolve(bars, ExitPlan(max_hold=timedelta(hours=1)))
    assert (fill.reason, fill.raw_price, fill.time) == (
        ExitReason.TIME,
        97,
        T0 + timedelta(hours=1),
    )


def test_deadline_inside_a_hole_of_an_unended_window() -> None:
    bars = candles([*flat(10), (100, 101, 99, 97)], ended=False)
    fill = resolve(bars, ExitPlan(max_hold=timedelta(hours=1)))
    assert (fill.reason, fill.raw_price) == (ExitReason.TIME, 97)


def test_series_end_closes_at_the_last_close() -> None:
    bars = candles([*flat(4), (100, 102, 98, 101)])
    fill = resolve(bars, ExitPlan(take_profit=0.2, max_hold=timedelta(hours=1)))
    assert (fill.reason, fill.raw_price, fill.time) == (ExitReason.SERIES_END, 101, T0 + 5 * M5)


def test_series_end_without_time_exit() -> None:
    fill = resolve(candles([*flat(2), (100, 100, 100, 99)]), ExitPlan(stop=0.5))
    assert (fill.reason, fill.raw_price) == (ExitReason.SERIES_END, 99)


def test_window_too_short_returns_none() -> None:
    assert resolve_exit(candles(flat(5), ended=False), ExitPlan(take_profit=0.2)) is None


# 9. Costs ------------------------------------------------------------------------------------

COSTS = Costs(fee=0.001, slippage=0.0005, latency=0.002)


def test_take_profit_costs() -> None:
    fill = resolve(candles([(100, 121, 99, 110)]), ExitPlan(take_profit=0.2), COSTS)
    assert fill.entry_price == pytest.approx(100 * 1.0025)
    assert fill.price == pytest.approx(120 * 0.9995)  # a limit: no latency
    assert fill.net_return == pytest.approx(120 * 0.9995 * 0.999 / (100.25 * 1.001) - 1)


def test_stop_costs_include_latency() -> None:
    fill = resolve(candles([(100, 100, 90, 92)]), ExitPlan(stop=0.05), COSTS)
    assert fill.price == pytest.approx(95 * 0.9975)
    assert fill.net_return == pytest.approx(95 * 0.9975 * 0.999 / (100.25 * 1.001) - 1)


def test_time_and_series_end_costs_include_latency() -> None:
    fill = resolve(candles(flat(13, 110)), ExitPlan(max_hold=timedelta(hours=1)), COSTS)
    assert fill.price == pytest.approx(110 * 0.9975)
    fill = resolve(candles(flat(3, 110)), ExitPlan(take_profit=0.5), COSTS)
    assert fill.reason is ExitReason.SERIES_END
    assert fill.price == pytest.approx(110 * 0.9975)


def test_default_costs_round_trip() -> None:
    fill = resolve(candles(flat(13)), ExitPlan(max_hold=timedelta(hours=1)), Costs())
    assert fill.net_return == pytest.approx(0.9995 * 0.999 / (1.0005 * 1.001) - 1)
    assert fill.net_return == pytest.approx(-0.003, abs=2e-5)


# Inputs --------------------------------------------------------------------------------------


def test_candles_from_kline_frame_and_window() -> None:
    times = [T0 + k * M5 for k in (2, 0, 1, 4)]
    frame = pl.DataFrame(
        {
            "open_time": times,
            "open": [3.0, 1.0, 2.0, 5.0],
            "high": [3.5, 1.5, 2.5, 5.5],
            "low": [2.5, 0.5, 1.5, 4.5],
            "close": [3.0, 1.0, 2.0, 5.0],
            "volume": [0.0] * 4,
            "quote_volume": [0.0] * 4,
            "trade_count": [0] * 4,
            "taker_buy_base_volume": [0.0] * 4,
            "taker_buy_quote_volume": [0.0] * 4,
        },
        schema=KLINE_SCHEMA,
    )
    series = Candles.from_frame(frame, Interval.M5)
    assert series.open.tolist() == [1, 2, 3, 5]
    assert series.open_time.dtype == np.int64
    assert series.open_time[0] == to_us(T0)
    window = series.window(T0 + M5, T0 + 3 * M5)
    assert (window.open.tolist(), window.ended) == ([2, 3], False)
    tail = series.window(T0 + 3 * M5)
    assert (tail.open.tolist(), tail.ended) == ([5], True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"take_profit": 0.0},
        {"stop": 1.0},
        {"trailing": -0.1},
        {"max_hold": timedelta(0)},
        {"tp_fill": "high"},
    ],
)
def test_exit_plan_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must be"):
        ExitPlan(**kwargs)  # type: ignore[arg-type]


def test_long_window_resolves_quickly() -> None:
    """Three days of 1m candles under 5m candles: one vectorised pass, no per-candle loop."""
    rng = np.random.default_rng(0)
    n = 864
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n * 5)))
    fine_bars = [(c, c * 1.0005, c * 0.9995, c) for c in closes]
    fine = candles(fine_bars, interval=Interval.M1)
    coarse_bars = [
        (
            closes[5 * k],
            closes[5 * k : 5 * k + 5].max() * 1.0005,
            closes[5 * k : 5 * k + 5].min() * 0.9995,
            closes[5 * k + 4],
        )
        for k in range(n)
    ]
    plan = ExitPlan(take_profit=10.0, max_hold=timedelta(days=3))
    fill = resolve(candles(coarse_bars), plan, fine=fine)
    assert fill.reason is ExitReason.TIME
    assert fill.raw_price == pytest.approx(closes[-1])
