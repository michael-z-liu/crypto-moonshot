import json
import os
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from crypto_moonshot.data.store import (
    SCHEMA_VERSION_KEY,
    InvalidFrameError,
    InvalidMonthFileError,
    KlineStore,
    SeriesMeta,
    add_gaps,
    check_month_file,
    missing_ranges,
    parse_month_path,
    read_meta,
    split_by_month,
    write_meta,
)
from crypto_moonshot.models import KLINE_SCHEMA, SCHEMA_VERSION, Interval

NOW = datetime(2026, 6, 1, tzinfo=UTC)
JAN = datetime(2024, 1, 1, tzinfo=UTC)
FEB = datetime(2024, 2, 1, tzinfo=UTC)
SYMBOL = "BTCUSDT"

Range = tuple[datetime, datetime]


def dt(year: int, month: int = 1, day: int = 1, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def make_frame(
    times: Iterable[datetime], price: float = 100.0, *, schema_overrides: bool = True
) -> pl.DataFrame:
    rows = list(times)
    n = len(rows)
    return pl.DataFrame(
        {
            "open_time": rows,
            "open": [price] * n,
            "high": [price + 2] * n,
            "low": [price - 1] * n,
            "close": [price + 1] * n,
            "volume": [10.5] * n,
            "quote_volume": [1050.0] * n,
            "trade_count": [7] * n,
            "taker_buy_base_volume": [5.0] * n,
            "taker_buy_quote_volume": [500.0] * n,
        },
        schema=KLINE_SCHEMA if schema_overrides else None,
    )


def hourly(start: datetime, count: int) -> list[datetime]:
    return [start + timedelta(hours=i) for i in range(count)]


@pytest.fixture
def store(tmp_path: Path) -> KlineStore:
    return KlineStore(tmp_path)


def put(
    root: Path,
    frame: pl.DataFrame,
    *,
    interval: str = "1h",
    symbol: str = SYMBOL,
    name: str = "2024-01.parquet",
    version: str | None = str(SCHEMA_VERSION),
) -> Path:
    """Write an arbitrary (possibly bad) month file at the standard location."""
    path = root / "klines" / "spot" / f"interval={interval}" / f"symbol={symbol}" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {} if version is None else {SCHEMA_VERSION_KEY: version}
    frame.write_parquet(path, metadata=metadata)
    return path


# ---------------------------------------------------------------- paths


def test_paths(store: KlineStore, tmp_path: Path) -> None:
    assert store.series_dir(SYMBOL, Interval.M5) == (
        tmp_path / "klines" / "spot" / "interval=5m" / "symbol=BTCUSDT"
    )
    assert store.month_path(SYMBOL, Interval.MO1, JAN).name == "2024-01.parquet"
    with pytest.raises(ValueError, match="symbol"):
        store.series_dir("../etc", Interval.H1)
    with pytest.raises(ValueError, match="symbol"):
        store.series_dir("btcusdt", Interval.H1)
    with pytest.raises(ValueError, match="start of a month"):
        store.month_path(SYMBOL, Interval.H1, dt(2024, 1, 2))


def test_parse_month_path_roundtrip(store: KlineStore) -> None:
    path = store.month_path("ETHUSDT", Interval.MO1, dt(2021, 12, 1))
    assert parse_month_path(path) == ("ETHUSDT", Interval.MO1, dt(2021, 12, 1))


def test_list_series_and_months(store: KlineStore, tmp_path: Path) -> None:
    store.write_month("ETHUSDT", Interval.H1, FEB, make_frame([FEB]))
    store.write_month(SYMBOL, Interval.H1, JAN, make_frame([JAN]))
    store.write_month(SYMBOL, Interval.H1, dt(2023, 12, 1), make_frame([dt(2023, 12, 1)]))
    store.write_month(SYMBOL, Interval.M5, JAN, make_frame([JAN]))
    # Things that must be ignored.
    (tmp_path / "klines" / "spot" / "interval=7x" / "symbol=BTCUSDT").mkdir(parents=True)
    (store.series_dir(SYMBOL, Interval.H1) / ".2024-03.parquet.abc.tmp").write_bytes(b"x")
    (store.series_dir(SYMBOL, Interval.H1) / "notes.parquet").write_bytes(b"x")
    assert store.list_series() == [
        ("BTCUSDT", Interval.M5),
        ("BTCUSDT", Interval.H1),
        ("ETHUSDT", Interval.H1),
    ]
    assert store.list_months(SYMBOL, Interval.H1) == [dt(2023, 12, 1), JAN]
    assert store.list_months("SOLUSDT", Interval.H1) == []
    assert KlineStore(tmp_path / "empty").list_series() == []


# ---------------------------------------------------------------- write/read


def test_write_read_roundtrip(store: KlineStore) -> None:
    # Written out of order on purpose: the store sorts.
    times = hourly(JAN, 48)[::-1]
    frame = make_frame(times)
    store.write_month(SYMBOL, Interval.H1, JAN, frame)
    path = store.month_path(SYMBOL, Interval.H1, JAN)

    back = store.read_month(SYMBOL, Interval.H1, JAN)
    assert dict(back.schema) == KLINE_SCHEMA
    assert list(back.columns) == list(KLINE_SCHEMA)
    assert back["open_time"].to_list() == sorted(times)
    assert back.equals(frame.sort("open_time"))
    assert pl.read_parquet_metadata(path)[SCHEMA_VERSION_KEY] == "1"
    assert check_month_file(path, now=NOW) == []
    # No temp files left behind.
    assert [p.name for p in path.parent.iterdir()] == ["2024-01.parquet"]


def test_week_candle_lives_in_month_of_its_open_time(store: KlineStore) -> None:
    opening = dt(2024, 1, 29)  # a Monday; the candle runs into February
    store.write_month(SYMBOL, Interval.W1, JAN, make_frame([opening]))
    assert store.list_months(SYMBOL, Interval.W1) == [JAN]


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (make_frame(hourly(JAN, 2)).drop("volume"), "missing columns: volume"),
        (make_frame(hourly(JAN, 2)).with_columns(extra=pl.lit(1)), "unexpected columns"),
        (
            make_frame(hourly(JAN, 2)).with_columns(pl.col("close").cast(pl.Float32)),
            "'close' has dtype Float32",
        ),
        (
            make_frame(hourly(JAN, 2)).with_columns(pl.col("open_time").dt.replace_time_zone(None)),
            "open_time must be Datetime",
        ),
        (make_frame(hourly(JAN, 2)).select(reversed(KLINE_SCHEMA)), "out of order"),
        (make_frame([dt(2023, 12, 31, 23), JAN]), "1 open_time values outside month 2024-01"),
        (make_frame([FEB]), "outside month"),
        (make_frame([JAN + timedelta(minutes=30)]), "not aligned to 1h"),
        (make_frame([JAN, JAN]), "2 duplicate open_time"),
        (make_frame([]), "no candles"),
    ],
    ids=[
        "missing-column",
        "extra-column",
        "wrong-dtype",
        "naive-time",
        "column-order",
        "before-month",
        "after-month",
        "misaligned",
        "duplicates",
        "empty",
    ],
)
def test_write_month_rejects_bad_frames(store: KlineStore, frame: pl.DataFrame, match: str) -> None:
    with pytest.raises(InvalidFrameError, match=match):
        store.write_month(SYMBOL, Interval.H1, JAN, frame)
    assert not store.series_dir(SYMBOL, Interval.H1).exists()


def test_write_month_rejects_misaligned_weekly_and_monthly(store: KlineStore) -> None:
    with pytest.raises(InvalidFrameError, match="not aligned to 1w"):
        store.write_month(SYMBOL, Interval.W1, JAN, make_frame([dt(2024, 1, 3)]))
    with pytest.raises(InvalidFrameError, match="not aligned to 1M"):
        store.write_month(SYMBOL, Interval.MO1, JAN, make_frame([dt(2024, 1, 2)]))
    store.write_month(SYMBOL, Interval.MO1, JAN, make_frame([JAN]))


# ---------------------------------------------------------------- atomicity


def test_failed_replace_leaves_old_file_and_no_temp(
    store: KlineStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write_month(SYMBOL, Interval.H1, JAN, make_frame(hourly(JAN, 3), price=1.0))
    path = store.month_path(SYMBOL, Interval.H1, JAN)
    before = path.read_bytes()

    def boom(src: object, dst: object) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk on fire"):
        store.write_month(SYMBOL, Interval.H1, JAN, make_frame(hourly(JAN, 5), price=2.0))
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == ["2024-01.parquet"]


def test_failed_write_leaves_no_partial_file(
    store: KlineStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def half_write(self: pl.DataFrame, file: Path, **kwargs: object) -> None:
        Path(file).write_bytes(b"PAR1 partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", half_write)
    with pytest.raises(RuntimeError, match="interrupted"):
        store.write_month(SYMBOL, Interval.H1, JAN, make_frame(hourly(JAN, 3)))
    assert list(store.series_dir(SYMBOL, Interval.H1).iterdir()) == []
    assert store.list_months(SYMBOL, Interval.H1) == []


def test_temp_file_name_is_hidden_and_not_parquet(
    store: KlineStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    real_replace = os.replace

    def spy(src: str | Path, dst: str | Path) -> None:
        seen.append(Path(src).name)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    store.write_month(SYMBOL, Interval.H1, JAN, make_frame([JAN]))
    store.write_meta(SYMBOL, Interval.H1, SeriesMeta(listing=JAN))
    assert len(seen) == 2
    assert all(name.startswith(".") and name.endswith(".tmp") for name in seen)
    assert not any(name.endswith(".parquet") for name in seen)


# ---------------------------------------------------------------- merge


def test_merge_into_new_month_and_idempotent(store: KlineStore) -> None:
    frame = make_frame(hourly(JAN, 5), price=1.0)
    assert store.merge_month(SYMBOL, Interval.H1, JAN, frame) == 5
    path = store.month_path(SYMBOL, Interval.H1, JAN)
    first = path.read_bytes()
    assert store.merge_month(SYMBOL, Interval.H1, JAN, frame) == 0
    assert path.read_bytes() == first  # not even rewritten
    assert store.merge_month(SYMBOL, Interval.H1, JAN, make_frame([])) == 0


def test_merge_existing_rows_win(store: KlineStore) -> None:
    store.write_month(SYMBOL, Interval.H1, JAN, make_frame(hourly(JAN, 3), price=1.0))
    new = make_frame(hourly(JAN, 6)[::-1], price=2.0)  # unsorted, overlaps 3 rows
    assert store.merge_month(SYMBOL, Interval.H1, JAN, new) == 3
    merged = store.read_month(SYMBOL, Interval.H1, JAN)
    assert merged["open_time"].to_list() == hourly(JAN, 6)
    assert merged["open"].to_list() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    assert pl.read_parquet_metadata(store.month_path(SYMBOL, Interval.H1, JAN))[
        SCHEMA_VERSION_KEY
    ] == str(SCHEMA_VERSION)


def test_merge_collapses_duplicates_within_new_frame(store: KlineStore) -> None:
    assert store.merge_month(SYMBOL, Interval.H1, JAN, make_frame([JAN, JAN])) == 1


def test_merge_validates_new_rows(store: KlineStore) -> None:
    with pytest.raises(InvalidFrameError, match="outside month"):
        store.merge_month(SYMBOL, Interval.H1, JAN, make_frame([FEB]))


def test_merge_over_invalid_existing_file_raises(store: KlineStore) -> None:
    path = store.month_path(SYMBOL, Interval.H1, JAN)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"garbage")
    with pytest.raises(InvalidMonthFileError):
        store.merge_month(SYMBOL, Interval.H1, JAN, make_frame([JAN]))
    # write_month replaces it.
    store.write_month(SYMBOL, Interval.H1, JAN, make_frame([JAN]))
    assert store.read_month(SYMBOL, Interval.H1, JAN).height == 1


def test_read_month_errors(store: KlineStore, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        store.read_month(SYMBOL, Interval.H1, JAN)
    put(tmp_path, make_frame([JAN]), version="9")
    with pytest.raises(InvalidMonthFileError, match="schema version '9'"):
        store.read_month(SYMBOL, Interval.H1, JAN)


# ---------------------------------------------------------------- stored_open_times


def test_stored_open_times_across_months(store: KlineStore) -> None:
    assert store.stored_open_times(SYMBOL, Interval.H1).len() == 0
    store.write_month(SYMBOL, Interval.H1, FEB, make_frame(hourly(FEB, 2)))
    store.write_month(SYMBOL, Interval.H1, JAN, make_frame(hourly(dt(2024, 1, 31, 22), 2)))
    times = store.stored_open_times(SYMBOL, Interval.H1)
    assert times.dtype == KLINE_SCHEMA["open_time"]
    assert times.to_list() == [
        dt(2024, 1, 31, 22),
        dt(2024, 1, 31, 23),
        FEB,
        FEB + timedelta(hours=1),
    ]
    only_feb = store.stored_open_times(SYMBOL, Interval.H1, months=[FEB])
    assert only_feb.len() == 2


def test_stored_open_times_unreadable_file(store: KlineStore) -> None:
    store.write_month(SYMBOL, Interval.H1, JAN, make_frame([JAN]))
    store.month_path(SYMBOL, Interval.H1, FEB).write_bytes(b"junk")
    with pytest.raises(InvalidMonthFileError):
        store.stored_open_times(SYMBOL, Interval.H1)


# ---------------------------------------------------------------- check_month_file


def test_check_good_file(tmp_path: Path) -> None:
    path = put(tmp_path, make_frame(hourly(JAN, 24)))
    assert check_month_file(path, now=NOW) == []


@pytest.mark.parametrize("interval", list(Interval))
def test_check_good_file_every_interval(tmp_path: Path, interval: Interval) -> None:
    times = [interval.floor(dt(2024, 1, 15))]
    path = put(tmp_path, make_frame(times), interval=interval.value)
    assert check_month_file(path, now=NOW) == []


def test_check_bad_path(tmp_path: Path) -> None:
    frame = make_frame([JAN])
    for kwargs, reason in [
        ({"interval": "7x"}, "unknown interval"),
        ({"interval": "1H"}, "unknown interval"),
        ({"symbol": "btcusdt"}, "invalid symbol"),
        ({"symbol": "BTC-USDT"}, "invalid symbol"),
        ({"name": "2024-13.parquet"}, "invalid month"),
        ({"name": "jan.parquet"}, "not YYYY-MM.parquet"),
    ]:
        problems = check_month_file(put(tmp_path, frame, **kwargs), now=NOW)
        assert len(problems) == 1
        assert problems[0].startswith("bad path:")
        assert reason in problems[0]
    loose = tmp_path / "x.parquet"
    frame.write_parquet(loose)
    assert "bad path" in check_month_file(loose, now=NOW)[0]


def test_check_schema_version(tmp_path: Path) -> None:
    frame = make_frame([JAN])
    missing = check_month_file(put(tmp_path, frame, version=None), now=NOW)
    assert missing == [f"missing {SCHEMA_VERSION_KEY} metadata"]
    wrong = check_month_file(put(tmp_path, frame, version="2"), now=NOW)
    assert wrong == ["schema version '2', expected '1'"]


def test_check_columns_and_dtypes(tmp_path: Path) -> None:
    frame = make_frame([JAN])
    problems = check_month_file(put(tmp_path, frame.drop("high")), now=NOW)
    assert problems == ["missing columns: high"]
    problems = check_month_file(put(tmp_path, frame.rename({"high": "hi"})), now=NOW)
    assert any("missing columns: high" in p for p in problems)
    assert any("unexpected columns: hi" in p for p in problems)
    problems = check_month_file(
        put(tmp_path, frame.with_columns(pl.col("trade_count").cast(pl.Int32))), now=NOW
    )
    assert problems == ["column 'trade_count' has dtype Int32, expected Int64"]
    problems = check_month_file(put(tmp_path, frame.select(reversed(KLINE_SCHEMA))), now=NOW)
    assert len(problems) == 1
    assert "out of order" in problems[0]


@pytest.mark.parametrize(
    "convert",
    [
        lambda c: c.dt.replace_time_zone(None),
        lambda c: c.dt.convert_time_zone("Europe/London"),
        lambda c: c.dt.cast_time_unit("ms"),
    ],
    ids=["naive", "non-utc", "millis"],
)
def test_check_open_time_must_be_utc_microseconds(
    tmp_path: Path, convert: Callable[[pl.Expr], pl.Expr]
) -> None:
    frame = make_frame([JAN]).with_columns(convert(pl.col("open_time")))
    problems = check_month_file(put(tmp_path, frame), now=NOW)
    assert len(problems) == 1
    assert problems[0].startswith("open_time must be Datetime(us, UTC)")


def test_check_out_of_month_and_misaligned_and_duplicate(tmp_path: Path) -> None:
    out = check_month_file(put(tmp_path, make_frame([dt(2023, 12, 31, 23), JAN])), now=NOW)
    assert out == ["1 open_time values outside month 2024-01"]
    mis = check_month_file(put(tmp_path, make_frame([JAN + timedelta(minutes=5)])), now=NOW)
    assert mis == ["1 open_time values not aligned to 1h"]
    dup = check_month_file(put(tmp_path, make_frame([JAN, JAN])), now=NOW)
    assert dup == ["2 duplicate open_time values"]
    unsorted = check_month_file(put(tmp_path, make_frame([JAN + timedelta(hours=1), JAN])), now=NOW)
    assert unsorted == ["open_time is not sorted ascending"]
    empty = check_month_file(put(tmp_path, make_frame([])), now=NOW)
    assert empty == ["file has no candles"]


def test_check_misaligned_week_and_month(tmp_path: Path) -> None:
    week = put(tmp_path, make_frame([dt(2024, 1, 3)]), interval="1w")
    assert check_month_file(week, now=NOW) == ["1 open_time values not aligned to 1w"]
    month = put(tmp_path, make_frame([dt(2024, 1, 2)]), interval="1M")
    assert check_month_file(month, now=NOW) == ["1 open_time values not aligned to 1M"]


def test_check_still_open_candle(tmp_path: Path) -> None:
    path = put(tmp_path, make_frame(hourly(JAN, 3)))
    last_close = JAN + timedelta(hours=3)
    assert check_month_file(path, now=last_close) == []
    problems = check_month_file(path, now=last_close - timedelta(seconds=1))
    assert len(problems) == 1
    assert "not closed yet" in problems[0]
    # A monthly candle closes at the start of the next month.
    month = put(tmp_path, make_frame([JAN]), interval="1M")
    assert check_month_file(month, now=dt(2024, 1, 31, 23, 59)) != []
    assert check_month_file(month, now=FEB) == []


def test_check_corrupt_files(tmp_path: Path) -> None:
    path = put(tmp_path, make_frame(hourly(JAN, 100)))
    for content in [b"", b"not parquet at all", path.read_bytes()[:50], path.read_bytes()[:-20]]:
        path.write_bytes(content)
        problems = check_month_file(path, now=NOW)
        assert len(problems) == 1
        assert problems[0].startswith("unreadable parquet file")
    assert check_month_file(
        tmp_path / "klines" / "spot" / "interval=1h" / "symbol=A" / "2024-01.parquet", now=NOW
    )[0].startswith("unreadable")


def test_invalid_months(store: KlineStore, tmp_path: Path) -> None:
    store.write_month(SYMBOL, Interval.H1, JAN, make_frame([JAN]))
    put(tmp_path, make_frame([FEB]), name="2024-02.parquet", version="3")
    store.month_path(SYMBOL, Interval.H1, dt(2024, 3, 1)).write_bytes(b"junk")
    invalid = store.invalid_months(SYMBOL, Interval.H1, now=NOW)
    assert list(invalid) == [FEB, dt(2024, 3, 1)]
    assert "schema version" in invalid[FEB][0]
    assert invalid[dt(2024, 3, 1)][0].startswith("unreadable")
    assert store.invalid_months("ETHUSDT", Interval.H1, now=NOW) == {}


# ---------------------------------------------------------------- meta


def test_meta_roundtrip_and_format(store: KlineStore) -> None:
    meta = SeriesMeta(
        listing=dt(2020, 8, 11, 6),
        gaps=[
            (dt(2020, 2, 19, 12), dt(2020, 2, 19, 17)),
            (dt(2021, 1, 1), dt(2021, 1, 2)),
        ],
    )
    store.write_meta(SYMBOL, Interval.H1, meta)
    path = store.series_dir(SYMBOL, Interval.H1) / "_meta.json"
    assert json.loads(path.read_text()) == {
        "schema_version": 1,
        "listing": "2020-08-11T06:00:00+00:00",
        "gaps": [
            ["2020-02-19T12:00:00+00:00", "2020-02-19T17:00:00+00:00"],
            ["2021-01-01T00:00:00+00:00", "2021-01-02T00:00:00+00:00"],
        ],
    }
    assert list(json.loads(path.read_text())) == ["schema_version", "listing", "gaps"]
    assert store.read_meta(SYMBOL, Interval.H1) == meta
    assert [p.name for p in path.parent.iterdir()] == ["_meta.json"]


def test_meta_null_listing_and_no_gaps(tmp_path: Path) -> None:
    write_meta(tmp_path, SeriesMeta())
    assert json.loads((tmp_path / "_meta.json").read_text()) == {
        "schema_version": 1,
        "listing": None,
        "gaps": [],
    }
    assert read_meta(tmp_path) == SeriesMeta()


def test_meta_written_gaps_are_sorted_and_merged(tmp_path: Path) -> None:
    gaps = [(dt(2020, 1, 3), dt(2020, 1, 4)), (dt(2020, 1, 1), dt(2020, 1, 3))]
    write_meta(tmp_path, SeriesMeta(gaps=gaps))
    raw = json.loads((tmp_path / "_meta.json").read_text())
    assert raw["gaps"] == [["2020-01-01T00:00:00+00:00", "2020-01-04T00:00:00+00:00"]]


def test_meta_missing_file_gives_default(tmp_path: Path) -> None:
    assert read_meta(tmp_path / "nope") == SeriesMeta()


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "",
        "[]",
        '{"schema_version": 2, "listing": null, "gaps": []}',
        '{"listing": null, "gaps": []}',
        '{"schema_version": 1, "listing": "yesterday", "gaps": []}',
        '{"schema_version": 1, "listing": "2020-01-01T00:00:00", "gaps": []}',
        '{"schema_version": 1, "listing": null, "gaps": "x"}',
        '{"schema_version": 1, "listing": null, "gaps": [["2020-01-01T00:00:00+00:00"]]}',
        '{"schema_version": 1, "listing": null, "gaps": [[1, 2]]}',
        '{"schema_version": 1, "listing": null, "gaps": '
        '[["2020-01-02T00:00:00+00:00", "2020-01-01T00:00:00+00:00"]]}',
    ],
)
def test_meta_corrupt_gives_default(tmp_path: Path, content: str) -> None:
    (tmp_path / "_meta.json").write_text(content)
    assert read_meta(tmp_path) == SeriesMeta()


def test_add_gaps_merges_overlapping_and_adjacent() -> None:
    existing = [(dt(2020, 1, 1), dt(2020, 1, 2)), (dt(2020, 1, 10), dt(2020, 1, 11))]
    new = [
        (dt(2020, 1, 2), dt(2020, 1, 3)),  # adjacent to the first
        (dt(2020, 1, 10, 12), dt(2020, 1, 12)),  # overlaps the second
        (dt(2020, 1, 20), dt(2020, 1, 21)),
        (dt(2020, 1, 5), dt(2020, 1, 5)),  # empty: dropped
    ]
    assert add_gaps(existing, new) == [
        (dt(2020, 1, 1), dt(2020, 1, 3)),
        (dt(2020, 1, 10), dt(2020, 1, 12)),
        (dt(2020, 1, 20), dt(2020, 1, 21)),
    ]
    meta = SeriesMeta(listing=dt(2020, 1, 1), gaps=existing).with_gaps(new)
    assert meta.listing == dt(2020, 1, 1)
    assert len(meta.gaps) == 3


# ---------------------------------------------------------------- missing_ranges


def times_series(times: Iterable[datetime]) -> pl.Series:
    return pl.Series("open_time", list(times), dtype=KLINE_SCHEMA["open_time"])


def test_missing_nothing_stored() -> None:
    start, end = dt(2024, 1, 1), dt(2024, 1, 2)
    assert missing_ranges(Interval.H1, start, end, times_series([])) == [(start, end)]


def test_missing_fully_stored() -> None:
    start, end = dt(2024, 1, 1), dt(2024, 1, 2)
    stored = times_series(Interval.H1.open_times(start, end))
    assert missing_ranges(Interval.H1, start, end, stored) == []


def test_missing_holes_in_middle_and_edges() -> None:
    start, end = dt(2024, 1, 1), dt(2024, 1, 1, 10)
    stored = times_series(h for i, h in enumerate(hourly(start, 10)) if i not in {0, 1, 4, 5, 6, 9})
    assert missing_ranges(Interval.H1, start, end, stored) == [
        (dt(2024, 1, 1, 0), dt(2024, 1, 1, 2)),
        (dt(2024, 1, 1, 4), dt(2024, 1, 1, 7)),
        (dt(2024, 1, 1, 9), dt(2024, 1, 1, 10)),
    ]


def test_missing_ignores_stored_outside_range_and_unsorted_duplicates() -> None:
    start, end = dt(2024, 1, 1, 5), dt(2024, 1, 1, 8)
    stored = times_series(
        [
            dt(2024, 1, 1, 6),
            dt(2024, 1, 1, 6),
            dt(2024, 1, 1, 1),
            dt(2024, 1, 1, 20),
            dt(2024, 1, 1, 5),
        ]
    )
    assert missing_ranges(Interval.H1, start, end, stored) == [
        (dt(2024, 1, 1, 7), dt(2024, 1, 1, 8))
    ]


def test_missing_gaps_exclude_ranges() -> None:
    start, end = dt(2024, 1, 1), dt(2024, 1, 1, 12)
    stored = times_series([dt(2024, 1, 1, 0), dt(2024, 1, 1, 11)])
    gaps = [(dt(2024, 1, 1, 3), dt(2024, 1, 1, 5)), (dt(2024, 1, 1, 8), dt(2024, 1, 1, 9))]
    assert missing_ranges(Interval.H1, start, end, stored, gaps) == [
        (dt(2024, 1, 1, 1), dt(2024, 1, 1, 3)),
        (dt(2024, 1, 1, 5), dt(2024, 1, 1, 8)),
        (dt(2024, 1, 1, 9), dt(2024, 1, 1, 11)),
    ]


def test_missing_gaps_cover_everything_or_extend_past_range() -> None:
    start, end = dt(2024, 1, 1), dt(2024, 1, 2)
    assert (
        missing_ranges(
            Interval.H1, start, end, times_series([]), [(dt(2023, 1, 1), dt(2025, 1, 1))]
        )
        == []
    )
    got = missing_ranges(
        Interval.H1, start, end, times_series([]), [(dt(2023, 1, 1), dt(2024, 1, 1, 6))]
    )
    assert got == [(dt(2024, 1, 1, 6), end)]


def test_missing_unaligned_gap_boundaries_are_rounded_to_candles() -> None:
    start, end = dt(2024, 1, 1), dt(2024, 1, 1, 6)
    gap = [(dt(2024, 1, 1, 2, 30), dt(2024, 1, 1, 3, 30))]
    # Open times 03:00 is inside the gap; 02:00 starts before it; 04:00 is after it.
    assert missing_ranges(Interval.H1, start, end, times_series([]), gap) == [
        (dt(2024, 1, 1, 0), dt(2024, 1, 1, 3)),
        (dt(2024, 1, 1, 4), dt(2024, 1, 1, 6)),
    ]


def test_missing_unaligned_start_and_end() -> None:
    # Candle open times in [00:07, 00:53) at 5m: 00:10 .. 00:50.
    start, end = dt(2024, 1, 1, 0, 7), dt(2024, 1, 1, 0, 53)
    got = missing_ranges(Interval.M5, start, end, times_series([dt(2024, 1, 1, 0, 20)]))
    assert got == [
        (dt(2024, 1, 1, 0, 10), dt(2024, 1, 1, 0, 20)),
        (dt(2024, 1, 1, 0, 25), dt(2024, 1, 1, 0, 55)),
    ]
    assert missing_ranges(Interval.M5, end, start, times_series([])) == []
    assert (
        missing_ranges(Interval.M5, dt(2024, 1, 1, 0, 1), dt(2024, 1, 1, 0, 4), times_series([]))
        == []
    )


def test_missing_weekly() -> None:
    start, end = dt(2024, 1, 1), dt(2024, 2, 1)  # Mondays: 1, 8, 15, 22, 29
    stored = times_series([dt(2024, 1, 8), dt(2024, 1, 22)])
    assert missing_ranges(Interval.W1, start, end, stored) == [
        (dt(2024, 1, 1), dt(2024, 1, 8)),
        (dt(2024, 1, 15), dt(2024, 1, 22)),
        (dt(2024, 1, 29), dt(2024, 2, 5)),
    ]
    # Start on a Wednesday: the first candle that opens at or after it is the 8th.
    assert missing_ranges(Interval.W1, dt(2024, 1, 3), end, stored)[0] == (
        dt(2024, 1, 15),
        dt(2024, 1, 22),
    )
    # A stored time that is not a Monday does not count.
    odd = times_series([dt(2024, 1, 9)])
    assert missing_ranges(Interval.W1, dt(2024, 1, 8), dt(2024, 1, 15), odd) == [
        (dt(2024, 1, 8), dt(2024, 1, 15))
    ]


def test_missing_monthly() -> None:
    start, end = dt(2023, 11, 1), dt(2024, 6, 1)
    stored = times_series([dt(2023, 12, 1), dt(2024, 3, 1)])
    gaps = [(dt(2024, 4, 1), dt(2024, 5, 1))]
    assert missing_ranges(Interval.MO1, start, end, stored, gaps) == [
        (dt(2023, 11, 1), dt(2023, 12, 1)),
        (dt(2024, 1, 1), dt(2024, 3, 1)),
        (dt(2024, 5, 1), dt(2024, 6, 1)),
    ]
    assert (
        missing_ranges(Interval.MO1, start, end, times_series(Interval.MO1.open_times(start, end)))
        == []
    )
    # Mid-month start/end round up to the next month start.
    assert missing_ranges(Interval.MO1, dt(2024, 1, 15), dt(2024, 3, 15), times_series([])) == [
        (dt(2024, 2, 1), dt(2024, 4, 1))
    ]


def test_missing_matches_brute_force_enumeration() -> None:
    interval = Interval.M15
    start, end = dt(2024, 1, 1, 0, 3), dt(2024, 1, 3, 5, 1)
    all_times = list(interval.open_times(start, end))
    stored_times = [t for i, t in enumerate(all_times) if (i * 7) % 5 in {0, 1}]
    gaps = [(dt(2024, 1, 1, 12), dt(2024, 1, 1, 14, 5)), (dt(2024, 1, 2, 20), dt(2024, 1, 2, 21))]
    expected_missing = {
        t for t in all_times if t not in set(stored_times) and not any(a <= t < b for a, b in gaps)
    }
    ranges = missing_ranges(interval, start, end, times_series(stored_times), gaps)
    covered = {t for a, b in ranges for t in interval.open_times(a, b)}
    assert covered == expected_missing
    assert all(a < b and interval.is_aligned(a) and interval.is_aligned(b) for a, b in ranges)
    assert all(ranges[i][1] < ranges[i + 1][0] for i in range(len(ranges) - 1))  # maximal runs


def test_missing_ranges_is_fast_for_700k_candles() -> None:
    start = dt(2020, 1, 1)
    count = 720_000
    end = start + timedelta(minutes=5 * (count + 100))
    # Every 1000th candle missing, plus a missing tail.
    all_times = pl.datetime_range(
        start,
        start + timedelta(minutes=5 * (count - 1)),
        "5m",
        time_zone="UTC",
        time_unit="us",
        eager=True,
    )
    stored = all_times.filter(pl.int_range(all_times.len(), eager=True) % 1000 != 0)
    began = time.perf_counter()
    ranges = missing_ranges(
        Interval.M5, start, end, stored, [(start + timedelta(days=3), start + timedelta(days=4))]
    )
    elapsed = time.perf_counter() - began
    assert elapsed < 2.0
    # 720 single-candle holes, one of them inside the gap, plus the missing tail.
    assert len(ranges) == 720
    assert ranges[0] == (start, start + timedelta(minutes=5))
    assert ranges[-1] == (start + timedelta(minutes=5 * count), end)


# ---------------------------------------------------------------- split_by_month


def test_split_by_month_across_year_boundary() -> None:
    ranges = [(dt(2023, 11, 20), dt(2024, 2, 10)), (dt(2024, 5, 3), dt(2024, 5, 4))]
    assert split_by_month(ranges) == {
        dt(2023, 11, 1): [(dt(2023, 11, 20), dt(2023, 12, 1))],
        dt(2023, 12, 1): [(dt(2023, 12, 1), dt(2024, 1, 1))],
        dt(2024, 1, 1): [(dt(2024, 1, 1), dt(2024, 2, 1))],
        dt(2024, 2, 1): [(dt(2024, 2, 1), dt(2024, 2, 10))],
        dt(2024, 5, 1): [(dt(2024, 5, 3), dt(2024, 5, 4))],
    }
    assert list(split_by_month(ranges)) == sorted(split_by_month(ranges))


def test_split_by_month_edges() -> None:
    assert split_by_month([]) == {}
    # A range ending exactly on a month start stays in the earlier month.
    assert split_by_month([(dt(2024, 1, 31), dt(2024, 2, 1))]) == {
        dt(2024, 1, 1): [(dt(2024, 1, 31), dt(2024, 2, 1))]
    }
    # Several ranges in one month are grouped, and out-of-order input gives ordered keys.
    got = split_by_month(
        [
            (dt(2024, 3, 5), dt(2024, 3, 6)),
            (dt(2024, 1, 5), dt(2024, 1, 6)),
            (dt(2024, 3, 9), dt(2024, 3, 10)),
        ]
    )
    assert list(got) == [dt(2024, 1, 1), dt(2024, 3, 1)]
    assert len(got[dt(2024, 3, 1)]) == 2
