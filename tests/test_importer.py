import hashlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from kline_fakes import JAN_1, candle_frame, hours

from crypto_moonshot.data.importer import (
    ImportAction,
    ImportReport,
    ImportSourceError,
    MetaAction,
    import_store,
)
from crypto_moonshot.data.status import store_report
from crypto_moonshot.data.store import SCHEMA_VERSION_KEY, KlineStore, SeriesMeta
from crypto_moonshot.models import SCHEMA_VERSION, Interval

H1 = Interval.H1
NOW = datetime(2025, 6, 1, tzinfo=UTC)
JAN = JAN_1
FEB = datetime(2024, 2, 1, tzinfo=UTC)


def put(
    root: Path,
    frame: pl.DataFrame,
    *,
    interval: str = "1h",
    symbol: str = "BTCUSDT",
    name: str = "2024-01.parquet",
    version: str | None = str(SCHEMA_VERSION),
) -> Path:
    """Write an arbitrary (possibly bad) month file at the standard location."""
    path = root / "klines" / "spot" / f"interval={interval}" / f"symbol={symbol}" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, metadata={} if version is None else {SCHEMA_VERSION_KEY: version})
    return path


def snapshot(root: Path) -> dict[str, str]:
    """Relative path to content hash of every file under `root`."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return tmp_path / "source"


@pytest.fixture
def dest(tmp_path: Path) -> KlineStore:
    return KlineStore(tmp_path / "dest")


def run(source: Path, dest: KlineStore, *, dry_run: bool = False) -> ImportReport:
    return import_store(source, dest, now=NOW, dry_run=dry_run)


def only(report: ImportReport) -> tuple[ImportAction, int, tuple[str, ...]]:
    (item,) = report.files
    return item.action, item.candles_added, item.problems


# -- valid imports ---------------------------------------------------------------------------


def test_valid_source_is_imported_into_an_empty_destination(source: Path, dest: KlineStore) -> None:
    src = KlineStore(source)
    src.write_month("BTCUSDT", H1, JAN, candle_frame(hours(JAN, 48)))
    src.write_month("BTCUSDT", H1, FEB, candle_frame(hours(FEB, 24)))
    src.write_month("ETHUSDT", Interval.D1, JAN, candle_frame([JAN, JAN + timedelta(days=1)]))
    gap = (JAN + timedelta(hours=5), JAN + timedelta(hours=7))
    src.write_meta("BTCUSDT", H1, SeriesMeta(listing=JAN, gaps=[gap]))

    report = run(source, dest)

    assert [item.action for item in report.files] == [ImportAction.IMPORTED] * 3
    assert report.totals() == {
        "files": 3,
        "imported": 3,
        "merged": 0,
        "replaced": 0,
        "unchanged": 0,
        "skipped": 0,
        "candles_added": 48 + 24 + 2,
    }
    for symbol, interval, month in [("BTCUSDT", H1, JAN), ("BTCUSDT", H1, FEB)]:
        assert dest.read_month(symbol, interval, month).equals(
            src.read_month(symbol, interval, month)
        )
    assert dest.read_month("ETHUSDT", Interval.D1, JAN).height == 2
    assert (
        dest.stored_open_times("BTCUSDT", H1).to_list()
        == src.stored_open_times("BTCUSDT", H1).to_list()
    )
    assert dest.read_meta("BTCUSDT", H1) == SeriesMeta(listing=JAN, gaps=[gap])
    assert [report_.to_dict() for report_ in store_report(dest, [], [], now=NOW)] == [
        report_.to_dict() for report_ in store_report(src, [], [], now=NOW)
    ]
    series = {(s.symbol, s.interval): s for s in report.series}
    assert series[("BTCUSDT", H1)].imported == 2
    assert series[("BTCUSDT", H1)].meta is MetaAction.UPDATED
    assert series[("ETHUSDT", Interval.D1)].meta is MetaAction.NONE


def test_hidden_and_non_parquet_files_are_ignored(source: Path, dest: KlineStore) -> None:
    path = put(source, candle_frame(hours(JAN, 3)))
    (path.parent / ".2024-02.parquet.abc.tmp").write_bytes(b"partial")
    (path.parent / ".hidden.parquet").write_bytes(b"x")
    (path.parent / "notes.txt").write_text("hi")

    report = run(source, dest)

    assert [item.path for item in report.files] == [
        "klines/spot/interval=1h/symbol=BTCUSDT/2024-01.parquet"
    ]


# -- skipped files ---------------------------------------------------------------------------


def _bad_path(source: Path) -> Path:
    path = source / "klines" / "spot" / "interval=1h" / "foo.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    candle_frame(hours(JAN, 3)).write_parquet(path)
    return path


def _too_deep(source: Path) -> Path:
    path = (
        source / "klines" / "spot" / "extra" / "interval=1h" / "symbol=BTCUSDT" / "2024-01.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    candle_frame(hours(JAN, 3)).write_parquet(path, metadata={SCHEMA_VERSION_KEY: "1"})
    return path


def _corrupt(source: Path) -> Path:
    path = put(source, candle_frame(hours(JAN, 100)))
    path.write_bytes(path.read_bytes()[:-20])
    return path


def _frame(times: list[datetime]) -> pl.DataFrame:
    return candle_frame(times)


CASES: list[tuple[str, Callable[[Path], Path], str]] = [
    ("bad path", _bad_path, "bad path"),
    ("too deep", _too_deep, "bad path"),
    (
        "missing version",
        lambda r: put(r, _frame(hours(JAN, 3)), version=None),
        "moonshot.schema_version",
    ),
    ("wrong version", lambda r: put(r, _frame(hours(JAN, 3)), version="2"), "schema version '2'"),
    (
        "missing column",
        lambda r: put(r, _frame(hours(JAN, 3)).drop("high")),
        "missing columns: high",
    ),
    (
        "wrong dtype",
        lambda r: put(r, _frame(hours(JAN, 3)).with_columns(pl.col("trade_count").cast(pl.Int32))),
        "trade_count",
    ),
    (
        "naive time",
        lambda r: put(
            r, _frame(hours(JAN, 3)).with_columns(pl.col("open_time").dt.replace_time_zone(None))
        ),
        "Datetime(us, UTC)",
    ),
    (
        "non-utc time",
        lambda r: put(
            r,
            _frame(hours(JAN, 3)).with_columns(
                pl.col("open_time").dt.convert_time_zone("Asia/Tokyo")
            ),
        ),
        "Datetime(us, UTC)",
    ),
    ("out of month", lambda r: put(r, _frame([JAN - timedelta(hours=1), JAN])), "outside month"),
    ("misaligned", lambda r: put(r, _frame([JAN + timedelta(minutes=5)])), "not aligned to 1h"),
    ("duplicate", lambda r: put(r, _frame([JAN, JAN])), "duplicate"),
    ("corrupt", _corrupt, "unreadable parquet file"),
]


@pytest.mark.parametrize(
    ("make", "reason"), [(c[1], c[2]) for c in CASES], ids=[c[0] for c in CASES]
)
def test_each_failing_check_skips_the_file_with_a_reason(
    source: Path, dest: KlineStore, make: Callable[[Path], Path], reason: str
) -> None:
    make(source)
    before = snapshot(source)

    report = run(source, dest)

    action, added, problems = only(report)
    assert action is ImportAction.SKIPPED
    assert added == 0
    assert problems
    assert any(reason in problem for problem in problems), problems
    assert not (dest.root / "klines").exists()
    assert snapshot(source) == before


def test_still_open_candle_is_skipped(source: Path, dest: KlineStore) -> None:
    put(source, candle_frame(hours(JAN, 3)))

    closed = import_store(source, dest, now=JAN + timedelta(hours=3), dry_run=True)
    still_open = import_store(source, dest, now=JAN + timedelta(hours=3, seconds=-1), dry_run=True)

    assert only(closed)[0] is ImportAction.IMPORTED
    action, _, problems = only(still_open)
    assert action is ImportAction.SKIPPED
    assert "not closed yet" in problems[0]


def test_valid_files_import_alongside_skipped_ones(source: Path, dest: KlineStore) -> None:
    KlineStore(source).write_month("BTCUSDT", H1, JAN, candle_frame(hours(JAN, 10)))
    put(source, candle_frame(hours(FEB, 3)).drop("low"), name="2024-02.parquet")
    _bad_path(source)
    put(source, candle_frame(hours(JAN, 5)), symbol="ETHUSDT", version="9")

    report = run(source, dest)

    assert report.totals()["imported"] == 1
    assert report.totals()["skipped"] == 3
    assert dest.stored_open_times("BTCUSDT", H1).to_list() == hours(JAN, 10)
    assert dest.list_series() == [("BTCUSDT", H1)]
    series = {(s.symbol, s.interval): s for s in report.series}
    assert (series[("BTCUSDT", H1)].imported, series[("BTCUSDT", H1)].skipped) == (1, 1)
    assert series[("ETHUSDT", H1)].skipped == 1
    skipped = {item.path: item for item in report.files if item.action is ImportAction.SKIPPED}
    assert skipped["klines/spot/interval=1h/foo.parquet"].symbol is None


# -- existing months -------------------------------------------------------------------------


def test_merge_keeps_existing_rows_and_counts_added(source: Path, dest: KlineStore) -> None:
    dest.write_month("BTCUSDT", H1, JAN, candle_frame(hours(JAN, 10)))
    put(source, candle_frame(hours(JAN, 15)).with_columns(close=pl.lit(99.0)))

    report = run(source, dest)

    assert only(report)[:2] == (ImportAction.MERGED, 5)
    merged = dest.read_month("BTCUSDT", H1, JAN)
    assert merged["open_time"].to_list() == hours(JAN, 15)
    assert merged["close"].to_list() == [1.5] * 10 + [99.0] * 5


def test_nothing_new_is_unchanged_and_the_file_is_not_rewritten(
    source: Path, dest: KlineStore
) -> None:
    dest.write_month("BTCUSDT", H1, JAN, candle_frame(hours(JAN, 10)))
    put(source, candle_frame(hours(JAN, 4)))
    before = snapshot(dest.root)

    report = run(source, dest)

    assert only(report)[:2] == (ImportAction.UNCHANGED, 0)
    assert snapshot(dest.root) == before


def test_invalid_destination_month_is_replaced(source: Path, dest: KlineStore) -> None:
    put(dest.root, candle_frame(hours(JAN, 3)).drop("high"))  # fails the schema check
    put(source, candle_frame(hours(JAN, 6)))

    report = run(source, dest)

    assert only(report)[:2] == (ImportAction.REPLACED, 6)
    assert dest.read_month("BTCUSDT", H1, JAN)["open_time"].to_list() == hours(JAN, 6)
    assert dest.invalid_months("BTCUSDT", H1, now=NOW) == {}


def test_corrupt_destination_month_is_replaced(source: Path, dest: KlineStore) -> None:
    target = dest.month_path("BTCUSDT", H1, JAN)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"garbage")
    put(source, candle_frame(hours(JAN, 6)))

    assert only(run(source, dest))[:2] == (ImportAction.REPLACED, 6)
    assert dest.stored_open_times("BTCUSDT", H1).len() == 6


# -- _meta.json ------------------------------------------------------------------------------


def _gap(start_hour: int, end_hour: int) -> tuple[datetime, datetime]:
    return JAN + timedelta(hours=start_hour), JAN + timedelta(hours=end_hour)


def test_meta_gaps_are_unioned_and_destination_listing_is_kept(
    source: Path, dest: KlineStore
) -> None:
    KlineStore(source).write_month("BTCUSDT", H1, JAN, candle_frame(hours(JAN, 24)))
    KlineStore(source).write_meta(
        "BTCUSDT", H1, SeriesMeta(listing=JAN, gaps=[_gap(2, 4), _gap(10, 12)])
    )
    dest.write_meta("BTCUSDT", H1, SeriesMeta(listing=JAN + timedelta(hours=1), gaps=[_gap(3, 6)]))

    report = run(source, dest)

    assert report.series[0].meta is MetaAction.UPDATED
    meta = dest.read_meta("BTCUSDT", H1)
    assert meta.listing == JAN + timedelta(hours=1)
    assert meta.gaps == [_gap(2, 6), _gap(10, 12)]


def test_meta_listing_is_taken_when_the_destination_has_none(
    source: Path, dest: KlineStore
) -> None:
    KlineStore(source).write_month("BTCUSDT", H1, JAN, candle_frame(hours(JAN, 24)))
    KlineStore(source).write_meta("BTCUSDT", H1, SeriesMeta(listing=JAN))
    dest.write_meta("BTCUSDT", H1, SeriesMeta(gaps=[_gap(3, 6)]))

    run(source, dest)

    assert dest.read_meta("BTCUSDT", H1) == SeriesMeta(listing=JAN, gaps=[_gap(3, 6)])


def test_meta_already_contained_is_unchanged(source: Path, dest: KlineStore) -> None:
    for store in (KlineStore(source), dest):
        store.write_month("BTCUSDT", H1, JAN, candle_frame(hours(JAN, 24)))
        store.write_meta("BTCUSDT", H1, SeriesMeta(listing=JAN, gaps=[_gap(3, 6)]))

    assert run(source, dest).series[0].meta is MetaAction.UNCHANGED


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"schema_version": 2, "listing": null, "gaps": []}',
        '{"listing": null, "gaps": []}',
        '{"schema_version": 1, "listing": "yesterday", "gaps": []}',
        '{"schema_version": 1, "listing": null, "gaps": [["2024-01-01T00:00:00+00:00"]]}',
        '{"schema_version": 1, "listing": null, "gaps": "none"}',
    ],
)
def test_invalid_source_meta_is_skipped_but_the_data_still_imports(
    source: Path, dest: KlineStore, text: str
) -> None:
    KlineStore(source).write_month("BTCUSDT", H1, JAN, candle_frame(hours(JAN, 24)))
    (KlineStore(source).series_dir("BTCUSDT", H1) / "_meta.json").write_text(text)
    dest.write_meta("BTCUSDT", H1, SeriesMeta(listing=JAN))

    report = run(source, dest)

    (series,) = report.series
    assert series.imported == 1
    assert series.meta is MetaAction.SKIPPED
    assert "_meta.json" in series.meta_problems[0]
    assert dest.read_meta("BTCUSDT", H1) == SeriesMeta(listing=JAN)
    assert dest.stored_open_times("BTCUSDT", H1).len() == 24


# -- dry run and the source ------------------------------------------------------------------


def _populate(source: Path, dest: KlineStore) -> None:
    src = KlineStore(source)
    src.write_month("BTCUSDT", H1, JAN, candle_frame(hours(JAN, 20)))  # merge, 10 new
    src.write_month("BTCUSDT", H1, FEB, candle_frame(hours(FEB, 5)))  # import
    src.write_month("ETHUSDT", H1, JAN, candle_frame(hours(JAN, 5)))  # replace
    src.write_month("SOLUSDT", H1, JAN, candle_frame(hours(JAN, 5)))  # unchanged
    src.write_meta("BTCUSDT", H1, SeriesMeta(listing=JAN, gaps=[_gap(30, 32)]))
    put(source, candle_frame(hours(JAN, 3)).drop("low"), symbol="XRPUSDT")  # skipped
    dest.write_month("BTCUSDT", H1, JAN, candle_frame(hours(JAN, 10)))
    dest.write_month("SOLUSDT", H1, JAN, candle_frame(hours(JAN, 5)))
    put(dest.root, candle_frame(hours(JAN, 2)).drop("low"), symbol="ETHUSDT")


def test_dry_run_writes_nothing_and_reports_what_a_real_run_does(
    source: Path, dest: KlineStore
) -> None:
    _populate(source, dest)
    dest_before, source_before = snapshot(dest.root), snapshot(source)

    dry = run(source, dest, dry_run=True)

    assert snapshot(dest.root) == dest_before
    assert snapshot(source) == source_before
    assert dry.dry_run
    real = run(source, dest)
    assert snapshot(dest.root) != dest_before
    assert not real.dry_run
    assert [(i.path, i.action, i.candles_added, i.problems) for i in dry.files] == [
        (i.path, i.action, i.candles_added, i.problems) for i in real.files
    ]
    assert dry.series == real.series
    assert [(i.action, i.candles_added) for i in real.files if "BTCUSDT" in i.path] == [
        (ImportAction.MERGED, 10),
        (ImportAction.IMPORTED, 5),
    ]
    assert dry.totals() == {
        "files": 5,
        "imported": 1,
        "merged": 1,
        "replaced": 1,
        "unchanged": 1,
        "skipped": 1,
        "candles_added": 10 + 5 + 5,
    }
    # Importing the same source again changes nothing.
    after = snapshot(dest.root)
    again = run(source, dest)
    assert again.candles_added == 0
    assert again.totals()["unchanged"] == 4
    assert snapshot(dest.root) == after


def test_source_is_byte_identical_after_a_real_import(source: Path, dest: KlineStore) -> None:
    _populate(source, dest)
    before = snapshot(source)

    run(source, dest)

    assert snapshot(source) == before


# -- usage errors and report format ----------------------------------------------------------


def test_missing_source_folder(tmp_path: Path, dest: KlineStore) -> None:
    with pytest.raises(ImportSourceError, match="does not exist"):
        run(tmp_path / "nope", dest)


def test_source_without_klines_folder_hints_at_the_parent(tmp_path: Path, dest: KlineStore) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ImportSourceError, match="no klines/ folder"):
        run(empty, dest)
    other = tmp_path / "other"
    (other / "klines" / "spot").mkdir(parents=True)
    with pytest.raises(ImportSourceError, match=f"did you mean {other.resolve()}"):
        run(other / "klines", dest)
    with pytest.raises(ImportSourceError, match=f"did you mean {other.resolve()}"):
        run(other / "klines" / "spot", dest)


def test_source_that_is_a_file(tmp_path: Path, dest: KlineStore) -> None:
    path = tmp_path / "file.txt"
    path.write_text("x")
    with pytest.raises(ImportSourceError, match="not a folder"):
        run(path, dest)


def test_source_same_as_destination(source: Path) -> None:
    put(source, candle_frame(hours(JAN, 3)))
    with pytest.raises(ImportSourceError, match="is the destination"):
        run(source, KlineStore(source))
    with pytest.raises(ImportSourceError, match="is the destination"):
        run(source / ".." / "source", KlineStore(source))


def test_source_with_klines_but_no_files_gives_an_empty_report(
    source: Path, dest: KlineStore
) -> None:
    (source / "klines").mkdir(parents=True)
    report = run(source, dest)
    assert report.files == ()
    assert report.series == ()


def test_report_to_dict_is_json_serialisable(source: Path, dest: KlineStore) -> None:
    import json

    _populate(source, dest)
    document = json.loads(json.dumps(run(source, dest, dry_run=True).to_dict()))

    assert set(document) == {"source", "destination", "dry_run", "totals", "series", "files"}
    assert document["dry_run"] is True
    assert document["totals"]["merged"] == 1
    assert set(document["files"][0]) == {
        "path",
        "symbol",
        "interval",
        "month",
        "action",
        "candles_added",
        "problems",
    }
    merged = next(f for f in document["files"] if f["action"] == "merged")
    assert (merged["symbol"], merged["interval"], merged["month"]) == ("BTCUSDT", "1h", "2024-01")
    assert set(document["series"][0]) >= {"symbol", "interval", "imported", "skipped", "meta"}
