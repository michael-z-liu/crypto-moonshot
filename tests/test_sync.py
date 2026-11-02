import asyncio
import json
import threading
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
import pytest

from crypto_moonshot.data.binance import (
    BinanceClient,
    BinanceError,
    ClientStats,
    IpBannedError,
    RateLimitAbortError,
    RateLimitHit,
    RateLimitHook,
)
from crypto_moonshot.data.store import KlineStore, check_month_file
from crypto_moonshot.data.sync import (
    KlineSource,
    MonthCompleted,
    MonthStarted,
    PageFetched,
    RateLimited,
    SeriesFinished,
    SeriesPlanned,
    SeriesStatus,
    SyncEvent,
    SyncFinished,
    SyncStarted,
    SyncSummary,
    gap_limit,
    sync_klines,
)
from crypto_moonshot.models import KLINE_SCHEMA, Interval

pytestmark = pytest.mark.anyio

H1 = Interval.H1
START = datetime(2024, 1, 1, tzinfo=UTC)
NOW = datetime(2024, 3, 15, 10, 30, tzinfo=UTC)
JAN = datetime(2024, 1, 1, tzinfo=UTC)
FEB = datetime(2024, 2, 1, tzinfo=UTC)
MAR = datetime(2024, 3, 1, tzinfo=UTC)
APR = datetime(2024, 4, 1, tzinfo=UTC)
LAST_CLOSED_END = datetime(2024, 3, 15, 10, tzinfo=UTC)  # floor(NOW): the 10:00 candle is forming
TIMEOUT_S = 5.0  # safety net so a broken test fails instead of hanging

Fetch = tuple[str, datetime, datetime]
FetchHook = Callable[[str, datetime, datetime], Awaitable[None]]


def dt(month: int, day: int = 1, hour: int = 0, year: int = 2024) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def hourly(start: datetime, end: datetime, *, without: Iterable[datetime] = ()) -> list[datetime]:
    skip = set(without)
    return [t for t in H1.open_times(start, end) if t not in skip]


def frame_of(times: Iterable[datetime]) -> pl.DataFrame:
    rows = list(times)
    n = len(rows)
    return pl.DataFrame(
        {
            "open_time": rows,
            "open": [1.0] * n,
            "high": [2.0] * n,
            "low": [0.5] * n,
            "close": [1.5] * n,
            "volume": [10.0] * n,
            "quote_volume": [15.0] * n,
            "trade_count": [3] * n,
            "taker_buy_base_volume": [5.0] * n,
            "taker_buy_quote_volume": [7.5] * n,
        },
        schema=KLINE_SCHEMA,
    )


async def yield_control(times: int = 5) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


class FakeSource:
    """In-memory Binance: hourly candles per symbol, with request and concurrency tracking."""

    def __init__(
        self,
        data: dict[str, list[datetime]],
        *,
        now: datetime = NOW,
        page_size: int = 1000,
        listing_errors: dict[str, Exception] | None = None,
        fetch_hook: FetchHook | None = None,
    ) -> None:
        self.data = data
        self.now = now
        self.page_size = page_size
        self.listing_errors = listing_errors or {}
        self.fetch_hook = fetch_hook
        self.stats = ClientStats()
        self.on_rate_limit: RateLimitHook | None = None
        self.fatal: BinanceError | None = None
        self.fetches: list[Fetch] = []
        self.listing_calls: list[str] = []
        self.in_flight: list[str] = []
        self.max_in_flight = 0
        self.max_symbols_in_flight = 0

    def _enter(self, symbol: str) -> None:
        if self.fatal is not None:
            raise self.fatal
        self.in_flight.append(symbol)
        self.max_in_flight = max(self.max_in_flight, len(self.in_flight))
        self.max_symbols_in_flight = max(self.max_symbols_in_flight, len(set(self.in_flight)))

    async def first_open_time(self, symbol: str, interval: Interval) -> datetime | None:
        assert interval is H1
        self._enter(symbol)
        try:
            self.listing_calls.append(symbol)
            self.stats.requests += 1
            await yield_control()
            if symbol in self.listing_errors:
                raise self.listing_errors[symbol]
            times = self.data.get(symbol, [])
            return times[0] if times else None
        finally:
            self.in_flight.remove(symbol)

    async def fetch_klines(
        self,
        symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
        *,
        on_page: Callable[[int], None] | None = None,
    ) -> pl.DataFrame:
        assert interval is H1
        assert start.tzinfo is not None
        assert end.tzinfo is not None
        self._enter(symbol)
        try:
            self.fetches.append((symbol, start, end))
            await yield_control()
            if self.fetch_hook is not None:
                await self.fetch_hook(symbol, start, end)
            times = [
                t
                for t in self.data.get(symbol, [])
                if start <= t < end and t + interval.length <= self.now
            ]
            pages = [times[i : i + self.page_size] for i in range(0, len(times), self.page_size)]
            for page in pages or [[]]:
                self.stats.requests += 1
                await yield_control(2)
                if on_page is not None:
                    on_page(len(page))
            return frame_of(times)
        finally:
            self.in_flight.remove(symbol)


def full(start: datetime = START, end: datetime = LAST_CLOSED_END) -> list[datetime]:
    return hourly(start, end)


async def run(
    source: FakeSource,
    store: KlineStore,
    symbols: Iterable[str] = ("BTCUSDT",),
    *,
    events: list[SyncEvent] | None = None,
    **kwargs: object,
) -> SyncSummary:
    options: dict[str, object] = {"start": START, "now": source.now, **kwargs}
    sink = events.append if events is not None else None
    async with asyncio.timeout(TIMEOUT_S):
        return await sync_klines(
            source,
            store,
            [(symbol, H1) for symbol in symbols],
            sink=sink,
            **options,  # type: ignore[arg-type]
        )


def stored(store: KlineStore, symbol: str) -> list[datetime]:
    return list(store.stored_open_times(symbol, H1))


def temp_files(root: Path) -> list[Path]:
    return [p for p in root.rglob(".*") if p.is_file()]


@pytest.fixture
def store(tmp_path: Path) -> KlineStore:
    return KlineStore(tmp_path / "data")


# -- protocol --------------------------------------------------------------------------------


async def test_binance_client_and_fake_satisfy_the_source_protocol() -> None:
    async with httpx.AsyncClient() as http:
        client: KlineSource = BinanceClient(http, base_url="https://binance.test")
        fake: KlineSource = FakeSource({})
        assert client.stats.requests == 0
        assert fake.on_rate_limit is None


# -- planning and fetching -------------------------------------------------------------------


async def test_first_sync_fetches_each_month_and_saves_the_listing(store: KlineStore) -> None:
    source = FakeSource({"BTCUSDT": full()})
    summary = await run(source, store)

    assert sorted(source.fetches) == [
        ("BTCUSDT", JAN, FEB),
        ("BTCUSDT", FEB, MAR),
        ("BTCUSDT", MAR, LAST_CLOSED_END),
    ]
    assert source.listing_calls == ["BTCUSDT"]
    assert store.read_meta("BTCUSDT", H1).listing == JAN
    assert store.list_months("BTCUSDT", H1) == [JAN, FEB, MAR]
    assert stored(store, "BTCUSDT") == full()
    for month in (JAN, FEB, MAR):
        assert check_month_file(store.month_path("BTCUSDT", H1, month), now=NOW) == []

    result = summary.series[0]
    assert result.status is SeriesStatus.OK
    assert result.candles_added == len(full())
    assert result.candles_to_fetch == len(full())
    assert result.months_written == result.months_to_fetch == 3
    assert result.gap_candles_found == 0
    assert result.first_open_time == JAN
    assert result.last_open_time == LAST_CLOSED_END - timedelta(hours=1)
    assert result.error is None
    assert summary.aborted is None
    assert summary.outcome == "ok"
    assert summary.requests == source.stats.requests == 4  # 1 listing + 3 months
    assert summary.end == NOW


async def test_second_run_makes_no_requests(store: KlineStore) -> None:
    await run(FakeSource({"BTCUSDT": full(), "ETHUSDT": full()}), store, ["BTCUSDT", "ETHUSDT"])

    again = FakeSource({"BTCUSDT": full(), "ETHUSDT": full()})
    summary = await run(again, store, ["BTCUSDT", "ETHUSDT"])

    assert again.fetches == []
    assert again.listing_calls == []
    assert again.stats.requests == 0
    assert summary.requests == 0
    assert [r.status for r in summary.series] == [SeriesStatus.UP_TO_DATE] * 2
    assert summary.series[0].last_open_time == LAST_CLOSED_END - timedelta(hours=1)
    assert summary.outcome == "ok"


async def test_later_run_fetches_only_new_candles_of_the_current_month(
    store: KlineStore,
) -> None:
    await run(FakeSource({"BTCUSDT": full()}), store)
    later = NOW + timedelta(hours=3)
    source = FakeSource({"BTCUSDT": full(end=dt(3, 15, 13))}, now=later)

    summary = await run(source, store)

    assert source.fetches == [("BTCUSDT", LAST_CLOSED_END, dt(3, 15, 13))]
    assert summary.series[0].candles_added == 3
    assert stored(store, "BTCUSDT") == full(end=dt(3, 15, 13))


async def test_run_across_a_month_boundary_fetches_both_months(store: KlineStore) -> None:
    await run(FakeSource({"BTCUSDT": full()}), store)
    later = datetime(2024, 4, 1, 2, 10, tzinfo=UTC)
    source = FakeSource({"BTCUSDT": full(end=dt(4, 1, 2))}, now=later)

    summary = await run(source, store)

    assert sorted(source.fetches) == [
        ("BTCUSDT", LAST_CLOSED_END, APR),
        ("BTCUSDT", APR, dt(4, 1, 2)),
    ]
    assert summary.series[0].months_written == 2


async def test_only_months_with_missing_candles_are_fetched(store: KlineStore) -> None:
    await run(FakeSource({"BTCUSDT": full()}), store)
    store.month_path("BTCUSDT", H1, FEB).unlink()
    holes = [dt(1, 10, 5), dt(1, 10, 6), dt(1, 20, 0)]
    store.write_month("BTCUSDT", H1, JAN, frame_of(hourly(JAN, FEB, without=holes)))

    source = FakeSource({"BTCUSDT": full()})
    summary = await run(source, store)

    assert sorted(source.fetches) == [
        ("BTCUSDT", dt(1, 10, 5), dt(1, 10, 7)),
        ("BTCUSDT", dt(1, 20, 0), dt(1, 20, 1)),
        ("BTCUSDT", FEB, MAR),
    ]
    assert source.listing_calls == []
    assert summary.series[0].candles_added == 3 + len(hourly(FEB, MAR))
    assert summary.series[0].months_to_fetch == 2
    assert stored(store, "BTCUSDT") == full()


async def test_end_limits_the_range(store: KlineStore) -> None:
    source = FakeSource({"BTCUSDT": full()})
    summary = await run(source, store, end=dt(1, 3))
    assert source.fetches == [("BTCUSDT", JAN, dt(1, 3))]
    assert summary.end == dt(1, 3)
    assert stored(store, "BTCUSDT") == hourly(JAN, dt(1, 3))


async def test_pre_listing_range_is_never_requested(store: KlineStore) -> None:
    listing = dt(2, 10, 12)
    source = FakeSource({"BTCUSDT": full(), "ETHUSDT": full(start=listing)})

    summary = await run(source, store, ["BTCUSDT", "ETHUSDT"])

    eth = sorted((s, e) for symbol, s, e in source.fetches if symbol == "ETHUSDT")
    assert eth == [(listing, MAR), (MAR, LAST_CLOSED_END)]
    assert store.read_meta("ETHUSDT", H1).listing == listing
    assert store.list_months("ETHUSDT", H1) == [FEB, MAR]
    assert summary.series[1].first_open_time == listing
    assert summary.series[1].gap_candles_found == 0  # pre-listing is not a gap
    assert store.read_meta("ETHUSDT", H1).gaps == []

    again = FakeSource({"BTCUSDT": full(), "ETHUSDT": full(start=listing)})
    await run(again, store, ["BTCUSDT", "ETHUSDT"], start=datetime(2020, 1, 1, tzinfo=UTC))
    assert again.listing_calls == []
    assert again.fetches == []


async def test_symbol_without_data_is_no_data(store: KlineStore) -> None:
    source = FakeSource({"BTCUSDT": full(), "NEWUSDT": []})
    summary = await run(source, store, ["BTCUSDT", "NEWUSDT"])
    assert summary.series[1].status is SeriesStatus.NO_DATA
    assert all(symbol != "NEWUSDT" for symbol, _, _ in source.fetches)
    assert summary.outcome == "ok"


async def test_duplicate_series_are_synced_once(store: KlineStore) -> None:
    source = FakeSource({"BTCUSDT": full()})
    summary = await run(source, store, ["BTCUSDT", "BTCUSDT"])
    assert len(summary.series) == 1
    assert len(source.fetches) == 3


async def test_invalid_arguments_are_rejected(store: KlineStore) -> None:
    source = FakeSource({})
    with pytest.raises(ValueError, match="concurrency"):
        await run(source, store, concurrency=0)
    with pytest.raises(ValueError, match="naive"):
        await run(source, store, start=datetime(2024, 1, 1))  # noqa: DTZ001


# -- invalid month files ---------------------------------------------------------------------


def corrupt(path: Path) -> None:
    path.write_bytes(b"not a parquet file")


def without_version_metadata(path: Path) -> None:
    frame_of(hourly(FEB, dt(2, 2))).write_parquet(path)


@pytest.mark.parametrize("damage", [corrupt, without_version_metadata])
async def test_invalid_month_is_refetched_in_full_and_overwritten(
    store: KlineStore, damage: Callable[[Path], None]
) -> None:
    await run(FakeSource({"BTCUSDT": full()}), store)
    path = store.month_path("BTCUSDT", H1, FEB)
    damage(path)
    assert check_month_file(path, now=NOW) != []

    events: list[SyncEvent] = []
    source = FakeSource({"BTCUSDT": full()})
    summary = await run(source, store, events=events)

    assert source.fetches == [("BTCUSDT", FEB, MAR)]
    assert check_month_file(path, now=NOW) == []
    assert stored(store, "BTCUSDT") == full()
    assert summary.series[0].candles_added == len(hourly(FEB, MAR))
    completed = [e for e in events if isinstance(e, MonthCompleted)]
    assert [(e.month, e.written, e.replaced_invalid) for e in completed] == [(FEB, True, True)]


# -- gaps ------------------------------------------------------------------------------------


async def test_gaps_are_recorded_and_not_refetched(store: KlineStore) -> None:
    outage = [dt(2, 10, 5), dt(2, 10, 6), dt(2, 10, 7)]
    data = {"BTCUSDT": hourly(START, LAST_CLOSED_END, without=outage)}

    summary = await run(FakeSource(data), store)

    meta = store.read_meta("BTCUSDT", H1)
    assert meta.gaps == [(dt(2, 10, 5), dt(2, 10, 8))]
    assert summary.series[0].gap_candles_found == 3
    assert summary.series[0].candles_added == len(full()) - 3
    assert summary.series[0].status is SeriesStatus.OK

    again = FakeSource(data)
    second = await run(again, store)
    assert again.fetches == []
    assert second.series[0].status is SeriesStatus.UP_TO_DATE


async def test_a_month_with_no_candles_at_all_is_recorded_as_a_gap(store: KlineStore) -> None:
    data = {"BTCUSDT": hourly(START, LAST_CLOSED_END, without=hourly(FEB, MAR))}
    summary = await run(FakeSource(data), store)
    assert store.list_months("BTCUSDT", H1) == [JAN, MAR]  # nothing written for February
    assert store.read_meta("BTCUSDT", H1).gaps == [(FEB, MAR)]
    assert summary.series[0].months_written == 2
    assert summary.series[0].status is SeriesStatus.OK


async def test_recent_hole_is_not_recorded_and_is_fetched_again(store: KlineStore) -> None:
    # The last two closed candles (08:00 and 09:00) end within 2 x 1h of 10:30.
    recent = [dt(3, 15, 8), dt(3, 15, 9)]
    data = {"BTCUSDT": hourly(START, LAST_CLOSED_END, without=recent)}

    summary = await run(FakeSource(data), store)
    assert store.read_meta("BTCUSDT", H1).gaps == []
    assert summary.series[0].gap_candles_found == 0

    published = FakeSource({"BTCUSDT": full()})
    second = await run(published, store)
    assert published.fetches == [("BTCUSDT", dt(3, 15, 8), LAST_CLOSED_END)]
    assert second.series[0].candles_added == 2
    assert stored(store, "BTCUSDT") == full()


async def test_hole_reaching_now_records_only_its_older_part(store: KlineStore) -> None:
    hole = hourly(dt(3, 15, 5), LAST_CLOSED_END)  # 05:00 to 09:00
    data = {"BTCUSDT": hourly(START, LAST_CLOSED_END, without=hole)}

    summary = await run(FakeSource(data), store)

    assert store.read_meta("BTCUSDT", H1).gaps == [(dt(3, 15, 5), dt(3, 15, 8))]
    assert summary.series[0].gap_candles_found == 3

    again = FakeSource(data)
    await run(again, store)
    assert again.fetches == [("BTCUSDT", dt(3, 15, 8), LAST_CLOSED_END)]


def test_gap_limit_keeps_two_candles_or_five_minutes() -> None:
    now = datetime(2024, 3, 15, 10, 31, tzinfo=UTC)
    assert gap_limit(Interval.H1, now) == dt(3, 15, 8)
    assert gap_limit(Interval.M1, now) == datetime(2024, 3, 15, 10, 26, tzinfo=UTC)
    assert gap_limit(Interval.M5, now) == datetime(2024, 3, 15, 10, 20, tzinfo=UTC)
    assert gap_limit(Interval.D1, now) == dt(3, 13)
    assert gap_limit(Interval.MO1, now) == dt(1, 1)


# -- concurrency -----------------------------------------------------------------------------


async def test_months_of_one_series_run_concurrently(store: KlineStore) -> None:
    source = FakeSource({"BTCUSDT": full()})
    await run(source, store, concurrency=2)
    assert source.max_in_flight == 2


async def test_units_run_concurrently_across_series_within_the_limit(store: KlineStore) -> None:
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    source = FakeSource({symbol: full() for symbol in symbols})

    summary = await run(source, store, symbols, concurrency=4)

    assert source.max_in_flight == 4
    assert source.max_symbols_in_flight > 1
    assert len(source.fetches) == 9
    assert all(r.status is SeriesStatus.OK for r in summary.series)


async def test_concurrency_one_runs_units_one_at_a_time(store: KlineStore) -> None:
    symbols = ["BTCUSDT", "ETHUSDT"]
    source = FakeSource({symbol: full() for symbol in symbols})
    await run(source, store, symbols, concurrency=1)
    assert source.max_in_flight == 1
    # Months are interleaved across series so every series makes progress.
    assert [(symbol, start) for symbol, start, _ in source.fetches] == [
        ("BTCUSDT", JAN),
        ("ETHUSDT", JAN),
        ("BTCUSDT", FEB),
        ("ETHUSDT", FEB),
        ("BTCUSDT", MAR),
        ("ETHUSDT", MAR),
    ]


# -- failures --------------------------------------------------------------------------------


async def test_unknown_symbol_fails_only_that_series(store: KlineStore) -> None:
    invalid = BinanceError(
        "HTTP 400, code -1121: Invalid symbol.", status=400, code=-1121, msg="Invalid symbol."
    )
    source = FakeSource(
        {"BTCUSDT": full(), "ETHUSDT": full()}, listing_errors={"NOPEUSDT": invalid}
    )

    summary = await run(source, store, ["BTCUSDT", "NOPEUSDT", "ETHUSDT"])

    nope = summary.series[1]
    assert nope.symbol == "NOPEUSDT"
    assert nope.status is SeriesStatus.FAILED
    assert nope.error is not None
    assert "-1121" in nope.error
    assert "Invalid symbol." in nope.error
    for index, symbol in ((0, "BTCUSDT"), (2, "ETHUSDT")):
        assert summary.series[index].status is SeriesStatus.OK
        assert stored(store, symbol) == full()
    assert summary.outcome == "failed"
    assert summary.aborted is None


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (BinanceError("HTTP 400, code -1100: Illegal characters.", code=-1100), "-1100"),
        (RuntimeError("boom"), "RuntimeError: boom"),
    ],
)
async def test_error_mid_run_fails_only_that_series(
    store: KlineStore, error: Exception, message: str
) -> None:
    async def hook(symbol: str, start: datetime, end: datetime) -> None:
        if (symbol, start) == ("ETHUSDT", FEB):
            raise error

    source = FakeSource({"BTCUSDT": full(), "ETHUSDT": full()}, fetch_hook=hook)
    events: list[SyncEvent] = []

    summary = await run(source, store, ["BTCUSDT", "ETHUSDT"], concurrency=1, events=events)

    btc, eth = summary.series
    assert btc.status is SeriesStatus.OK
    assert stored(store, "BTCUSDT") == full()
    assert eth.status is SeriesStatus.FAILED
    assert eth.error is not None
    assert message in eth.error
    # ETH's January was written before the failure; its March was cancelled, never fetched.
    assert [s for symbol, s, _ in source.fetches if symbol == "ETHUSDT"] == [JAN, FEB]
    assert eth.months_written == 1
    assert store.list_months("ETHUSDT", H1) == [JAN]
    assert eth.last_open_time == FEB - timedelta(hours=1)
    eth_events = [e for e in events if _symbol_of(e) == "ETHUSDT"]
    assert isinstance(eth_events[-1], SeriesFinished)


async def test_failure_cancels_the_series_units_in_flight(store: KlineStore) -> None:
    march_started = asyncio.Event()
    never = asyncio.Event()

    async def hook(symbol: str, start: datetime, end: datetime) -> None:
        if symbol != "ETHUSDT":
            return
        if start == MAR:
            march_started.set()
            await never.wait()  # would hang the run unless cancelled
        if start == FEB:
            await march_started.wait()
            raise BinanceError("HTTP 400: bad request", status=400)

    source = FakeSource({"BTCUSDT": full(), "ETHUSDT": full()}, fetch_hook=hook)
    summary = await run(source, store, ["BTCUSDT", "ETHUSDT"], concurrency=8)

    assert summary.series[0].status is SeriesStatus.OK
    assert summary.series[1].status is SeriesStatus.FAILED
    assert source.in_flight == []


class FailingStore(KlineStore):
    """A store whose month writes for one symbol fail like a full disk."""

    def merge_month(
        self, symbol: str, interval: Interval, month: datetime, frame: pl.DataFrame
    ) -> int:
        if symbol == "ETHUSDT":
            raise OSError(28, "No space left on device")
        return super().merge_month(symbol, interval, month, frame)


async def test_write_error_fails_only_that_series(tmp_path: Path) -> None:
    store = FailingStore(tmp_path / "data")
    source = FakeSource({"BTCUSDT": full(), "ETHUSDT": full()})

    summary = await run(source, store, ["BTCUSDT", "ETHUSDT"])

    btc, eth = summary.series
    assert btc.status is SeriesStatus.OK
    assert stored(store, "BTCUSDT") == full()
    assert eth.status is SeriesStatus.FAILED
    assert eth.error is not None
    assert "No space left on device" in eth.error
    assert store.list_months("ETHUSDT", H1) == []
    assert temp_files(store.root) == []


@pytest.mark.parametrize(
    "fatal",
    [
        RateLimitAbortError("aborting after 3 rate-limit (HTTP 429) responses", status=429),
        IpBannedError("HTTP 418: IP banned by Binance; stop all requests", status=418),
    ],
)
async def test_fatal_error_aborts_all_series_and_keeps_written_months(
    store: KlineStore, fatal: BinanceError
) -> None:
    source = FakeSource({"BTCUSDT": full(), "ETHUSDT": full()})

    async def hook(symbol: str, start: datetime, end: datetime) -> None:
        if (symbol, start) == ("BTCUSDT", FEB):
            source.fatal = fatal
            raise fatal

    source.fetch_hook = hook
    events: list[SyncEvent] = []

    summary = await run(source, store, ["BTCUSDT", "ETHUSDT"], concurrency=1, events=events)

    assert summary.aborted == str(fatal)
    assert summary.outcome == "aborted"
    assert [r.status for r in summary.series] == [SeriesStatus.ABORTED] * 2
    assert all(r.error == str(fatal) for r in summary.series)
    assert [r.months_written for r in summary.series] == [1, 1]
    assert len(source.fetches) == 3  # BTC Jan, ETH Jan, BTC Feb; nothing after the abort
    for symbol in ("BTCUSDT", "ETHUSDT"):
        assert store.list_months(symbol, H1) == [JAN]
        assert check_month_file(store.month_path(symbol, H1, JAN), now=NOW) == []
    assert isinstance(events[-1], SyncFinished)
    assert events[-1].summary.aborted == str(fatal)
    assert temp_files(store.root) == []


async def test_fatal_error_while_planning_aborts_everything(store: KlineStore) -> None:
    banned = IpBannedError("HTTP 418: IP banned by Binance; stop all requests", status=418)
    source = FakeSource({"BTCUSDT": full(), "ETHUSDT": full()}, listing_errors={"BTCUSDT": banned})

    summary = await run(source, store, ["BTCUSDT", "ETHUSDT"], concurrency=1)

    assert summary.aborted == str(banned)
    assert [r.status for r in summary.series] == [SeriesStatus.ABORTED] * 2
    assert source.fetches == []


async def test_abort_cancels_units_in_flight(store: KlineStore) -> None:
    never = asyncio.Event()
    btc_started = asyncio.Event()
    source = FakeSource({"BTCUSDT": full(), "ETHUSDT": full()})

    async def hook(symbol: str, start: datetime, end: datetime) -> None:
        if symbol == "BTCUSDT":
            btc_started.set()
            await never.wait()
        else:
            await btc_started.wait()
            source.fatal = RateLimitAbortError("too many 429s", status=429)
            raise source.fatal

    source.fetch_hook = hook
    summary = await run(source, store, ["BTCUSDT", "ETHUSDT"], concurrency=8)

    assert summary.aborted == "too many 429s"
    assert source.in_flight == []


# -- cancellation ----------------------------------------------------------------------------


async def test_cancelling_the_caller_stops_promptly(store: KlineStore) -> None:
    never = asyncio.Event()
    january_done = asyncio.Event()

    async def hook(symbol: str, start: datetime, end: datetime) -> None:
        if start != JAN:
            await never.wait()

    def sink(event: SyncEvent) -> None:
        events.append(event)
        if isinstance(event, MonthCompleted) and event.month == JAN:
            january_done.set()

    events: list[SyncEvent] = []
    source = FakeSource({"BTCUSDT": full()}, fetch_hook=hook)
    task = asyncio.create_task(
        sync_klines(source, store, [("BTCUSDT", H1)], start=START, now=NOW, sink=sink)
    )
    async with asyncio.timeout(TIMEOUT_S):
        await january_done.wait()
        await yield_control(20)  # let February and March get stuck in their fetches
        assert len(source.in_flight) == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert source.in_flight == []
    assert store.list_months("BTCUSDT", H1) == [JAN]
    assert check_month_file(store.month_path("BTCUSDT", H1, JAN), now=NOW) == []
    assert temp_files(store.root) == []
    assert not any(isinstance(e, SyncFinished) for e in events)


class BlockingStore(KlineStore):
    """A store whose first month merge blocks (in its worker thread) until released."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.entered = threading.Event()
        self.release = threading.Event()

    def merge_month(
        self, symbol: str, interval: Interval, month: datetime, frame: pl.DataFrame
    ) -> int:
        self.entered.set()
        assert self.release.wait(TIMEOUT_S)
        return super().merge_month(symbol, interval, month, frame)


async def test_cancel_during_a_write_waits_for_the_write_to_finish(tmp_path: Path) -> None:
    store = BlockingStore(tmp_path / "data")
    source = FakeSource({"BTCUSDT": hourly(JAN, FEB)})
    task = asyncio.create_task(
        sync_klines(source, store, [("BTCUSDT", H1)], start=START, end=FEB, now=NOW)
    )
    async with asyncio.timeout(TIMEOUT_S):
        assert await asyncio.to_thread(store.entered.wait, TIMEOUT_S)
        task.cancel()
        await yield_control(10)
        assert not task.done()  # still waiting for the write in progress
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert check_month_file(store.month_path("BTCUSDT", H1, JAN), now=NOW) == []
    assert stored(store, "BTCUSDT") == hourly(JAN, FEB)
    assert temp_files(store.root) == []


# -- events ----------------------------------------------------------------------------------


def _symbol_of(event: SyncEvent) -> str | None:
    if isinstance(event, SeriesFinished):
        return event.result.symbol
    return getattr(event, "symbol", None)


async def test_events_arrive_in_order_and_add_up(store: KlineStore) -> None:
    symbols = ["BTCUSDT", "ETHUSDT"]
    outage = [dt(2, 10, 5), dt(2, 10, 6)]
    data = {
        "BTCUSDT": hourly(START, LAST_CLOSED_END, without=outage),
        "ETHUSDT": full(start=dt(1, 20)),
    }
    events: list[SyncEvent] = []

    summary = await run(FakeSource(data, page_size=100), store, symbols, events=events)

    assert isinstance(events[0], SyncStarted)
    assert events[0].series == (("BTCUSDT", H1), ("ETHUSDT", H1))
    assert isinstance(events[-1], SyncFinished)
    assert events[-1].summary == summary
    for symbol, result in zip(symbols, summary.series, strict=True):
        mine = [e for e in events if _symbol_of(e) == symbol]
        planned = mine[0]
        assert isinstance(planned, SeriesPlanned)
        assert mine[-1] == SeriesFinished(result)
        assert sum(isinstance(e, SeriesFinished) for e in mine) == 1
        assert len(planned.months) == planned.months_to_fetch == result.months_to_fetch == 3

        pages = [e for e in mine if isinstance(e, PageFetched)]
        completed = [e for e in mine if isinstance(e, MonthCompleted)]
        assert len(pages) > len(completed)  # several pages per month
        assert sum(e.candles for e in pages) == sum(e.candles_fetched for e in completed)
        assert sum(e.candles for e in pages) == result.candles_added
        assert sum(e.candles_fetched + e.candles_missing for e in completed) == (
            planned.candles_to_fetch
        )
        assert sum(e.gap_candles for e in completed) == result.gap_candles_found
        assert sorted(e.months_done for e in completed) == [1, 2, 3]
        assert all(e.months_total == 3 for e in completed)

        for month in planned.months:
            in_month = [
                i
                for i, e in enumerate(mine)
                if isinstance(e, MonthStarted | PageFetched | MonthCompleted) and e.month == month
            ]
            assert isinstance(mine[in_month[0]], MonthStarted)
            assert isinstance(mine[in_month[-1]], MonthCompleted)
            assert all(isinstance(mine[i], PageFetched) for i in in_month[1:-1])

    btc_planned = next(e for e in events if isinstance(e, SeriesPlanned) and e.symbol == "BTCUSDT")
    assert btc_planned.start == START
    assert btc_planned.end == LAST_CLOSED_END
    assert btc_planned.candles_stored == 0
    assert summary.series[0].gap_candles_found == 2


async def test_planned_counts_stored_candles(store: KlineStore) -> None:
    await run(FakeSource({"BTCUSDT": full()}), store, end=FEB)
    events: list[SyncEvent] = []
    await run(FakeSource({"BTCUSDT": full()}), store, events=events)
    planned = next(e for e in events if isinstance(e, SeriesPlanned))
    assert planned.candles_stored == len(hourly(JAN, FEB))
    assert planned.candles_to_fetch == len(hourly(FEB, LAST_CLOSED_END))
    assert planned.months == (FEB, MAR)


async def test_a_raising_sink_does_not_break_the_sync(store: KlineStore) -> None:
    calls: list[SyncEvent] = []

    def sink(event: SyncEvent) -> None:
        calls.append(event)
        raise RuntimeError("renderer bug")

    source = FakeSource({"BTCUSDT": full()})
    summary = await sync_klines(source, store, [("BTCUSDT", H1)], start=START, now=NOW, sink=sink)

    assert summary.series[0].status is SeriesStatus.OK
    assert stored(store, "BTCUSDT") == full()
    assert isinstance(calls[-1], SyncFinished)


async def test_rate_limit_hits_become_events_and_the_hook_is_restored(
    store: KlineStore,
) -> None:
    hit = RateLimitHit(hits=1, max_hits=3, retry_after=7.0, aborting=False)
    previous: list[RateLimitHit] = []
    source = FakeSource({"BTCUSDT": full()})

    async def hook(symbol: str, start: datetime, end: datetime) -> None:
        if start == FEB:
            assert source.on_rate_limit is not None
            source.on_rate_limit(hit)
            source.stats.rate_limit_hits += 1

    source.fetch_hook = hook
    source.on_rate_limit = previous.append
    events: list[SyncEvent] = []

    summary = await run(source, store, events=events)

    assert [e for e in events if isinstance(e, RateLimited)] == [
        RateLimited(hits=1, max_hits=3, retry_after=7.0, aborting=False)
    ]
    assert previous == [hit]  # an existing hook is chained, then restored
    assert source.on_rate_limit == previous.append
    assert summary.rate_limit_hits == 1


# -- summary ---------------------------------------------------------------------------------


async def test_summary_to_dict_is_json_serialisable(store: KlineStore) -> None:
    invalid = BinanceError("HTTP 400, code -1121: Invalid symbol.", code=-1121)
    source = FakeSource({"BTCUSDT": full()}, listing_errors={"NOPEUSDT": invalid})
    summary = await run(source, store, ["BTCUSDT", "NOPEUSDT"])

    document = json.loads(json.dumps(summary.to_dict()))

    assert document["outcome"] == "failed"
    assert document["aborted"] is None
    assert document["start"] == "2024-01-01T00:00:00+00:00"
    assert document["now"] == "2024-03-15T10:30:00+00:00"
    assert document["requests"] == summary.requests
    assert document["candles_added"] == len(full())
    btc, nope = document["series"]
    assert btc == {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "status": "ok",
        "months_to_fetch": 3,
        "candles_to_fetch": len(full()),
        "candles_added": len(full()),
        "months_written": 3,
        "gap_candles_found": 0,
        "first_open_time": "2024-01-01T00:00:00+00:00",
        "last_open_time": "2024-03-15T09:00:00+00:00",
        "error": None,
    }
    assert nope["status"] == "failed"
    assert "-1121" in nope["error"]
    assert nope["first_open_time"] is None
