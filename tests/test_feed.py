from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from kline_fakes import FakeBinance
from store_fakes import kline_frame, write_series
from stream_fakes import FakeStreamServer, kline_message, until

from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.data.stream import StreamConfig
from crypto_moonshot.live.feed import (
    Backfilled,
    FeedConnected,
    FeedDisconnected,
    FeedEvent,
    LiveFeed,
    Seeded,
    StaleSymbolsChanged,
    SubscriptionFailed,
    Tick,
)
from crypto_moonshot.models import Interval

pytestmark = pytest.mark.anyio

M5 = Interval.M5
STEP = M5.length
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def at(k: int) -> datetime:
    """Open time of the k-th 5m candle after T0 (k may be negative)."""
    return T0 + k * STEP


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class Harness:
    def __init__(
        self,
        server: FakeStreamServer,
        symbols: list[str],
        rest: FakeBinance,
        *,
        warmup: int = 4,
        data_root: Path | None = None,
        tick_timeout: float = 0.15,
        clock_now: datetime = T0,
        stream_kwargs: dict[str, object] | None = None,
    ) -> None:
        self.server = server
        self.clock = Clock(clock_now)
        self.events: list[FeedEvent] = []
        self.ticks: list[Tick] = []
        config = StreamConfig(
            base_url=server.url,
            subscribe_gap=0.0,
            backoff_base=0.01,
            backoff_max=0.05,
            **(stream_kwargs or {}),  # type: ignore[arg-type]
        )
        self.feed = LiveFeed(
            {M5: symbols},
            {M5: warmup},
            client=rest,
            stream_config=config,
            data_root=data_root,
            tick_timeout=tick_timeout,
            clock=self.clock,
            on_event=self.events.append,
        )
        self.symbols = symbols
        self._task = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        async for tick in self.feed.ticks():
            self.ticks.append(tick)

    async def ready(self) -> None:
        """Seeded and subscribed."""
        await until(lambda: any(isinstance(e, Seeded) for e in self.events))
        await until(lambda: len(self.server.connections) >= 1 and self.server.subscribed(0) != [])
        await until(lambda: any(isinstance(e, FeedConnected) for e in self.events))

    async def close(self) -> None:
        await self.feed.aclose()
        await self._task

    async def send(self, symbol: str, k: int, price: float = 100.0, *, closed: bool = True) -> None:
        await self.server.send(kline_message(symbol, M5, at(k), closed=closed, price=price))

    async def close_all(self, k: int, symbols: list[str] | None = None) -> None:
        for symbol in symbols or self.symbols:
            await self.send(symbol, k)

    def times(self, symbol: str, time: datetime, warmup: int = 100) -> list[datetime]:
        frame = self.feed.windows(M5, [symbol], time, warmup).get(symbol)
        return [] if frame is None else frame["open_time"].to_list()


def rest_with(symbols: list[str], first: int, last: int) -> FakeBinance:
    """REST knowing candles first..last (indices, inclusive) for every symbol."""
    return FakeBinance({s: [at(k) for k in range(first, last + 1)] for s in symbols})


# -- ticks ------------------------------------------------------------------------------------


async def test_one_tick_per_candle_in_order_ignoring_forming_updates() -> None:
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    async with FakeStreamServer() as server:
        h = Harness(server, symbols, rest_with(symbols, -4, -1))
        await h.ready()
        await h.send("AAAUSDT", 0, 50.0, closed=False)  # forming: ignored
        for k in (0, 1, 2):
            for i, s in enumerate(symbols):
                await h.send(s, k, 100.0 + 10 * k + i)
        await until(lambda: len(h.ticks) == 3)
        await asyncio.sleep(0.05)
        await h.close()
    assert [t.open_time for t in h.ticks] == [at(0), at(1), at(2)]
    for k, tick in enumerate(h.ticks):
        assert tick.interval is M5
        assert tick.missing == ()
        assert set(tick.candles) == set(symbols)
        assert tick.candles["BBBUSDT"].open == 100.0 + 10 * k + 1
        assert tick.candles["AAAUSDT"].open_time == at(k)
    assert h.times("AAAUSDT", at(2), 4) == [at(k) for k in range(-1, 3)]


async def test_late_symbol_is_missing_then_lands_in_window_without_a_second_tick() -> None:
    symbols = ["AAAUSDT", "BBBUSDT"]
    async with FakeStreamServer() as server:
        h = Harness(server, symbols, rest_with(symbols, -4, -1))
        await h.ready()
        h.clock.now = at(1)  # candle 0 has just closed
        await h.send("AAAUSDT", 0)
        await until(lambda: len(h.ticks) == 1)
        assert h.ticks[0].missing == ("BBBUSDT",)
        assert set(h.ticks[0].candles) == {"AAAUSDT"}
        assert h.times("BBBUSDT", at(0))[-1] == at(-1)
        await h.send("BBBUSDT", 0)
        await until(lambda: h.times("BBBUSDT", at(0))[-1:] == [at(0)])
        await asyncio.sleep(0.3)
        await h.close()
    assert len(h.ticks) == 1
    assert h.times("BBBUSDT", at(0), 3) == [at(-2), at(-1), at(0)]


async def test_out_of_order_and_duplicate_candles() -> None:
    symbols = ["AAAUSDT", "BBBUSDT"]
    async with FakeStreamServer() as server:
        h = Harness(server, symbols, rest_with(symbols, -4, -1))
        await h.ready()
        await h.send("AAAUSDT", 0)
        await h.send("AAAUSDT", 0)  # duplicate
        await h.send("AAAUSDT", 1)  # A is ahead of B
        await h.send("BBBUSDT", 0)
        await until(lambda: len(h.ticks) == 1)
        await asyncio.sleep(0.05)
        # during tick 0 the held-back candle 1 is not visible, for any symbol
        assert h.times("AAAUSDT", at(0))[-1] == at(0)
        assert h.times("BBBUSDT", at(0))[-1] == at(0)
        await h.send("BBBUSDT", 1)
        await h.send("BBBUSDT", 1)
        await until(lambda: len(h.ticks) == 2)
        await asyncio.sleep(0.05)
        await h.close()
    assert [t.open_time for t in h.ticks] == [at(0), at(1)]
    for s in symbols:
        times = h.times(s, at(1))
        assert times == sorted(set(times))
        assert times[-1] == at(1)
        assert len(times) == 6  # everything held: the 4 seeded candles and candles 0 and 1
    assert h.times("AAAUSDT", at(1), 4) == [at(-2), at(-1), at(0), at(1)]


# -- reconnect and gaps -----------------------------------------------------------------------


async def test_reconnect_backfills_missed_candles_before_the_next_tick() -> None:
    symbols = ["AAAUSDT", "BBBUSDT"]
    rest = rest_with(symbols, -4, -1)
    async with FakeStreamServer() as server:
        h = Harness(server, symbols, rest)
        await h.ready()
        await h.close_all(0)
        await until(lambda: len(h.ticks) == 1)
        # the connection drops; candles 1 and 2 close while it is down
        for s in symbols:
            rest.data[s] += [at(1), at(2)]
        h.clock.now = at(3)
        await server.drop()
        await until(lambda: len(server.connections) == 2 and server.subscribed(1) != [])
        await until(lambda: any(isinstance(e, Backfilled) for e in h.events))
        await h.close_all(3)
        await until(lambda: len(h.ticks) == 4)
        await asyncio.sleep(0.05)
        await h.close()
    assert [t.open_time for t in h.ticks] == [at(k) for k in range(4)]
    assert all(t.missing == () for t in h.ticks)
    for s in symbols:
        assert h.times(s, at(3)) == [at(k) for k in range(-4, 4)]  # gap-free, no duplicates
    assert h.feed.health().reconnects == 1
    assert h.feed.health().connected
    kinds = [type(e) for e in h.events if isinstance(e, FeedConnected | FeedDisconnected)]
    assert kinds == [FeedConnected, FeedDisconnected, FeedConnected]
    backfill = next(e for e in h.events if isinstance(e, Backfilled))
    assert (backfill.reason, backfill.symbols, backfill.candles) == ("reconnect", 2, 4)
    assert server.subscribed(0) == server.subscribed(1)


async def test_gap_in_one_symbol_triggers_backfill_and_unfillable_gap_stays_a_hole() -> None:
    symbols = ["AAAUSDT", "BBBUSDT"]
    rest = rest_with(symbols, -4, -1)
    # REST knows A's candle 1 but not candle 2 (an exchange outage)
    rest.data["AAAUSDT"] += [at(1)]
    async with FakeStreamServer() as server:
        h = Harness(server, symbols, rest)
        await h.ready()
        h.clock.now = at(10)  # every timeout is long over
        await h.send("AAAUSDT", 0)
        await h.send("AAAUSDT", 3)  # candles 1 and 2 missing from A's stream
        for k in range(4):
            await h.send("BBBUSDT", k)
        await until(lambda: len(h.ticks) == 4)
        await asyncio.sleep(0.05)
        await h.close()
    assert [t.open_time for t in h.ticks] == [at(k) for k in range(4)]
    assert h.ticks[1].missing == ()  # backfilled before tick 1 went out
    assert h.ticks[2].missing == ("AAAUSDT",)  # REST could not fill it
    assert set(h.ticks[1].candles) == set(symbols)
    assert h.times("AAAUSDT", at(3), 5) == [at(-2), at(-1), at(0), at(1), at(3)]
    assert h.times("BBBUSDT", at(3), 5) == [at(k) for k in range(-1, 4)]
    gaps = [e for e in h.events if isinstance(e, Backfilled) and e.reason == "gap"]
    assert len(gaps) == 1
    assert gaps[0].candles == 1
    assert len([f for f in rest.fetches if f[0] == "AAAUSDT"]) == 2  # seed + one backfill


async def test_subscription_error_is_reported_as_an_event() -> None:
    async with FakeStreamServer() as server:
        server.reject_subscriptions = True
        h = Harness(server, ["AAAUSDT"], rest_with(["AAAUSDT"], -4, -1))
        await until(lambda: any(isinstance(e, SubscriptionFailed) for e in h.events))
        await h.close()


# -- seeding ----------------------------------------------------------------------------------


async def test_seed_from_rest_only() -> None:
    symbols = ["AAAUSDT", "BBBUSDT"]
    rest = rest_with(symbols, -10, 5)  # REST also has candles that have not closed yet
    async with FakeStreamServer() as server:
        h = Harness(server, symbols, rest, warmup=4)
        await h.ready()
        seeded = next(e for e in h.events if isinstance(e, Seeded))
        assert (seeded.symbols, seeded.from_store) == (2, 0)
        for s in symbols:
            assert h.times(s, at(100)) == [at(k) for k in range(-4, 0)]
            assert h.feed.last_candle(M5, s) is not None
        assert {f[2] for f in rest.fetches} == {at(-4)}
        assert {f[3] for f in rest.fetches} == {at(0)}
        await h.close()
    assert h.ticks == []


async def test_seed_from_store_then_rest_reads_only_needed_months(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2024, 3, 1, 0, 20, tzinfo=UTC)
    last_closed = Interval.M5.floor(now) - STEP
    all_times = [Interval.M5.floor(now) - k * STEP for k in range(1, 3000)][::-1]
    stored = [t for t in all_times if t <= last_closed - 2 * STEP]  # the store lags by 2 candles
    write_series(tmp_path, "AAAUSDT", M5, kline_frame(stored, [100.0] * len(stored)))
    rest = FakeBinance({"AAAUSDT": all_times, "BBBUSDT": all_times})
    read: list[datetime] = []
    original = KlineStore.read_month

    def spy(self: KlineStore, symbol: str, interval: Interval, month: datetime) -> pl.DataFrame:
        read.append(month)
        return original(self, symbol, interval, month)

    monkeypatch.setattr(KlineStore, "read_month", spy)
    async with FakeStreamServer() as server:
        h = Harness(
            server,
            ["AAAUSDT", "BBBUSDT"],
            rest,
            warmup=6,
            data_root=tmp_path,
            clock_now=now,
        )
        await h.ready()
        seeded = next(e for e in h.events if isinstance(e, Seeded))
        assert seeded.from_store == 4  # AAA: 4 of its 6 candles come from the store
        expected = [last_closed - k * STEP for k in reversed(range(6))]
        for s in ("AAAUSDT", "BBBUSDT"):
            assert h.times(s, now, 100) == expected
        await h.close()
    # the window starts on 29 Feb; only the February and March files are read, and only for AAA
    assert set(read) == {datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 3, 1, tzinfo=UTC)}
    aaa_fetch = [f for f in rest.fetches if f[0] == "AAAUSDT"]
    assert aaa_fetch[0][2] == last_closed - STEP  # only the two missing candles over REST
    assert KlineStore(tmp_path).list_months("AAAUSDT", M5)  # the store is untouched by the feed


# -- windows, staleness, health ---------------------------------------------------------------


async def test_windows_and_stale_symbols() -> None:
    data = {
        "AAAUSDT": [at(k) for k in range(-6, 0)],
        "BBBUSDT": [at(k) for k in range(-6, -3)],  # lags: last candle at -4
        "CCCUSDT": [],  # no candles at all
    }
    async with FakeStreamServer() as server:
        h = Harness(server, list(data), FakeBinance(data), warmup=5)
        await h.ready()
        windows = h.feed.windows(M5, ["AAAUSDT", "BBBUSDT", "CCCUSDT", "ZZZUSDT"], at(0), 3)
        assert set(windows) == {"AAAUSDT", "BBBUSDT"}
        assert windows["AAAUSDT"]["open_time"].to_list() == [at(-3), at(-2), at(-1)]
        assert windows["BBBUSDT"]["open_time"].to_list() == [at(-5), at(-4)]
        earlier = h.feed.windows(M5, ["AAAUSDT"], at(-3), 10)["AAAUSDT"]
        assert earlier["open_time"][-1] == at(-3)
        assert earlier.height == 3
        assert list(earlier.columns) == [
            "open_time", "open", "high", "low", "close", "volume", "quote_volume",
            "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume",
        ]  # fmt: skip
        # at(0) is forming: BBB's latest candle is 4 intervals old, AAA's 1
        assert h.feed.stale_symbols(M5, at(0), 2) == {"BBBUSDT", "CCCUSDT"}
        assert h.feed.stale_symbols(M5, at(0), 4) == {"CCCUSDT"}
        assert h.feed.stale_symbols(M5, at(0) + timedelta(minutes=3), 3) == {"BBBUSDT", "CCCUSDT"}
        await h.close()


async def test_health_and_stale_event() -> None:
    symbols = ["AAAUSDT", "BBBUSDT"]
    async with FakeStreamServer() as server:
        h = Harness(server, symbols, rest_with(symbols, -4, -1))
        assert not h.feed.health().connected
        await h.ready()
        health = h.feed.health()
        assert health.connected
        assert (health.connections, health.reconnects) == (1, 0)
        assert health.last_tick_at is None
        h.clock.now = at(8)  # candles up to -1 are now far behind
        await h.close_all(0)
        await until(lambda: len(h.ticks) == 1)
        health = h.feed.health()
        assert health.last_tick == {M5: at(0)}
        assert health.last_message_at is not None
        stale = [e for e in h.events if isinstance(e, StaleSymbolsChanged)]
        # at(8) is the clock: five intervals after candle 0, so both symbols are stale
        assert [(e.interval, e.symbols) for e in stale] == [(M5, ("AAAUSDT", "BBBUSDT"))]
        await h.close()
