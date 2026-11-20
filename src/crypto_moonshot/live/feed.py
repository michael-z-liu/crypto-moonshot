"""The market data feed of the trading daemon: ticks and windows of closed candles.

`Feed` is what the engine uses; `LiveFeed` implements it over the Binance WebSocket
(`data.stream`) with REST seeding and backfill, and the replay feed (M5) implements it over
stored candles. Only *closed* candles exist here. The daemon never writes the Parquet store.

Timing rules of `LiveFeed`
--------------------------
- **Seeding.** At start every (interval, symbol) gets its last `warmup` candles that closed by
  `clock()` (open time < `interval.floor(now)`): from the store when a data root is given (only
  the month files the window touches), then the rest over REST. Candles up to that point never
  produce ticks (`emitted_through`).
- **Ticks.** Closed candles are grouped by (interval, open time). Tick T of an interval is
  emitted when every subscribed symbol has closed T, or `tick_timeout` seconds after T's close
  time (measured with the `clock`), whichever is first, and only after every older pending tick
  of the interval and every running backfill of the interval. Ticks of an interval come in
  open-time order and never twice. `Tick.missing` lists the subscribed symbols without a T
  candle at emission.
- **Hold-back.** A candle for a not yet emitted T is held and joins its window when tick T is
  emitted, so during tick T every window ends at T (or earlier for a missing symbol), whatever
  newer candles have arrived. The window keeps `max(warmup) + HISTORY_SLACK` candles, and
  `windows(time=T)` ignores candles after T.
- **Late candles.** A candle with open time at or before the last emitted tick joins its window
  at once if it is new (newer than the window or filling a hole) and never makes a tick.
- **Backfill.** After a reconnect of a connection, and when a symbol's candle opens more than one
  interval after its last known candle, the missing range is fetched over REST. Ticks of the
  interval wait for it. Candles it returns for a pending tick count for that tick; a range REST
  cannot fill stays a hole (it is not retried, and a failed request is reported, not raised).
- **Staleness.** `stale_symbols(interval, now, stale_after)`: symbols whose latest held candle
  opens more than `stale_after` intervals before the candle forming at `now` (no candle at all
  counts as stale).
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from crypto_moonshot.data.binance import BinanceError
from crypto_moonshot.data.store import InvalidMonthFileError, KlineStore
from crypto_moonshot.data.stream import (
    Candle,
    ClosedCandle,
    Connected,
    Disconnected,
    KlineStream,
    StreamConfig,
    SubscriptionError,
)
from crypto_moonshot.models import KLINE_SCHEMA, Interval, month_start

logger = logging.getLogger(__name__)

HISTORY_SLACK = 32
"""Candles held beyond the largest warm-up, so a window is still full for a tick that the
engine handles after newer ticks were emitted."""

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


# -- public types -------------------------------------------------------------------------


@dataclass(frozen=True)
class Tick:
    """Candle `open_time` of `interval` has closed."""

    interval: Interval
    open_time: datetime
    candles: Mapping[str, Candle]
    """The symbols that closed this candle, with the candle."""
    missing: tuple[str, ...]
    """Subscribed symbols of the interval without this candle when the tick was emitted."""
    emitted_at: datetime


@dataclass(frozen=True)
class FeedHealth:
    connected: bool
    """Every WebSocket connection is up."""
    connections: int
    last_message_at: datetime | None
    reconnects: int
    last_tick_at: datetime | None
    last_tick: Mapping[Interval, datetime]
    """Open time of the latest tick of each interval."""


class Feed(Protocol):
    """What the engine needs from a market data source (live or replay)."""

    def ticks(self) -> AsyncIterator[Tick]:
        """Ticks of all subscribed intervals (per interval in open-time order) until closed."""
        ...

    def windows(
        self, interval: Interval, symbols: Sequence[str], time: datetime, warmup: int
    ) -> dict[str, pl.DataFrame]:
        """Each symbol's last `warmup` closed candles with open time <= `time`, oldest first,
        in `KLINE_SCHEMA`. A symbol without such candles is left out."""
        ...

    def last_candle(self, interval: Interval, symbol: str) -> Candle | None: ...

    def stale_symbols(self, interval: Interval, now: datetime, stale_after: int) -> set[str]: ...

    def health(self) -> FeedHealth: ...

    async def aclose(self) -> None: ...


class KlineFetcher(Protocol):
    """The REST side (`BinanceClient` implements it)."""

    async def fetch_klines(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> pl.DataFrame: ...


# -- events ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class FeedConnected:
    connection: int
    reconnect: bool


@dataclass(frozen=True)
class FeedDisconnected:
    connection: int
    reason: str


@dataclass(frozen=True)
class Seeded:
    interval: Interval
    symbols: int
    candles: int
    from_store: int


@dataclass(frozen=True)
class SeedFailed:
    interval: Interval
    symbol: str
    error: str


@dataclass(frozen=True)
class Backfilled:
    interval: Interval
    reason: str
    """`reconnect` or `gap`."""
    symbols: int
    candles: int


@dataclass(frozen=True)
class BackfillFailed:
    interval: Interval
    symbol: str
    error: str


@dataclass(frozen=True)
class SubscriptionFailed:
    connection: int
    message: str


@dataclass(frozen=True)
class StaleSymbolsChanged:
    interval: Interval
    symbols: tuple[str, ...]
    """The stale symbols now (empty once none is)."""


FeedEvent = (
    FeedConnected
    | FeedDisconnected
    | Seeded
    | SeedFailed
    | Backfilled
    | BackfillFailed
    | SubscriptionFailed
    | StaleSymbolsChanged
)
EventSink = Callable[[FeedEvent], None]


# -- windows --------------------------------------------------------------------------------


def candles_frame(candles: Sequence[Candle]) -> pl.DataFrame:
    return pl.DataFrame(candles, schema=KLINE_SCHEMA, orient="row")


class _Window:
    """One symbol's closed candles: a sorted, duplicate-free frame of at most `cap` rows.

    Appending a newer candle (the usual case) is O(1): it waits in `_tail` and is concatenated
    once, when the frame is read.
    """

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._frame = pl.DataFrame(schema=KLINE_SCHEMA)
        self._tail: list[Candle] = []
        self.last: Candle | None = None

    def add(self, candle: Candle) -> None:
        """Insert unless a candle with that open time is held already."""
        if self.last is None or candle.open_time > self.last.open_time:
            self._tail.append(candle)
            self.last = candle
            return
        frame = self.frame()
        if (frame["open_time"] == candle.open_time).any():
            return
        merged = pl.concat([frame, candles_frame([candle])]).sort("open_time")
        self._frame = merged.tail(self._cap)

    def seed(self, frame: pl.DataFrame) -> None:
        self._frame = frame.sort("open_time").unique("open_time", keep="last").tail(self._cap)
        self._tail = []
        self.last = Candle(*self._frame.row(-1)) if self._frame.height else None

    def frame(self) -> pl.DataFrame:
        if self._tail:
            self._frame = pl.concat([self._frame, candles_frame(self._tail)]).tail(self._cap)
            self._tail = []
        return self._frame


@dataclass
class _State:
    """Everything the feed keeps for one interval."""

    interval: Interval
    symbols: tuple[str, ...]
    warmup: int
    windows: dict[str, _Window]
    known_last: dict[str, datetime] = field(default_factory=dict)
    """Latest open time seen per symbol, held back or not (for gap detection)."""
    pending: dict[datetime, dict[str, Candle]] = field(default_factory=dict)
    expired: set[datetime] = field(default_factory=set)
    timers: dict[datetime, asyncio.Task[None]] = field(default_factory=dict)
    backfills: set[asyncio.Task[None]] = field(default_factory=set)
    emitted_through: datetime | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stale: frozenset[str] = frozenset()


class LiveFeed:
    """A `Feed` over Binance's kline streams. See the module docstring for the timing rules.

    Args:
        subscriptions: symbols per interval (context symbols included).
        warmup: candles each interval's windows must hold (at least); missing intervals use 1.
        client: REST source for seeding and backfill (`BinanceClient`).
        stream_config: WebSocket settings (base URL, chunking, backoff, silence timeout).
        data_root: Parquet store to seed from before REST; the feed only reads it.
        tick_timeout: seconds to wait after a candle closes for every symbol's candle.
        clock: aware-UTC "now" (the timeout is measured against it).
        on_event: receives `FeedEvent`s; the feed never prints.
        concurrency: parallel seed / backfill symbol fetches.
    """

    def __init__(
        self,
        subscriptions: Mapping[Interval, Sequence[str]],
        warmup: Mapping[Interval, int],
        *,
        client: KlineFetcher,
        stream_config: StreamConfig | None = None,
        data_root: Path | None = None,
        tick_timeout: float = 10.0,
        clock: Clock = _utc_now,
        on_event: EventSink | None = None,
        concurrency: int = 8,
        stale_after: int = 2,
    ) -> None:
        self._client = client
        self._store = KlineStore(data_root) if data_root is not None else None
        self._tick_timeout = tick_timeout
        self._clock = clock
        self._on_event = on_event
        self._stale_after = stale_after
        self._sem = asyncio.Semaphore(concurrency)
        self._states: dict[Interval, _State] = {}
        for interval, symbols in subscriptions.items():
            if interval.is_monthly:
                msg = "1M candles are not supported by the live feed"
                raise ValueError(msg)
            unique = tuple(dict.fromkeys(s.upper() for s in symbols))
            n = max(1, warmup.get(interval, 1))
            self._states[interval] = _State(
                interval,
                unique,
                n,
                {s: _Window(n + HISTORY_SLACK) for s in unique},
            )
        self._stream = KlineStream(
            [(s, i) for i, st in self._states.items() for s in st.symbols],
            stream_config,
            now=clock,
        )
        self._out: asyncio.Queue[Tick | BaseException | None] = asyncio.Queue()
        self._main: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._up: set[int] = set()
        self._reconnects = 0
        self._last_tick_at: datetime | None = None
        self._last_tick: dict[Interval, datetime] = {}
        self._closed = False

    # -- Feed API ---------------------------------------------------------------------------

    async def ticks(self) -> AsyncIterator[Tick]:
        if self._main is None:
            self._main = asyncio.create_task(self._run(), name="live-feed")
        try:
            while True:
                item = await self._out.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = list(self._tasks)
        if self._main is not None:
            tasks.append(self._main)
        for st in self._states.values():
            tasks.extend(st.timers.values())
            tasks.extend(st.backfills)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._stream.aclose()
        self._out.put_nowait(None)

    def windows(
        self, interval: Interval, symbols: Sequence[str], time: datetime, warmup: int
    ) -> dict[str, pl.DataFrame]:
        st = self._states[interval]
        result: dict[str, pl.DataFrame] = {}
        for symbol in symbols:
            window = st.windows.get(symbol)
            if window is None or window.last is None:
                continue
            frame = window.frame()
            if window.last.open_time > time:
                frame = frame.filter(pl.col("open_time") <= time)
            if frame.height:
                result[symbol] = frame.tail(warmup)
        return result

    def last_candle(self, interval: Interval, symbol: str) -> Candle | None:
        window = self._states[interval].windows.get(symbol)
        return window.last if window is not None else None

    def stale_symbols(self, interval: Interval, now: datetime, stale_after: int) -> set[str]:
        st = self._states[interval]
        limit = interval.floor(now) - stale_after * interval.length
        return {s for s, w in st.windows.items() if w.last is None or w.last.open_time < limit}

    def health(self) -> FeedHealth:
        return FeedHealth(
            connected=len(self._up) == self._stream.connection_count > 0,
            connections=self._stream.connection_count,
            last_message_at=self._stream.last_message_at,
            reconnects=self._reconnects,
            last_tick_at=self._last_tick_at,
            last_tick=dict(self._last_tick),
        )

    # -- main loop --------------------------------------------------------------------------

    def _emit_event(self, event: FeedEvent) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            logger.exception("feed event sink raised; ignoring it")

    def _spawn(
        self, coro: Coroutine[Any, Any, None], st: _State | None = None
    ) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        bucket = st.backfills if st is not None else self._tasks
        bucket.add(task)
        task.add_done_callback(bucket.discard)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._out.put_nowait(exc)

    async def _run(self) -> None:
        try:
            self._stream.start()
            await asyncio.gather(*(self._seed(st) for st in self._states.values()))
            async for event in self._stream.events():
                self._handle(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._out.put_nowait(exc)

    def _handle(self, event: object) -> None:
        if isinstance(event, ClosedCandle):
            st = self._states.get(event.interval)
            if st is not None and event.symbol in st.windows:
                self._accept(st, event.symbol, event.candle, detect_gap=True)
        elif isinstance(event, Connected):
            self._up.add(event.connection)
            self._emit_event(FeedConnected(event.connection, event.reconnect))
            if event.reconnect:
                self._reconnects += 1
                self._backfill_reconnect(event)
        elif isinstance(event, Disconnected):
            self._up.discard(event.connection)
            self._emit_event(FeedDisconnected(event.connection, event.reason))
        elif isinstance(event, SubscriptionError):
            self._emit_event(SubscriptionFailed(event.connection, event.message))

    # -- seeding ----------------------------------------------------------------------------

    async def _seed(self, st: _State) -> None:
        end = st.interval.floor(self._clock())
        start = end - st.warmup * st.interval.length
        st.emitted_through = end - st.interval.length
        counts = [0, 0]

        async def one(symbol: str) -> None:
            async with self._sem:
                stored = await asyncio.to_thread(self._read_store, st.interval, symbol, start, end)
                fetch_from = start
                if stored.height:
                    fetch_from = max(start, st.interval.next(stored["open_time"][-1]))
                frame = stored
                if fetch_from < end:
                    try:
                        fetched = await self._client.fetch_klines(
                            symbol, st.interval, fetch_from, end
                        )
                        frame = pl.concat([stored, fetched])
                    except BinanceError as exc:
                        self._emit_event(SeedFailed(st.interval, symbol, str(exc)))
            window = st.windows[symbol]
            window.seed(frame)
            counts[0] += frame.height
            counts[1] += stored.height
            if window.last is not None:
                st.known_last[symbol] = window.last.open_time

        await asyncio.gather(*(one(s) for s in st.symbols))
        self._emit_event(Seeded(st.interval, len(st.symbols), counts[0], counts[1]))

    def _read_store(
        self, interval: Interval, symbol: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        """Stored candles in `[start, end)`, reading only the month files that overlap it."""
        empty = pl.DataFrame(schema=KLINE_SCHEMA)
        if self._store is None:
            return empty
        first = month_start(start)
        frames = [empty]
        for month in self._store.list_months(symbol, interval):
            if month < first or month >= end:
                continue
            try:
                frames.append(self._store.read_month(symbol, interval, month))
            except (FileNotFoundError, InvalidMonthFileError):
                continue
        merged = pl.concat(frames)
        return (
            merged.filter((pl.col("open_time") >= start) & (pl.col("open_time") < end))
            .sort("open_time")
            .unique("open_time", keep="last", maintain_order=True)
        )

    # -- candles and ticks ------------------------------------------------------------------

    def _accept(self, st: _State, symbol: str, candle: Candle, *, detect_gap: bool) -> None:
        t = candle.open_time
        prev = st.known_last.get(symbol)
        if detect_gap and prev is not None and t > st.interval.next(prev):
            self._spawn(
                self._backfill(st, [(symbol, st.interval.next(prev), t)], "gap"),
                st,
            )
        if prev is None or t > prev:
            st.known_last[symbol] = t
        if st.emitted_through is not None and t <= st.emitted_through:
            st.windows[symbol].add(candle)
            return
        bucket = st.pending.get(t)
        if bucket is None:
            bucket = st.pending[t] = {}
            st.timers[t] = asyncio.create_task(self._timer(st, t))
        bucket[symbol] = candle
        if len(bucket) == len(st.symbols):
            self._kick(st)

    async def _timer(self, st: _State, t: datetime) -> None:
        deadline = st.interval.next(t) + timedelta(seconds=self._tick_timeout)
        await asyncio.sleep(max(0.0, (deadline - self._clock()).total_seconds()))
        st.expired.add(t)
        self._kick(st)

    def _kick(self, st: _State) -> None:
        self._spawn(self._emit_ready(st))

    async def _emit_ready(self, st: _State) -> None:
        async with st.lock:
            while True:
                running = [t for t in st.backfills if not t.done()]
                if running:
                    await asyncio.wait(running)
                    continue
                if not st.pending:
                    return
                t = min(st.pending)
                bucket = st.pending[t]
                if len(bucket) < len(st.symbols) and t not in st.expired:
                    return
                self._emit_tick(st, t)

    def _emit_tick(self, st: _State, t: datetime) -> None:
        bucket = st.pending.pop(t)
        st.expired.discard(t)
        timer = st.timers.pop(t, None)
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()
        st.emitted_through = t
        for symbol, candle in bucket.items():
            st.windows[symbol].add(candle)
        now = self._clock()
        missing = tuple(s for s in st.symbols if s not in bucket)
        self._last_tick_at = now
        self._last_tick[st.interval] = t
        self._out.put_nowait(Tick(st.interval, t, bucket, missing, now))
        stale = frozenset(self.stale_symbols(st.interval, now, self._stale_after))
        if stale != st.stale:
            st.stale = stale
            self._emit_event(StaleSymbolsChanged(st.interval, tuple(sorted(stale))))

    # -- backfill ---------------------------------------------------------------------------

    def _backfill_reconnect(self, event: Connected) -> None:
        by_interval: dict[Interval, list[str]] = defaultdict(list)
        for symbol, interval in event.streams:
            if interval in self._states:
                by_interval[interval].append(symbol)
        for interval, symbols in by_interval.items():
            st = self._states[interval]
            end = interval.floor(self._clock())
            ranges: list[tuple[str, datetime, datetime]] = []
            for symbol in symbols:
                prev = st.known_last.get(symbol)
                start = (
                    interval.next(prev) if prev is not None else end - st.warmup * interval.length
                )
                if start < end:
                    ranges.append((symbol, start, end))
            if ranges:
                self._spawn(self._backfill(st, ranges, "reconnect"), st)

    async def _backfill(
        self, st: _State, ranges: Sequence[tuple[str, datetime, datetime]], reason: str
    ) -> None:
        counts = [0, 0]

        async def one(symbol: str, start: datetime, end: datetime) -> None:
            async with self._sem:
                try:
                    frame = await self._client.fetch_klines(symbol, st.interval, start, end)
                except BinanceError as exc:
                    self._emit_event(BackfillFailed(st.interval, symbol, str(exc)))
                    return
            for row in frame.iter_rows():
                self._accept(st, symbol, Candle(*row), detect_gap=False)
            if frame.height:
                counts[0] += 1
                counts[1] += frame.height

        try:
            await asyncio.gather(*(one(*r) for r in ranges))
        finally:
            self._emit_event(Backfilled(st.interval, reason, counts[0], counts[1]))
            if st.pending:
                # Runs after this task leaves `st.backfills` (its done callback), via the loop.
                asyncio.get_running_loop().call_soon(self._kick, st)
