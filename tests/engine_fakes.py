"""Fakes for engine tests: a strategy with scheduled signals, a hand-driven feed and a clock."""

from __future__ import annotations

import asyncio
import time as _time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from store_fakes import kline_frame, write_series

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.data.stream import Candle
from crypto_moonshot.live.feed import FeedHealth, Tick
from crypto_moonshot.live.state import StateStore
from crypto_moonshot.models import KLINE_SCHEMA, Interval

type Bar = tuple[float, float, float, float]
type Row = tuple[str, float] | tuple[str, float, str]
"""(symbol, priority) or (symbol, priority, action)."""


def flat_bars(n: int, price: float = 100.0) -> list[Bar | None]:
    return [(price, price, price, price)] * n


def write_bars(
    root: Path, symbol: str, bars: Sequence[Bar | None], interval: Interval, start: datetime
) -> None:
    """Write (open, high, low, close) bars from `start`, one per interval; None is a hole."""
    rows = [(start + k * interval.length, b) for k, b in enumerate(bars) if b is not None]
    frame = kline_frame(
        [t for t, _ in rows],
        [b[3] for _, b in rows],
        open_=[b[0] for _, b in rows],
        high=[b[1] for _, b in rows],
        low=[b[2] for _, b in rows],
    )
    write_series(root, symbol, interval, frame)


class Scheduled:
    """A plugin whose signals are given in advance: `schedule[time]` lists the rows for the
    candle `time`. `delay` sleeps (in the worker thread) before answering; `fail_at` raises
    for those candle times."""

    context_symbols: tuple[str, ...] = ()

    def __init__(
        self,
        schedule: Mapping[datetime, Sequence[Row]],
        *,
        name: str = "sched",
        interval: Interval = Interval.H1,
        exit: ExitPlan | None = None,
        warmup: int = 3,
        delay: float = 0.0,
        fail_at: Sequence[datetime] = (),
    ) -> None:
        self.name = name
        self.interval = interval
        self.warmup = warmup
        self.exit = exit
        self.schedule = dict(schedule)
        self.delay = delay
        self.fail_at = set(fail_at)
        self.calls: list[datetime] = []

    def on_candles(
        self,
        windows: Mapping[str, pl.DataFrame],
        time: datetime,
        context: Mapping[str, pl.DataFrame],
    ) -> pl.DataFrame:
        self.calls.append(time)
        if self.delay:
            _time.sleep(self.delay)
        if time in self.fail_at:
            msg = "boom"
            raise RuntimeError(msg)
        rows = self.schedule.get(time, [])
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(
            {
                "symbol": [r[0] for r in rows],
                "priority": [float(r[1]) for r in rows],
                "action": [r[2] if len(r) > 2 else "enter" for r in rows],
            }
        )

    def signal_frame(self) -> pl.DataFrame:
        """The scheduled entries as a backtest signal frame."""
        rows = [
            (r[0], t, float(r[1]))
            for t, rs in sorted(self.schedule.items())
            for r in rs
            if len(r) < 3 or r[2] == "enter"
        ]
        return pl.DataFrame(
            rows,
            schema={
                "symbol": pl.String,
                "open_time": pl.Datetime("us", "UTC"),
                "priority": pl.Float64,
            },
            orient="row",
        )


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@dataclass
class FakeFeed:
    """A `Feed` the test drives: `emit(k)` closes candle `k` (from `bars`) for the given symbols
    and yields its tick. Candles of symbols left out never arrive unless `late` adds them."""

    interval: Interval
    start: datetime
    bars: dict[str, list[Bar | None]]
    known: dict[str, list[Candle]] = field(default_factory=dict)
    queue: asyncio.Queue[Tick | None] = field(default_factory=asyncio.Queue)
    closed: bool = False

    def time(self, k: int) -> datetime:
        return self.start + k * self.interval.length

    def candle(self, symbol: str, k: int) -> Candle | None:
        bars = self.bars[symbol]
        bar = bars[k] if k < len(bars) else None
        if bar is None:
            return None
        o, h, lo, c = bar
        return Candle(self.time(k), o, h, lo, c, 1.0, c, 1, 0.5, c / 2)

    def seed(self, upto: int) -> None:
        """Every symbol's candles before index `upto` are known (no ticks)."""
        for symbol in self.bars:
            for k in range(upto):
                self._add(symbol, k)

    def _add(self, symbol: str, k: int) -> Candle | None:
        candle = self.candle(symbol, k)
        if candle is not None:
            held = self.known.setdefault(symbol, [])
            held.append(candle)
            held.sort(key=lambda c: c.open_time)
        return candle

    def late(self, symbol: str, k: int) -> None:
        self._add(symbol, k)

    async def emit(self, k: int, symbols: Sequence[str] | None = None) -> Tick:
        candles = {}
        for symbol in symbols if symbols is not None else list(self.bars):
            candle = self._add(symbol, k)
            if candle is not None:
                candles[symbol] = candle
        missing = tuple(s for s in self.bars if s not in candles)
        tick = Tick(self.interval, self.time(k), candles, missing, self.time(k + 1))
        await self.queue.put(tick)
        return tick

    async def ticks(self) -> AsyncIterator[Tick]:
        while True:
            tick = await self.queue.get()
            if tick is None:
                return
            yield tick

    def windows(
        self, interval: Interval, symbols: Sequence[str], time: datetime, warmup: int
    ) -> dict[str, pl.DataFrame]:
        out = {}
        for symbol in symbols:
            held = [c for c in self.known.get(symbol, []) if c.open_time <= time]
            if held:
                out[symbol] = pl.DataFrame(held[-warmup:], schema=KLINE_SCHEMA, orient="row")
        return out

    def last_candle(self, interval: Interval, symbol: str) -> Candle | None:
        held = self.known.get(symbol)
        return held[-1] if held else None

    def stale_symbols(self, interval: Interval, now: datetime, stale_after: int) -> set[str]:
        limit = interval.floor(now) - stale_after * interval.length
        return {
            s for s in self.bars if not self.known.get(s) or self.known[s][-1].open_time < limit
        }

    def health(self) -> FeedHealth:
        return FeedHealth(True, 1, None, 0, None, {})

    async def aclose(self) -> None:
        if not self.closed:
            self.closed = True
            await self.queue.put(None)


async def until(check: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not check():
        if asyncio.get_running_loop().time() > deadline:
            msg = "condition not reached"
            raise TimeoutError(msg)
        await asyncio.sleep(0.005)


def dump_state(store: StateStore) -> dict[str, list[tuple[object, ...]]]:
    """Every table that must survive a restart unchanged (not events or providers)."""
    db = store._db
    tables = (
        "strategies",
        "ticks",
        "batches",
        "signals",
        "orders",
        "fills",
        "positions",
        "controls",
    )
    return {t: [tuple(r) for r in db.execute(f"SELECT * FROM {t} ORDER BY rowid")] for t in tables}


HOURS = timedelta(hours=1)
