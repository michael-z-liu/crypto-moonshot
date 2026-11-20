"""Replay: the live engine over stored candles, with a simulated clock.

`ReplayFeed` implements `Feed` over the Parquet store for `[start, end)`: one tick for every
candle time at which at least one subscribed symbol has a candle (ticks of several intervals in
order of close time, then interval length), windows sliced from series loaded once, and
staleness by the live rule. Its clock is the close time of the tick being processed, so every
timestamp the engine writes is deterministic. `run_replay` runs the same `Engine` over it in
replay mode (it waits for every strategy's batch, never a wall-clock deadline) and writes the
state file plus the backtest's result files. A replay resumes from an existing state file's
last processed tick.

With `signals="backtest"`, strategies that implement `backtest_signals` are run once over the
whole range and the engine gets those precomputed signals per tick (`PrecomputedStrategy`)
instead of calling `on_candles` on every candle: much faster on real universes. The rolling
path (`on_candles`) is the default and gives exactly the live signals.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from crypto_moonshot.backtest.engine import (
    TRADE_COLUMNS,
    Marks,
    equity_curve,
    position_marks,
)
from crypto_moonshot.backtest.metrics import summarize, trade_metrics
from crypto_moonshot.backtest.results import write_run_files
from crypto_moonshot.backtest.source import ParquetCandleSource
from crypto_moonshot.broker.candles import to_us
from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.data.stream import Candle
from crypto_moonshot.live.config import RunConfig
from crypto_moonshot.live.engine import (
    Engine,
    StepHook,
    StrategySpec,
    build_specs,
    feed_subscriptions,
    feed_warmups,
)
from crypto_moonshot.live.feed import HISTORY_SLACK, EventSink, FeedHealth, Tick
from crypto_moonshot.live.lock import StateLock
from crypto_moonshot.live.plugin import BacktestSignalsProvider, Strategy, empty_signal_frame
from crypto_moonshot.live.state import PositionRecord, StateStore
from crypto_moonshot.models import KLINE_SCHEMA, Interval, as_utc, month_start, next_month
from crypto_moonshot.strategy.base import Signals

_US = timedelta(microseconds=1)
_TIME = pl.Datetime("us", "UTC")


def read_range(
    store: KlineStore, symbol: str, interval: Interval, start: datetime, end: datetime
) -> pl.DataFrame:
    """Stored candles opening in `[start, end)`, sorted, reading only the month files that
    overlap the range (as stored, without the format checks)."""
    months = [m for m in store.list_months(symbol, interval) if m < end and next_month(m) > start]
    if not months:
        return pl.DataFrame(schema=KLINE_SCHEMA)
    paths = [store.month_path(symbol, interval, m) for m in months]
    return (
        pl.scan_parquet(paths)
        .select(list(KLINE_SCHEMA))
        .filter((pl.col("open_time") >= start) & (pl.col("open_time") < end))
        .collect()
        .sort("open_time")
        .unique("open_time", keep="first", maintain_order=True)
    )


class StoreHistory:
    """A `KlineFetcher` over the Parquet store: candles a resumed replay's broker needs from
    before the replay feed's windows."""

    def __init__(self, root: Path) -> None:
        self._store = KlineStore(root)

    async def fetch_klines(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> pl.DataFrame:
        return read_range(self._store, symbol, interval, as_utc(start), as_utc(end))


@dataclass
class _Series:
    frame: pl.DataFrame
    times: np.ndarray  # open times, µs


class ReplayFeed:
    """A `Feed` over stored candles in `[start, end)`.

    Each (symbol, interval) series is read once, from `lookback` candles of the interval
    before `start` (default: twice the warm-up plus `HISTORY_SLACK`, rounded down to a month
    start), so windows near `start` hold the full warm-up unless holes are longer than that.
    `clock()` is the close time of the latest tick (`start` before the first).
    """

    def __init__(
        self,
        root: Path,
        subscriptions: Mapping[Interval, Sequence[str]],
        start: datetime,
        end: datetime,
        *,
        warmup: Mapping[Interval, int] | None = None,
        lookback: Mapping[Interval, int] | None = None,
        on_event: EventSink | None = None,
    ) -> None:
        self.start, self.end = as_utc(start), as_utc(end)
        self.on_event = on_event
        store = KlineStore(root)
        self._now = self.start
        self._closed = False
        self._subscriptions = {i: tuple(dict.fromkeys(s)) for i, s in subscriptions.items()}
        self._series: dict[tuple[Interval, str], _Series] = {}
        self._last_tick: dict[Interval, datetime] = {}
        self._last_tick_at: datetime | None = None
        schedule: list[tuple[datetime, timedelta, Interval, datetime]] = []
        self._ticks: dict[Interval, tuple[pl.DataFrame, np.ndarray, np.ndarray]] = {}
        for interval, symbols in self._subscriptions.items():
            length = interval.length
            if lookback is not None and interval in lookback:
                back = lookback[interval]
                first = self.start - back * length
            else:
                back = 2 * ((warmup or {}).get(interval, 1) + HISTORY_SLACK)
                first = month_start(self.start - back * length)
            parts = []
            for symbol in symbols:
                frame = read_range(store, symbol, interval, first, self.end)
                times = frame["open_time"].dt.epoch("us").to_numpy().astype(np.int64)
                self._series[interval, symbol] = _Series(frame, times)
                parts.append(
                    frame.filter(pl.col("open_time") >= self.start).with_columns(
                        symbol=pl.lit(symbol, dtype=pl.String)
                    )
                )
            table = pl.concat(parts) if parts else pl.DataFrame(schema=KLINE_SCHEMA)
            if "symbol" not in table.columns:
                table = table.with_columns(symbol=pl.lit(None, dtype=pl.String))
            table = table.sort("open_time", "symbol")
            stamps = table["open_time"].dt.epoch("us").to_numpy().astype(np.int64)
            unique, offsets = np.unique(stamps, return_index=True)
            self._ticks[interval] = (table, unique, offsets)
            for us in unique.tolist():
                t = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=us)
                schedule.append((t + length, length, interval, t))
        schedule.sort(key=lambda e: (e[0], e[1]))
        self._schedule = schedule

    def clock(self) -> datetime:
        """The simulated now: the close time of the tick being processed."""
        return self._now

    @property
    def tick_count(self) -> int:
        return len(self._schedule)

    async def ticks(self) -> AsyncIterator[Tick]:
        index = {i: 0 for i in self._ticks}
        for close, _, interval, time in self._schedule:
            if self._closed:
                return
            table, unique, offsets = self._ticks[interval]
            k = index[interval]
            index[interval] = k + 1
            lo = int(offsets[k])
            hi = int(offsets[k + 1]) if k + 1 < len(offsets) else table.height
            assert int(unique[k]) == to_us(time)
            candles: dict[str, Candle] = {}
            for row in table.slice(lo, hi - lo).iter_rows():
                candles[row[-1]] = Candle(*row[:-1])
            missing = tuple(s for s in self._subscriptions[interval] if s not in candles)
            self._now = close
            self._last_tick[interval] = time
            self._last_tick_at = close
            yield Tick(interval, time, candles, missing, close)
            await asyncio.sleep(0)

    def windows(
        self, interval: Interval, symbols: Sequence[str], time: datetime, warmup: int
    ) -> dict[str, pl.DataFrame]:
        us = to_us(time)
        out: dict[str, pl.DataFrame] = {}
        for symbol in symbols:
            series = self._series.get((interval, symbol))
            if series is None:
                continue
            stop = int(np.searchsorted(series.times, us, side="right"))
            if stop == 0:
                continue
            out[symbol] = series.frame.slice(max(0, stop - warmup), min(warmup, stop))
        return out

    def last_candle(self, interval: Interval, symbol: str) -> Candle | None:
        series = self._series.get((interval, symbol))
        if series is None:
            return None
        closed = to_us(interval.floor(self._now) - interval.length)
        stop = int(np.searchsorted(series.times, closed, side="right"))
        return None if stop == 0 else Candle(*series.frame.row(stop - 1))

    def stale_symbols(self, interval: Interval, now: datetime, stale_after: int) -> set[str]:
        """The live rule: no candle closed by `now`, or the latest opens more than
        `stale_after` intervals before the candle forming at `now`."""
        floor = interval.floor(now)
        closed = to_us(floor - interval.length)
        limit = to_us(floor - stale_after * interval.length)
        out = set()
        for symbol in self._subscriptions.get(interval, ()):
            series = self._series[interval, symbol]
            stop = int(np.searchsorted(series.times, closed, side="right"))
            if stop == 0 or int(series.times[stop - 1]) < limit:
                out.add(symbol)
        return out

    def health(self) -> FeedHealth:
        return FeedHealth(
            connected=True,
            connections=0,
            last_message_at=None,
            reconnects=0,
            last_tick_at=self._last_tick_at,
            last_tick=dict(self._last_tick),
        )

    async def aclose(self) -> None:
        self._closed = True


class PrecomputedStrategy:
    """A plugin that serves precomputed signals (a `Signals` frame with `open_time`) per
    candle instead of computing them: the replay's fast path. It needs no windows."""

    needs_windows = False

    def __init__(self, base: Strategy, signals: pl.DataFrame) -> None:
        self.name = base.name
        self.interval = base.interval
        self.warmup = base.warmup
        self.context_symbols = tuple(base.context_symbols)
        self.exit = base.exit
        frame = signals.with_columns(pl.col("open_time").dt.convert_time_zone("UTC"))
        self._by_time: dict[datetime, pl.DataFrame] = {}
        for (time,), part in frame.group_by("open_time", maintain_order=True):
            assert isinstance(time, datetime)
            self._by_time[time] = part.drop("open_time")

    def on_candles(
        self,
        windows: Mapping[str, pl.DataFrame],
        time: datetime,
        context: Mapping[str, pl.DataFrame],
    ) -> pl.DataFrame:
        return self._by_time.get(as_utc(time), empty_signal_frame())


SignalMode = Literal["on_candles", "backtest"]


@dataclass(frozen=True)
class ReplayResult:
    """A replay's outputs: the trade list (closed positions), the equity curve, metrics and
    the per-year table, plus how many positions were still open at the end."""

    trades: pl.DataFrame
    equity: pl.DataFrame
    metrics: dict[str, float]
    yearly: pl.DataFrame
    open_positions: int
    document: dict[str, Any]


class ReplayRun:
    """A prepared replay: the store, feed and engine are built, nothing has run yet (M6 wires
    its API to `engine` before calling `run`)."""

    def __init__(
        self,
        config: RunConfig,
        specs: Sequence[StrategySpec],
        store: StateStore,
        feed: ReplayFeed,
        engine: Engine,
        data_root: Path,
        start: datetime,
        end: datetime,
        out_dir: Path | None,
        lock: StateLock | None = None,
    ) -> None:
        self._lock = lock
        self.config = config
        self.specs = list(specs)
        self.store = store
        self.feed = feed
        self.engine = engine
        self.data_root = data_root
        self.start = start
        self.end = end
        self.out_dir = out_dir

    async def run(self) -> ReplayResult:
        """Run the engine to the end, then build (and write, with `out_dir`) the results.
        The store stays open; `close` it when done."""
        await self.engine.run(self.feed)
        result = replay_outputs(self.store, self.config, self.specs, self.data_root, self.end)
        document = {**result.document, "start": self.start, "end": self.end}
        result = replace(result, document=document)
        if self.out_dir is not None:
            write_run_files(self.out_dir, result.trades, result.equity, result.yearly, document)
        return result

    def close(self) -> None:
        """Close the state file and release its lock."""
        self.store.close()
        if self._lock is not None:
            self._lock.release()


def _precompute(
    specs: Sequence[StrategySpec],
    root: Path,
    start: datetime,
    end: datetime,
    signals: SignalMode,
    precomputed: Mapping[str, Signals] | None,
) -> list[StrategySpec]:
    out = []
    for spec in specs:
        plugin = spec.plugin
        frame: pl.DataFrame | None = None
        if plugin is not None and precomputed is not None and spec.name in precomputed:
            frame = precomputed[spec.name].frame
        elif signals == "backtest" and isinstance(plugin, BacktestSignalsProvider):
            frame = plugin.backtest_signals(root, list(spec.symbols), end).frame
        if frame is not None and plugin is not None:
            frame = frame.filter((pl.col("open_time") >= start) & (pl.col("open_time") < end))
            spec = replace(spec, plugin=PrecomputedStrategy(plugin, frame))
        out.append(spec)
    return out


def prepare_replay(
    config: RunConfig,
    start: datetime,
    end: datetime,
    data_root: Path,
    out_dir: Path | None = None,
    *,
    signals: SignalMode = "on_candles",
    plugins: Mapping[str, Strategy] | None = None,
    precomputed: Mapping[str, Signals] | None = None,
    step_hook: StepHook | None = None,
    lookback: Mapping[Interval, int] | None = None,
) -> ReplayRun:
    """Build a replay of `config` over `[start, end)` of the store at `data_root`.

    The state file is `config.state` (created with its folder if needed); when it already holds
    processed ticks the replay resumes after them. `plugins` supplies plugin instances by
    strategy name; `precomputed` supplies signals by strategy name (as `signals="backtest"`
    does from `backtest_signals`). `step_hook` and `lookback` are for tests.
    """
    config.require_runnable()
    start, end = as_utc(start), as_utc(end)
    config.state.parent.mkdir(parents=True, exist_ok=True)
    lock = StateLock(config.state)
    lock.acquire()  # a replay writes the state file: never beside a daemon or another replay
    store: StateStore | None = None
    try:
        specs = build_specs(config, data_root, plugins=plugins)
        specs = _precompute(specs, data_root, start, end, signals, precomputed)
        store = StateStore.open(config.state)
        last = store.last_ticks()
        intervals = {s.interval for s in specs}
        resume = [last[i] + i.length for i in intervals if i in last]
        fresh = [i for i in intervals if i not in last]
        feed_start = min([*resume, *([start] if fresh else [])], default=start)
        feed = ReplayFeed(
            data_root,
            feed_subscriptions(specs, store.live_positions()),
            feed_start,
            end,
            warmup=feed_warmups(specs),
            lookback=lookback,
        )
        engine = Engine(
            config,
            specs,
            store,
            clock=feed.clock,
            replay=True,
            history=StoreHistory(data_root),
            step_hook=step_hook,
        )
        feed.on_event = engine.on_feed_event
    except BaseException:
        if store is not None:
            store.close()
        lock.release()
        raise
    return ReplayRun(config, specs, store, feed, engine, data_root, start, end, out_dir, lock)


def run_replay(
    config: RunConfig,
    start: datetime,
    end: datetime,
    data_root: Path,
    out_dir: Path | None = None,
    *,
    signals: SignalMode = "on_candles",
    plugins: Mapping[str, Strategy] | None = None,
    precomputed: Mapping[str, Signals] | None = None,
    step_hook: StepHook | None = None,
    lookback: Mapping[Interval, int] | None = None,
) -> ReplayResult:
    """Replay `config` over `[start, end)` and return (and with `out_dir`, write) the results;
    see `prepare_replay`. Runs its own event loop."""
    prepared = prepare_replay(
        config,
        start,
        end,
        data_root,
        out_dir,
        signals=signals,
        plugins=plugins,
        precomputed=precomputed,
        step_hook=step_hook,
        lookback=lookback,
    )
    try:
        return asyncio.run(prepared.run())
    finally:
        prepared.close()


# -- outputs ----------------------------------------------------------------------------------


def trade_frame(store: StateStore) -> pl.DataFrame:
    """The closed positions as a trade list in the backtest's format (`TRADE_COLUMNS`), in
    placement order, then `strategy` and the signals' `info` columns."""
    positions = sorted(store.closed_positions(), key=lambda p: p.id)
    rows: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    for p in positions:
        assert p.entry_time is not None
        assert p.exit_time is not None
        assert p.pnl is not None
        priority = None
        order = store.order(p.entry_order_id)
        if order is not None and order.signal_id is not None:
            signal = store.signal(order.signal_id)
            priority = None if signal is None else signal.priority
        rows.append(
            {
                "symbol": p.symbol,
                "signal_time": p.placed_at - p.interval.length,
                "entry_time": p.entry_time,
                "exit_time": p.exit_time,
                "exit_reason": p.exit_reason,
                "entry_ref": p.entry_ref,
                "entry_price": p.entry_price,
                "exit_price": p.exit_price,
                "quantity": p.quantity,
                "stake": p.cost,
                "pnl": p.pnl,
                "net_return": p.pnl / p.cost,
                "priority": priority,
                "strategy": p.strategy,
            }
        )
        infos.append(p.signal_info or {})
    schema: dict[str, Any] = {
        "symbol": pl.String,
        "signal_time": _TIME,
        "entry_time": _TIME,
        "exit_time": _TIME,
        "exit_reason": pl.String,
        "entry_ref": pl.Float64,
        "entry_price": pl.Float64,
        "exit_price": pl.Float64,
        "quantity": pl.Float64,
        "stake": pl.Float64,
        "pnl": pl.Float64,
        "net_return": pl.Float64,
        "priority": pl.Float64,
        "strategy": pl.String,
    }
    frame = pl.DataFrame(rows, schema=schema, orient="row")
    keys = list(dict.fromkeys(k for info in infos for k in info if k not in schema))
    if keys:
        extras = pl.DataFrame([{k: info.get(k) for k in keys} for info in infos])
        frame = pl.concat([frame, extras], how="horizontal")
    return frame.select(*TRADE_COLUMNS, "strategy", *keys)


def _equity(
    store: StateStore, source: ParquetCandleSource, initial_cash: float, end: datetime
) -> pl.DataFrame:
    """The mark-to-market curve: closed positions as traded, positions still open at `end`
    valued at their last close before `end` (not in the trade list)."""
    every = timedelta(days=1) // _US
    entries: list[int] = []
    stakes: list[float] = []
    exits: list[int] = []
    proceeds: list[float] = []
    marks = Marks()

    def add(p: PositionRecord, exit_time: datetime, back: float | None) -> None:
        assert p.entry_time is not None
        assert p.quantity is not None
        assert p.entry_ref is not None
        window = source.window(p.symbol, p.interval, p.entry_time, exit_time)
        if back is None:  # still open: the last close before the end
            closes = window.close if window is not None else np.empty(0)
            back = p.quantity * (float(closes[-1]) if len(closes) else p.entry_ref)
        entries.append(to_us(p.entry_time))
        stakes.append(p.cost)
        exits.append(to_us(exit_time))
        proceeds.append(back)
        if window is not None and len(window):
            marked = position_marks(window, p.entry_ref, p.quantity, to_us(exit_time), every)
            if marked is not None:
                marks.times.append(marked[0])
                marks.values.append(marked[1])

    for p in store.closed_positions():
        assert p.exit_time is not None
        add(p, p.exit_time, p.proceeds)
    for p in store.live_positions():
        if p.entry_time is not None:
            add(p, end, None)
    return equity_curve(
        np.array(entries, dtype=np.int64),
        np.array(stakes, dtype=np.float64),
        np.array(exits, dtype=np.int64),
        np.array(proceeds, dtype=np.float64),
        marks,
        initial_cash,
        every,
    )


def replay_outputs(
    store: StateStore,
    config: RunConfig,
    specs: Sequence[StrategySpec],
    data_root: Path,
    end: datetime,
) -> ReplayResult:
    """The trade list, equity curve, metrics and `metrics.json` document of a replay's state."""
    trades = trade_frame(store)
    source = ParquetCandleSource(data_root, until=end)
    initial = config.account.initial_cash
    equity = _equity(store, source, initial, end)
    metrics, yearly = summarize(trades, equity, initial)
    open_positions = sum(1 for p in store.live_positions() if p.entry_time is not None)
    per_strategy = {
        s.name: trade_metrics(trades.filter(pl.col("strategy") == s.name)) for s in specs
    }
    document: dict[str, Any] = {
        "mode": "replay",
        "created": datetime.now(UTC),
        "state": str(store.path),
        "metrics": metrics,
        "strategies": per_strategy,
        "signals": store.signal_outcomes(),
        "open_positions": open_positions,
        "config": config.model_dump(mode="json"),
    }
    return ReplayResult(trades, equity, metrics, yearly, open_positions, document)
