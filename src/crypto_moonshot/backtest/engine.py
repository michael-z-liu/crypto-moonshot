"""The backtest engine: signals in, trades, a mark-to-market equity curve and metrics out.

Signals are a polars frame with `symbol`, `open_time` (the open time of the signal candle *t*,
whose close the signal uses) and `priority` (lower first), plus any other columns, which are
carried into the trade list. A run:

1. Drops signals outside `[start, end)` (by `open_time`) and computes each entry time: the open
   of candle `t + 1 + entry_delay`.
2. Walks signals in order of entry time, then priority, then symbol. Before each entry, every
   position whose exit time is at or before the entry time is closed (cash returns, its slot
   frees), as in the research books.
3. Sizes the entry (`portfolio.Sizing`, on realised equity), runs the risk checks
   (`risk.RiskCheck`), checks cash, and loads the position's candle window. The entry candle
   must exist at exactly the entry time; otherwise the signal is skipped (`no_entry_candle`).
4. Places a market buy carrying the run's `ExitPlan` with a `SimulatedBroker`, which fills it at
   the entry candle's open and resolves the bracket exit over the window (on 1m candles where
   available with `use_fine`). A window too short to know the exit is extended, but never past
   the source's data end: a position still open there closes at the last close (`series_end`).

Every skipped signal is counted by reason in `BacktestResult.skipped`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import numpy as np
import polars as pl

from crypto_moonshot.backtest.metrics import summarize
from crypto_moonshot.backtest.source import CandleSource
from crypto_moonshot.broker.base import Costs, ExitPlan, Fill, Order, Side
from crypto_moonshot.broker.candles import Candles, from_us, to_us
from crypto_moonshot.broker.simulated import ExitFill, SimulatedBroker
from crypto_moonshot.models import Interval, as_utc
from crypto_moonshot.portfolio import Book, FixedStake, Sizing
from crypto_moonshot.risk import (
    DailyLossLimit,
    MaxExposure,
    MaxPositions,
    OnePerSymbol,
    RiskCheck,
    first_rejection,
)

_US = timedelta(microseconds=1)
_MAX_WINDOW = timedelta(days=3660)
_FAR_FUTURE = datetime(9999, 1, 1, tzinfo=UTC)
SIGNAL_COLUMNS = ("symbol", "open_time", "priority")
UTC_TIME = pl.Datetime("us", "UTC")


class SkipReason(StrEnum):
    """Why a signal was not taken."""

    BEFORE_START = "before_start"
    AFTER_END = "after_end"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    SLOTS_FULL = "slots_full"
    SYMBOL_HELD = "symbol_held"
    EXPOSURE = "exposure"
    CASH = "cash"
    NO_ENTRY_CANDLE = "no_entry_candle"


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """One backtest run's settings.

    - `entry_delay`: extra candles between the signal candle and the entry candle (0: enter at
      the open of the next candle).
    - `sizing`: USDT spent per entry, fee included (`FixedStake` or `EqualSlots`).
    - `max_positions`, `max_exposure` (open cost basis over realised equity), `daily_loss_limit`
      (fraction of the day's starting realised equity): risk checks; `None` disables one. One
      position per symbol always applies (the broker holds one position per symbol).
    - `start`, `end`: take signals whose `open_time` is in `[start, end)`.
    - `use_fine`: fill exits on `fine_interval` candles where they cover a signal candle.
    - `mark_every`: spacing of the equity curve's points, aligned to the Unix epoch (midnight
      UTC for whole days).
    - `window`: the first candle window loaded per position when the exit plan has no
      `max_hold`; doubled until the exit is known or the source's data end is reached.
    """

    exit: ExitPlan
    interval: Interval = Interval.M5
    entry_delay: int = 0
    costs: Costs = field(default_factory=Costs)
    sizing: Sizing = field(default_factory=FixedStake)
    max_positions: int | None = None
    max_exposure: float | None = None
    daily_loss_limit: float | None = None
    initial_capital: float = 10_000.0
    start: datetime | None = None
    end: datetime | None = None
    use_fine: bool = False
    fine_interval: Interval = Interval.M1
    mark_every: timedelta = timedelta(days=1)
    window: timedelta = timedelta(days=3)

    def __post_init__(self) -> None:
        if self.entry_delay < 0:
            msg = f"entry_delay must be >= 0, got {self.entry_delay}"
            raise ValueError(msg)
        if not self.initial_capital > 0:
            msg = f"initial_capital must be positive, got {self.initial_capital}"
            raise ValueError(msg)
        if self.mark_every <= timedelta(0) or self.window <= timedelta(0):
            msg = "mark_every and window must be positive"
            raise ValueError(msg)
        for name in ("start", "end"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, as_utc(value))
        _ = self.interval.length  # raises for 1M, which has no fixed length

    def risk_checks(self) -> list[RiskCheck]:
        """The configured checks, in the order they are applied."""
        checks: list[RiskCheck] = []
        if self.daily_loss_limit is not None:
            checks.append(DailyLossLimit(self.daily_loss_limit))
        if self.max_positions is not None:
            checks.append(MaxPositions(self.max_positions))
        checks.append(OnePerSymbol())
        if self.max_exposure is not None:
            checks.append(MaxExposure(self.max_exposure))
        return checks


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """A run's outputs.

    - `trades`: one row per position, in entry order (see `TRADE_COLUMNS`), then the signal's
      extra columns. `stake` is the USDT spent including the entry fee, `pnl` the USDT returned
      after the exit fee minus the stake, `net_return = pnl / stake`.
    - `equity`: the mark-to-market curve (`time, equity, cash, open_value, open_positions`).
    - `skipped`: signals not taken, by `SkipReason`.
    - `metrics`, `yearly`: see `backtest.metrics`.
    """

    trades: pl.DataFrame
    equity: pl.DataFrame
    skipped: dict[str, int]
    metrics: dict[str, float]
    yearly: pl.DataFrame


TRADE_COLUMNS = (
    "symbol",
    "signal_time",
    "entry_time",
    "exit_time",
    "exit_reason",
    "entry_ref",
    "entry_price",
    "exit_price",
    "quantity",
    "stake",
    "pnl",
    "net_return",
    "priority",
)
EQUITY_SCHEMA: dict[str, pl.DataType] = {
    "time": UTC_TIME,
    "equity": pl.Float64(),
    "cash": pl.Float64(),
    "open_value": pl.Float64(),
    "open_positions": pl.Int64(),
}


@dataclass(slots=True)
class _Open:
    """A position the engine is tracking until its exit fill arrives."""

    row: int
    seq: int
    entry: Fill
    entry_ref: float
    stake: float


@dataclass(slots=True)
class _Trade:
    row: int
    seq: int
    symbol: str
    entry_us: int
    exit_us: int
    exit_reason: str
    entry_ref: float
    entry_price: float
    exit_price: float
    quantity: float
    stake: float
    proceeds: float


@dataclass(slots=True)
class _Marks:
    """Per position: the open value at each curve point while it is open."""

    times: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)


def _prepare(signals: pl.DataFrame, config: BacktestConfig) -> tuple[pl.DataFrame, dict[str, int]]:
    """Validate the signal frame, drop signals outside the run, add entry times; sorted in
    processing order with a `_row` index into the original frame."""
    missing = [c for c in SIGNAL_COLUMNS if c not in signals.columns]
    if missing:
        msg = f"signals are missing columns: {', '.join(missing)}"
        raise ValueError(msg)
    dtype = signals.schema["open_time"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
        msg = f"signal open_time must be a timezone-aware datetime, got {dtype}"
        raise ValueError(msg)
    frame = signals.with_row_index("_row").with_columns(
        pl.col("open_time").dt.convert_time_zone("UTC").dt.cast_time_unit("us"),
        pl.col("priority").cast(pl.Float64),
    )
    skipped = {reason.value: 0 for reason in SkipReason}
    if config.start is not None:
        early = pl.col("open_time") < config.start
        skipped[SkipReason.BEFORE_START] = frame.filter(early).height
        frame = frame.filter(~early)
    if config.end is not None:
        late = pl.col("open_time") >= config.end
        skipped[SkipReason.AFTER_END] = frame.filter(late).height
        frame = frame.filter(~late)
    offset = (1 + config.entry_delay) * (config.interval.length // _US)
    frame = frame.with_columns(_entry=pl.col("open_time").dt.epoch("us") + offset).sort(
        "_entry", "priority", "symbol", nulls_last=True, maintain_order=True
    )
    return frame, skipped


class _Run:
    def __init__(self, source: CandleSource, config: BacktestConfig) -> None:
        self.source = source
        self.config = config
        self.broker = SimulatedBroker(cash=config.initial_capital, costs=config.costs)
        self.book = Book(cash=config.initial_capital)
        self.checks = config.risk_checks()
        self.open: dict[str, _Open] = {}
        self.trades: list[_Trade] = []
        self.marks = _Marks()
        self.every_us = config.mark_every // _US
        self.seq = 0

    def run(self, frame: pl.DataFrame, skipped: dict[str, int]) -> None:
        last_settled: int | None = None
        rows = frame["_row"].to_list()
        symbols = frame["symbol"].to_list()
        entries = frame["_entry"].to_list()
        for row, symbol, entry_us in zip(rows, symbols, entries, strict=True):
            entry_time = from_us(entry_us)
            if entry_us != last_settled:
                self._settle(entry_time)
                last_settled = entry_us
            self.book.roll_day(entry_time)
            stake = self.config.sizing.stake(self.book)
            reason = first_rejection(self.checks, self.book, symbol, stake)
            if reason is None and not 0 < stake <= self.book.cash * (1 + 1e-9):
                reason = SkipReason.CASH
            if reason is None and not self._enter(
                row, symbol, entry_time, min(stake, self.book.cash)
            ):
                reason = SkipReason.NO_ENTRY_CANDLE
            if reason is not None:
                skipped[reason] += 1
        self._settle(_FAR_FUTURE)

    def _settle(self, until: datetime) -> None:
        for fill in self.broker.settle(until):
            self._close(fill)

    def _load(
        self, symbol: str, start: datetime, span: timedelta
    ) -> tuple[Candles | None, Candles | None]:
        end = start + span
        data_end = self.source.data_end(self.config.interval)
        capped = data_end is not None and end >= data_end
        if data_end is not None and capped:
            end = data_end
        coarse = self.source.window(symbol, self.config.interval, start, end)
        fine = None
        if self.config.use_fine:
            fine = self.source.window(symbol, self.config.fine_interval, start, end)
        if capped:
            # The window reaches the data end: nothing more can come, so a position still open
            # there closes at the last close (`series_end`) rather than waiting for candles.
            coarse = None if coarse is None else replace(coarse, ended=True)
            fine = None if fine is None else replace(fine, ended=True)
        return coarse, fine

    def _enter(self, row: int, symbol: str, entry_time: datetime, stake: float) -> bool:
        plan = self.config.exit
        span = plan.max_hold if plan.max_hold is not None else self.config.window
        coarse, fine = self._load(symbol, entry_time, span)
        if coarse is None or len(coarse) == 0 or int(coarse.open_time[0]) != to_us(entry_time):
            return False
        self.broker.set_candles(symbol, coarse, fine)
        order = Order(symbol, Side.BUY, entry_time, quote_amount=stake, exit_plan=plan)
        entry = self.broker.place_order(order)
        assert entry is not None
        exit_fill = self.broker.pending_exit(symbol)
        while exit_fill is None:
            span *= 2
            if span > _MAX_WINDOW:
                msg = f"no exit for {symbol} entered {entry_time:%Y-%m-%d %H:%M} within {span}"
                raise RuntimeError(msg)
            coarse, fine = self._load(symbol, entry_time, span)
            assert coarse is not None
            self.broker.set_candles(symbol, coarse, fine)
            exit_fill = self.broker.pending_exit(symbol)
        self.book.open(symbol, stake, entry_time)
        position = self.broker.positions[symbol]
        self.open[symbol] = _Open(row, self.seq, entry, position.entry_ref, stake)
        self.seq += 1
        self._mark(coarse, position.entry_ref, entry.quantity, exit_fill)
        return True

    def _mark(
        self, candles: Candles, entry_ref: float, quantity: float, exit_fill: ExitFill
    ) -> None:
        """The position's value at each curve point while open (`entry <= t < exit`): quantity x
        the last close at or before the point (the entry open before any candle has closed)."""
        every = self.every_us
        entry_us, exit_us = int(candles.open_time[0]), to_us(exit_fill.time)
        first = -(-entry_us // every) * every
        if first >= exit_us:
            return
        times = np.arange(first, exit_us, every, dtype=np.int64)
        closed = candles.open_time + candles.length_us
        i = np.searchsorted(closed, times, side="right") - 1
        prices = np.where(i >= 0, candles.close[np.maximum(i, 0)], entry_ref)
        self.marks.times.append(times)
        self.marks.values.append(prices * quantity)

    def _close(self, fill: Fill) -> None:
        position = self.open.pop(fill.symbol)
        proceeds = fill.notional - fill.fee
        self.book.close(fill.symbol, proceeds, fill.time)
        assert fill.reason is not None
        self.trades.append(
            _Trade(
                row=position.row,
                seq=position.seq,
                symbol=fill.symbol,
                entry_us=to_us(position.entry.time),
                exit_us=to_us(fill.time),
                exit_reason=fill.reason.value,
                entry_ref=position.entry_ref,
                entry_price=position.entry.price,
                exit_price=fill.price,
                quantity=fill.quantity,
                stake=position.stake,
                proceeds=proceeds,
            )
        )

    def trade_frame(self, signals: pl.DataFrame) -> pl.DataFrame:
        trades = sorted(self.trades, key=lambda t: t.seq)
        frame = pl.DataFrame(
            {
                "_row": [t.row for t in trades],
                "symbol": [t.symbol for t in trades],
                "entry_time": [t.entry_us for t in trades],
                "exit_time": [t.exit_us for t in trades],
                "exit_reason": [t.exit_reason for t in trades],
                "entry_ref": [t.entry_ref for t in trades],
                "entry_price": [t.entry_price for t in trades],
                "exit_price": [t.exit_price for t in trades],
                "quantity": [t.quantity for t in trades],
                "stake": [t.stake for t in trades],
                "pnl": [t.proceeds - t.stake for t in trades],
            },
            schema={
                "_row": pl.UInt32,
                "symbol": pl.String,
                "entry_time": pl.Int64,
                "exit_time": pl.Int64,
                "exit_reason": pl.String,
                "entry_ref": pl.Float64,
                "entry_price": pl.Float64,
                "exit_price": pl.Float64,
                "quantity": pl.Float64,
                "stake": pl.Float64,
                "pnl": pl.Float64,
            },
        ).with_columns(
            pl.from_epoch("entry_time", "us").dt.replace_time_zone("UTC"),
            pl.from_epoch("exit_time", "us").dt.replace_time_zone("UTC"),
            net_return=pl.col("pnl") / pl.col("stake"),
        )
        extra = [c for c in signals.columns if c not in ("symbol", "_entry")]
        carried = signals.select(extra).rename({"open_time": "signal_time"})
        frame = frame.join(carried, on="_row", how="left", maintain_order="left")
        rest = [c for c in frame.columns if c not in TRADE_COLUMNS and c != "_row"]
        return frame.select(*TRADE_COLUMNS, *rest)

    def equity_frame(self) -> pl.DataFrame:
        """Mark-to-market equity at every `mark_every` boundary from the start of the first
        entry's period to the end of the last exit's period (both included)."""
        if not self.trades:
            return pl.DataFrame(schema=EQUITY_SCHEMA)
        every = self.every_us
        entry = np.sort(np.array([t.entry_us for t in self.trades], dtype=np.int64))
        stakes = np.array([t.stake for t in sorted(self.trades, key=lambda t: t.entry_us)])
        by_exit = sorted(self.trades, key=lambda t: t.exit_us)
        exit_ = np.array([t.exit_us for t in by_exit], dtype=np.int64)
        proceeds = np.array([t.proceeds for t in by_exit])
        t0 = int(entry[0]) // every * every
        t1 = -(-int(exit_[-1]) // every) * every
        grid = np.arange(t0, t1 + every, every, dtype=np.int64)
        n_in = np.searchsorted(entry, grid, side="right")
        n_out = np.searchsorted(exit_, grid, side="right")
        spent = np.concatenate(([0.0], np.cumsum(stakes)))[n_in]
        returned = np.concatenate(([0.0], np.cumsum(proceeds)))[n_out]
        cash = self.config.initial_capital - spent + returned
        open_value = np.zeros(len(grid))
        if self.marks.times:
            times = np.concatenate(self.marks.times)
            values = np.concatenate(self.marks.values)
            open_value = np.bincount((times - t0) // every, values, minlength=len(grid))
        return pl.DataFrame(
            {
                "time": grid,
                "equity": cash + open_value,
                "cash": cash,
                "open_value": open_value,
                "open_positions": (n_in - n_out).astype(np.int64),
            }
        ).with_columns(pl.from_epoch("time", "us").dt.replace_time_zone("UTC"))


def run_backtest(
    signals: pl.DataFrame, source: CandleSource, config: BacktestConfig
) -> BacktestResult:
    """Run `config` over `signals`, with candles from `source`. See the module docstring."""
    frame, skipped = _prepare(signals, config)
    run = _Run(source, config)
    run.run(frame, skipped)
    trades = run.trade_frame(frame)
    equity = run.equity_frame()
    metrics, yearly = summarize(trades, equity, config.initial_capital, config.mark_every)
    return BacktestResult(trades, equity, skipped, metrics, yearly)
