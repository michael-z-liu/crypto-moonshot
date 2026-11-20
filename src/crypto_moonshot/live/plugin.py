"""Strategy plugins: the protocol, the loader, the rule adapter and rolling backtests.

A plugin is any object that satisfies `Strategy`. It lives outside the platform and is loaded
by path at start-up (`load_strategy`). The same plugin is backtested (`rolling_signals`, or its
own `backtest_signals`), replayed and paper-traded by calling `on_candles` once per closed
candle, so what a backtest of a plugin finds is what the engine would have seen.

`on_candles` output frames
--------------------------

Required columns: `symbol` (String) and `priority` (number, lower first). Optional: `action`
(`enter`, the default, or `exit`), `weight` (> 0), exit-plan override columns (`take_profit`,
`stop`, `trailing`, `max_hold` as a duration or a string such as `90m`, `tp_fill`; a null keeps
the strategy's `exit` value for that field) and any extra columns, carried along as `info`.
Only signals for the candle `time` are allowed, so there is no `open_time` column: a frame
that has one is accepted when every value equals `time`, and the column is dropped.
`normalize_signal_frame` is the one place that checks and canonicalises such a frame.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import importlib.util
import re
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.models import KLINE_SCHEMA, Interval, as_utc, parse_duration
from crypto_moonshot.strategy.base import SignalRule, Signals
from crypto_moonshot.strategy.compute import compute_signals
from crypto_moonshot.strategy.features import MIN_HISTORY, bars

ACTIONS = ("enter", "exit")
EXIT_COLUMNS = ("take_profit", "stop", "trailing", "max_hold", "tp_fill")
CANONICAL_COLUMNS = ("symbol", "action", "priority", "weight", *EXIT_COLUMNS)
"""The columns of a normalized frame, in order; extra columns follow."""
_DURATION = pl.Duration("us")


class PluginError(Exception):
    """A strategy source could not be loaded, built or validated; the message names it."""


class SignalFrameError(ValueError):
    """An `on_candles` output frame breaks the contract."""


@runtime_checkable
class Strategy(Protocol):
    """What a plugin is. See `on_candles` for the contract."""

    name: str  # set from the config's instance name
    interval: Interval
    warmup: int  # candles of history each window holds (at least)
    context_symbols: tuple[str, ...]  # extra symbols whose windows it reads (e.g. BTCUSDT)
    exit: ExitPlan | None  # default bracket for its entries

    def on_candles(
        self,
        windows: Mapping[str, pl.DataFrame],
        time: datetime,
        context: Mapping[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Signals for the candle that opened at `time` (just closed).

        `windows`: the traded symbols' windows. `context`: the windows of `context_symbols`
        (a symbol may be in both). A window holds closed candles in `KLINE_SCHEMA`, oldest
        first, `warmup` of them at most, ending at the candle that opened at `time`, or earlier
        for a symbol whose candle at `time` is missing; holes are possible. Returns a frame as
        described in the module docstring.
        """
        ...


@runtime_checkable
class BacktestSignalsProvider(Protocol):
    """An optional fast batch computation over stored history (`RuleStrategy` has one)."""

    def backtest_signals(
        self, root: Path, symbols: Sequence[str], until: datetime | None
    ) -> Signals: ...


# --------------------------------------------------------------------------------------
# Output frames
# --------------------------------------------------------------------------------------


def empty_signal_frame() -> pl.DataFrame:
    """A normalized frame with no signals."""
    return pl.DataFrame(
        schema={
            "symbol": pl.String,
            "action": pl.String,
            "priority": pl.Float64,
            "weight": pl.Float64,
            "take_profit": pl.Float64,
            "stop": pl.Float64,
            "trailing": pl.Float64,
            "max_hold": _DURATION,
            "tp_fill": pl.String,
        }
    )


def normalize_signal_frame(frame: pl.DataFrame, time: datetime) -> pl.DataFrame:
    """Check an `on_candles` output frame for candle `time` and return it canonical.

    The result has `CANONICAL_COLUMNS` (absent optional ones null, `action` "enter") with fixed
    dtypes, then the extra columns as given. A frame with no columns and no rows is a valid
    "no signals". Raises `SignalFrameError` for a missing or mistyped required column, a null
    or NaN priority, an unknown action, a weight that is not a positive number, an invalid exit
    override, an `open_time` column that is not `time`, and more than one signal for the same
    symbol and action.
    """
    time = as_utc(time)
    if frame.width == 0 and frame.height == 0:
        return empty_signal_frame()
    missing = [c for c in ("symbol", "priority") if c not in frame.columns]
    if missing:
        msg = f"signal frame is missing required columns: {', '.join(missing)}"
        raise SignalFrameError(msg)
    if "open_time" in frame.columns:
        stamps = frame["open_time"]
        if not isinstance(stamps.dtype, pl.Datetime) or stamps.dtype.time_zone is None:
            msg = f"signal open_time must be a timezone-aware datetime, got {stamps.dtype}"
            raise SignalFrameError(msg)
        if (stamps != time).any():
            msg = f"signals may only be for the candle {time.isoformat()}: open_time differs"
            raise SignalFrameError(msg)
        frame = frame.drop("open_time")

    if frame.schema["symbol"] != pl.String:
        msg = f"signal symbol must be a string, got {frame.schema['symbol']}"
        raise SignalFrameError(msg)
    if frame["symbol"].null_count():
        raise SignalFrameError("signal symbol must not be null")

    priority = _number(frame, "priority")
    if priority.null_count() or priority.is_nan().any():
        raise SignalFrameError("signal priority must be a number, not null or NaN")

    if "action" in frame.columns:
        action = frame["action"]
        if action.dtype != pl.String or action.null_count():
            msg = "signal action must be 'enter' or 'exit' (string, not null)"
            raise SignalFrameError(msg)
        bad = sorted(set(action.to_list()) - set(ACTIONS))
        if bad:
            msg = f"unknown signal action {bad[0]!r}: use 'enter' or 'exit'"
            raise SignalFrameError(msg)
    else:
        action = pl.Series("action", ["enter"] * frame.height, dtype=pl.String)

    weight = _number(frame, "weight") if "weight" in frame.columns else _nulls("weight", frame)
    if weight.is_not_null().any() and (
        (weight <= 0).any() or weight.is_nan().any() or weight.is_infinite().any()
    ):
        raise SignalFrameError("signal weight must be a finite number > 0 (or null)")

    levels = [
        _number(frame, c) if c in frame.columns else _nulls(c, frame) for c in EXIT_COLUMNS[:3]
    ]
    hold = _hold(frame)
    fill = _fill(frame)
    _check_overrides(frame.height, levels, hold, fill)

    head = pl.DataFrame(
        [
            frame["symbol"],
            action.alias("action"),
            priority.alias("priority"),
            weight,
            *levels,
            hold,
            fill,
        ]
    )
    extras = [c for c in frame.columns if c not in CANONICAL_COLUMNS]
    out = pl.concat([head, frame.select(extras)], how="horizontal") if extras else head
    if out.select(["symbol", "action"]).is_duplicated().any():
        dup = out.filter(pl.struct("symbol", "action").is_duplicated()).row(0, named=True)
        msg = f"more than one {dup['action']} signal for {dup['symbol']} in one candle"
        raise SignalFrameError(msg)
    return out


def _number(frame: pl.DataFrame, name: str) -> pl.Series:
    column = frame[name]
    if not column.dtype.is_numeric() and column.dtype != pl.Null:
        msg = f"signal {name} must be numeric, got {column.dtype}"
        raise SignalFrameError(msg)
    return column.cast(pl.Float64).alias(name)


def _nulls(name: str, frame: pl.DataFrame) -> pl.Series:
    return pl.Series(name, [None] * frame.height, dtype=pl.Float64)


def _hold(frame: pl.DataFrame) -> pl.Series:
    if "max_hold" not in frame.columns:
        return pl.Series("max_hold", [None] * frame.height, dtype=_DURATION)
    column = frame["max_hold"]
    try:
        if isinstance(column.dtype, pl.Duration):
            return column.cast(_DURATION).alias("max_hold")
        if column.dtype == pl.String:
            parsed = [None if v is None else parse_duration(v) for v in column.to_list()]
            return pl.Series("max_hold", parsed, dtype=_DURATION)
        if column.dtype == pl.Null:
            return pl.Series("max_hold", [None] * frame.height, dtype=_DURATION)
    except ValueError as exc:
        raise SignalFrameError(f"signal max_hold: {exc}") from None
    msg = f"signal max_hold must be a duration or a string such as '90m', got {column.dtype}"
    raise SignalFrameError(msg)


def _fill(frame: pl.DataFrame) -> pl.Series:
    if "tp_fill" not in frame.columns:
        return pl.Series("tp_fill", [None] * frame.height, dtype=pl.String)
    column = frame["tp_fill"]
    if column.dtype not in (pl.String, pl.Null):
        msg = f"signal tp_fill must be a string, got {column.dtype}"
        raise SignalFrameError(msg)
    return column.cast(pl.String).alias("tp_fill")


def _check_overrides(n: int, levels: list[pl.Series], hold: pl.Series, fill: pl.Series) -> None:
    for i in range(n):
        values = [s[i] for s in levels]
        try:
            ExitPlan(
                take_profit=values[0],
                stop=values[1],
                trailing=values[2],
                max_hold=hold[i],
                tp_fill=fill[i] or "touch",
            )
        except ValueError as exc:
            raise SignalFrameError(f"invalid exit override: {exc}") from None


# --------------------------------------------------------------------------------------
# The adapter for rule strategies
# --------------------------------------------------------------------------------------


def default_warmup(interval: Interval) -> int:
    """Candles of history a window needs for the capitulation rules to match their
    full-history signals: `MIN_HISTORY` (8 days: the 7-day rolling high plus one day).

    `features.spikes` gives nothing for a series shorter than `MIN_HISTORY` stored candles and
    skips the first `WARMUP` (7 days) of the gap-free grid; every feature of a candle looks back
    at most 7 days plus the shifted bar. A window of `MIN_HISTORY` candles that ends at the
    signal candle therefore holds the whole 7-day lookback of that candle, and (being full)
    passes the length check, so its last row has the full-history value of every feature.
    Holes only make the window span more time, never less.
    """
    return bars(MIN_HISTORY, interval)


class RuleStrategy:
    """Adapts a `SignalRule` to `Strategy`.

    `on_candles` runs `rule.candidates` on each traded symbol's window, joins `rule.context` on
    `open_time` as `strategy.compute.candidates` does, applies `rule.select` and returns the
    signals whose `open_time` is `time`, without the `open_time` column (the frame is for one
    candle). `backtest_signals` runs `compute_signals`, the fast path over the whole history.

    Parity with the full-history computation holds for the capitulation rules with the default
    warm-up (`default_warmup`), with one exception, an unavoidable edge case: a symbol in its
    first `MIN_HISTORY` of stored candles. The full-series computation applies the length check
    to the symbol's whole series, so it signals on candles from the 8th day of the grid on once
    the series ever reaches 8 days, while a window (or live) cannot know the future length and
    gives nothing until it holds `warmup` candles. So signals on a symbol's candles numbered
    `WARMUP` to `MIN_HISTORY` (days 7 to 8 after its first candle) appear in
    `backtest_signals` but not in `on_candles`.
    """

    def __init__(
        self,
        rule: SignalRule,
        *,
        name: str | None = None,
        warmup: int | None = None,
        cache_dir: Path | None = None,
        workers: int | None = None,
    ) -> None:
        self.rule = rule
        self.name = rule.name if name is None else name
        self.interval = rule.interval
        self.context_symbols = tuple(rule.context_symbols)
        self.exit: ExitPlan | None = rule.exit
        if warmup is None:
            warmup = getattr(rule, "warmup", None)
        if warmup is None:
            try:
                warmup = default_warmup(rule.interval)
            except ValueError as exc:
                msg = f"{rule.name}: pass warmup, no default for {rule.interval.value} ({exc})"
                raise ValueError(msg) from None
        if warmup < 1:
            msg = f"warmup must be at least 1, got {warmup}"
            raise ValueError(msg)
        self.warmup = int(warmup)
        self.cache_dir = cache_dir
        self.workers = workers

    def on_candles(
        self,
        windows: Mapping[str, pl.DataFrame],
        time: datetime,
        context: Mapping[str, pl.DataFrame],
    ) -> pl.DataFrame:
        time = as_utc(time)
        parts = []
        for symbol, window in windows.items():
            part = self.rule.candidates(window)
            part = part.select(
                pl.lit(symbol, dtype=pl.String).alias("symbol"), pl.exclude("symbol")
            )
            parts.append(part.filter(pl.col("open_time") == time))
        if not parts:
            parts = [
                self.rule.candidates(pl.DataFrame(schema=KLINE_SCHEMA)).select(
                    pl.lit("", dtype=pl.String).alias("symbol"), pl.exclude("symbol")
                )
            ]
        table = pl.concat(parts, how="vertical_relaxed")
        if self.context_symbols:
            frame = self.rule.context({s: context[s] for s in self.context_symbols if s in context})
            if frame is not None:
                table = table.join(frame, on="open_time", how="left", maintain_order="left")
        chosen = self.rule.select(table)
        return chosen.filter(pl.col("open_time") == time).drop("open_time")

    def backtest_signals(
        self, root: Path, symbols: Sequence[str], until: datetime | None
    ) -> Signals:
        run = compute_signals(
            self.rule,
            symbols,
            root,
            until=until,
            workers=self.workers,
            cache_dir=self.cache_dir,
        )
        return run.signals


# --------------------------------------------------------------------------------------
# Rolling backtest
# --------------------------------------------------------------------------------------


def _ends(series: Mapping[str, pl.DataFrame]) -> dict[str, np.ndarray]:
    return {
        s: frame["open_time"].dt.epoch("us").to_numpy() if frame.height else np.empty(0, np.int64)
        for s, frame in series.items()
    }


def rolling_signals(
    strategy: Strategy,
    series: Mapping[str, pl.DataFrame],
    context: Mapping[str, pl.DataFrame],
    start: datetime,
    end: datetime,
    *,
    stale_after: int | None = 2,
    on_tick: Callable[[datetime], None] | None = None,
) -> Signals:
    """Signals from running `strategy.on_candles` at every candle time in `[start, end)`.

    Each call gets, per traded symbol in `series`, a window of the last `strategy.warmup`
    candles up to the candle at the time (a slice of the whole series: nothing is copied or
    re-read), exactly as the engine would; `context` windows are cut the same way for
    `strategy.context_symbols` (a context symbol with no candles yet gets an empty frame). A
    traded symbol is left out of a tick when it has no candle yet, or when its latest candle is
    more than `stale_after` intervals old (None: never). Times at which no traded symbol has a
    candle are skipped. This is slower than `backtest_signals` but gives the live signals.

    The result is a `Signals` frame: `symbol`, `open_time`, `priority`, then `weight` and the
    exit-override columns when any signal sets them, then the extra columns; only "enter"
    signals are kept (the backtest engine has no exit signals). The exit plan is
    `strategy.exit` (an empty plan when None).
    """
    interval = strategy.interval
    start, end = as_utc(start), as_utc(end)
    length_us = interval.length // timedelta(microseconds=1)
    traded = {s: f.sort("open_time") for s, f in series.items()}
    ctx = {
        s: context[s].sort("open_time") if s in context else pl.DataFrame(schema=KLINE_SCHEMA)
        for s in strategy.context_symbols
    }
    traded_ends = _ends(traded)
    context_ends = _ends(ctx)
    have = set(np.unique(np.concatenate([*traded_ends.values(), np.empty(0, np.int64)])).tolist())
    parts: list[pl.DataFrame] = []
    for time in interval.open_times(start, end):
        us = int(time.timestamp()) * 1_000_000 + time.microsecond
        if us not in have:  # no traded symbol has this candle: the engine would not tick
            continue
        windows = _windows(traded, traded_ends, us, strategy.warmup, length_us, stale_after)
        ctx_windows = _windows(ctx, context_ends, us, strategy.warmup, length_us, None, keep=True)
        frame = normalize_signal_frame(strategy.on_candles(windows, time, ctx_windows), time)
        if on_tick is not None:
            on_tick(time)
        frame = frame.filter(pl.col("action") == "enter")
        if frame.height:
            parts.append(frame.with_columns(open_time=pl.lit(time, dtype=pl.Datetime("us", "UTC"))))
    plan = strategy.exit or ExitPlan()
    if not parts:
        empty = empty_signal_frame().drop("action", *EXIT_COLUMNS, "weight")
        frame = empty.with_columns(open_time=pl.lit(None, dtype=pl.Datetime("us", "UTC")))
        return Signals(frame.select("symbol", "open_time", "priority"), plan, interval)
    table = pl.concat(parts, how="diagonal_relaxed").drop("action")
    optional = [c for c in ("weight", *EXIT_COLUMNS) if table[c].null_count() == table.height]
    table = table.drop(optional)
    front = ["symbol", "open_time", "priority"]
    return Signals(table.select(*front, pl.exclude(front)), plan, interval)


def _windows(
    series: Mapping[str, pl.DataFrame],
    ends: Mapping[str, np.ndarray],
    us: int,
    warmup: int,
    length_us: int,
    stale_after: int | None,
    *,
    keep: bool = False,
) -> dict[str, pl.DataFrame]:
    out: dict[str, pl.DataFrame] = {}
    for symbol, frame in series.items():
        stamps = ends[symbol]
        stop = int(np.searchsorted(stamps, us, side="right"))
        if stop == 0:
            if keep:
                out[symbol] = frame.head(0)
            continue
        if stale_after is not None and us - int(stamps[stop - 1]) > stale_after * length_us:
            continue
        out[symbol] = frame.slice(max(0, stop - warmup), min(warmup, stop))
    return out


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

_SPEC_RE = re.compile(
    r"^(?P<module>.+):(?P<attr>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)$"
)
_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _is_rule(obj: object) -> bool:
    return all(callable(getattr(obj, n, None)) for n in ("candidates", "context", "select", "key"))


def _is_instance(obj: object) -> bool:
    """An already-built strategy or rule (not a class or a factory)."""
    if isinstance(obj, type):
        return False
    return callable(getattr(obj, "on_candles", None)) or _is_rule(obj)


def _import_file(path: Path, source: str) -> ModuleType:
    if not path.is_file():
        msg = f"{source}: file {path} not found"
        raise PluginError(msg)
    name = f"_moonshot_plugin_{path.stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"{source}: {path} cannot be loaded as a Python module"
        raise PluginError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec: dataclasses look the module up by name
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        msg = f"{source}: importing {path} failed: {type(exc).__name__}: {exc}"
        raise PluginError(msg) from exc
    return module


def _import_module(module: str, source: str) -> ModuleType:
    if not _MODULE_RE.match(module):
        msg = f"{source}: {module!r} is not a module name or a .py file path"
        raise PluginError(msg)
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        msg = f"{source}: cannot import module {module!r}: {exc}"
        raise PluginError(msg) from exc
    except Exception as exc:
        msg = f"{source}: importing {module!r} failed: {type(exc).__name__}: {exc}"
        raise PluginError(msg) from exc


def _problems(obj: object) -> list[str]:
    problems: list[str] = []
    if not isinstance(getattr(obj, "name", None), str):
        problems.append("name must be a string")
    if not isinstance(getattr(obj, "interval", None), Interval):
        problems.append("interval must be an Interval")
    warmup = getattr(obj, "warmup", None)
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 1:
        problems.append("warmup must be an integer >= 1")
    symbols = getattr(obj, "context_symbols", None)
    if not isinstance(symbols, tuple) or not all(isinstance(s, str) for s in symbols):
        problems.append("context_symbols must be a tuple of strings")
    if not hasattr(obj, "exit"):
        problems.append("exit is missing (an ExitPlan, or None)")
    elif obj.exit is not None and not isinstance(obj.exit, ExitPlan):
        problems.append("exit must be an ExitPlan or None")
    if not callable(getattr(obj, "on_candles", None)):
        problems.append("on_candles(windows, time, context) is missing")
    return problems


def load_strategy(
    source: str, *, name: str, params: Mapping[str, object] | None = None
) -> Strategy:
    """Load a plugin from `package.module:attr` or `path/to/file.py:attr`.

    A file is imported under a unique module name, without putting its folder on `sys.path`.
    If `attr` is a class or factory (callable, and not already a strategy or rule instance) it
    is called with `**params`; params for anything else are an error. A `SignalRule` is wrapped
    in `RuleStrategy`. The strategy's `name` is set to `name` (an instance found in a module is
    copied first, so the module's own object is untouched). The result is checked against the
    protocol. Every failure is a `PluginError` whose message starts with `source`.
    """
    spec = _SPEC_RE.match(source.strip())
    if spec is None:
        msg = f"{source!r}: expected package.module:attr or path/to/file.py:attr"
        raise PluginError(msg)
    where = spec["module"]
    if where.endswith(".py") or "/" in where or "\\" in where:
        module = _import_file(Path(where).expanduser(), source)
    else:
        module = _import_module(where, source)
    obj: object = module
    for part in spec["attr"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            msg = f"{source}: {where} has no attribute {spec['attr']!r}"
            raise PluginError(msg) from None

    if _is_instance(obj):
        if params:
            msg = (
                f"{source}: params given, but {spec['attr']} is an instance, not a class or factory"
            )
            raise PluginError(msg)
        built = obj
    elif callable(obj):
        try:
            built = obj(**(params or {}))
        except Exception as exc:
            msg = f"{source}: calling {spec['attr']} failed: {type(exc).__name__}: {exc}"
            raise PluginError(msg) from exc
    else:
        msg = f"{source}: {spec['attr']} is neither callable nor a strategy or rule"
        raise PluginError(msg)

    if not hasattr(built, "on_candles") and _is_rule(built):
        try:
            strategy: object = RuleStrategy(built, name=name)  # type: ignore[arg-type]
        except ValueError as exc:
            msg = f"{source}: {exc}"
            raise PluginError(msg) from exc
    else:
        if obj is built:
            with contextlib.suppress(Exception):  # an uncopyable plugin is used as it is
                built = copy.copy(built)
        try:
            built.name = name  # type: ignore[attr-defined]
        except Exception as exc:
            msg = f"{source}: cannot set the strategy's name: {type(exc).__name__}: {exc}"
            raise PluginError(msg) from exc
        strategy = built
    problems = _problems(strategy)
    if problems:
        msg = f"{source}: not a strategy: {'; '.join(problems)}"
        raise PluginError(msg)
    assert isinstance(strategy, Strategy)
    return strategy
