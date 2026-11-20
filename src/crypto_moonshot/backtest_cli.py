"""`moonshot backtest`: run a strategy over the stored candles and write the results.

Commands: `f2`, `capitulation` (rule strategies on a universe), `scores` (a precomputed score
file) and `convert-scores` (research walk-forward `.npz` to a score file). Every run command
takes the same options; the strategy only sets the defaults (universe, sizing, exits, signal
range), so `moonshot backtest f2 --until 2026-11-02` is the golden F2 case and
`moonshot backtest capitulation` the golden R2 case (`docs/backtest-validation.md`).

A run writes, into `--out` (default `results/<strategy>-<UTC timestamp>/`): `trades.parquet` and
`trades.csv`, `equity.csv`, `yearly.csv` and `metrics.json` (overall metrics, skipped signals by
reason, the full configuration and the data end used). Signal caches go to `--cache-dir`
(default `.cache/` in the current directory), never into the data folder. Progress notes go to
stderr; the summary (or `--json`) to stdout.

Exit codes: 0 ok, 1 the run could not start (no candles, no signals source), 2 usage error.
"""

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import typer

from crypto_moonshot.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from crypto_moonshot.backtest.source import ParquetCandleSource
from crypto_moonshot.broker.base import Costs, ExitPlan
from crypto_moonshot.cli_output import print_backtest_summary, stdout_is_tty
from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.data.universe import Universe, universe_symbols
from crypto_moonshot.data_cli import parse_when, resolve_settings
from crypto_moonshot.live.plugin import (
    BacktestSignalsProvider,
    PluginError,
    RuleStrategy,
    Strategy,
    load_strategy,
    rolling_signals,
)
from crypto_moonshot.models import Interval, parse_duration
from crypto_moonshot.portfolio import EqualSlots, FixedStake, Sizing
from crypto_moonshot.strategy.base import Signals
from crypto_moonshot.strategy.capitulation import F2, R2, R2_LAST_ENTRY, CapitulationRule
from crypto_moonshot.strategy.compute import compute_signals
from crypto_moonshot.strategy.scores import BASE_CNN, ScoreStrategy, convert_wf_npz, read_scores

backtest_app = typer.Typer(
    no_args_is_help=True, help="Backtest a strategy on the stored candles, or prepare its inputs."
)

DEFAULT_CAPITAL = 10_000.0
DEFAULT_CACHE_DIR = Path(".cache")


class TpFill(StrEnum):
    TOUCH = "touch"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class Preset:
    """What a strategy command defaults to."""

    universe: Universe | None
    exit: ExitPlan
    max_positions: int
    slots: int | None
    stake: float | None
    end: datetime | None


PRESETS = {
    "f2": Preset(Universe.POOL, F2.exit, 5, 5, None, None),
    "capitulation": Preset(Universe.ALL, R2.exit, 200, None, 50.0, R2_LAST_ENTRY),
    "scores": Preset(None, BASE_CNN.exit, 5, 5, None, None),
}
RULES: dict[str, CapitulationRule] = {"f2": F2, "capitulation": R2}


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def parse_hold(text: str) -> timedelta | None:
    """`1h`, `90m`, `3d`, `1.5h`, `2w` or `30s` (units s m h d w); `none` for no time exit."""
    try:
        return parse_duration(text)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--hold") from None


def build_exit(
    base: ExitPlan,
    *,
    hold: str | None,
    take_profit: float | None,
    stop: float | None,
    trailing: float | None,
    tp_fill: TpFill | None,
) -> ExitPlan:
    """The preset's exit plan with the given overrides (a level of 0 switches it off)."""
    changes: dict[str, Any] = {}
    if hold is not None:
        changes["max_hold"] = parse_hold(hold)
    for name, value in (("take_profit", take_profit), ("stop", stop), ("trailing", trailing)):
        if value is not None:
            changes[name] = value or None
    if tp_fill is not None:
        changes["tp_fill"] = tp_fill.value
    try:
        return replace(base, **changes)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def build_sizing(preset: Preset, slots: int | None, stake: float | None) -> Sizing:
    """`--slots` (equal-slot compounding) or `--stake` (fixed), else the preset's choice."""
    if slots is not None and stake is not None:
        msg = "--slots and --stake cannot be used together: choose compounding or a fixed stake"
        raise typer.BadParameter(msg, param_hint="--slots")
    try:
        if slots is not None:
            return EqualSlots(slots)
        if stake is not None:
            return FixedStake(stake)
        if preset.slots is not None:
            return EqualSlots(preset.slots)
        assert preset.stake is not None
        return FixedStake(preset.stake)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def _config_error(build: Callable[[], BacktestConfig]) -> BacktestConfig:
    try:
        return build()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------


def _note(text: str) -> None:
    typer.echo(text, err=True)


def _fail(text: str) -> typer.Exit:
    typer.echo(f"error: {text}", err=True)
    return typer.Exit(1)


def _signals(
    strategy: str,
    universe: Universe | None,
    root: Path,
    until: datetime | None,
    scores: Path | None,
    threshold: float | None,
    exit_plan: ExitPlan,
    cache_dir: Path,
    workers: int | None,
) -> Signals:
    if strategy == "scores":
        assert scores is not None
        if not scores.is_file():
            raise typer.BadParameter(f"{scores} is not a file", param_hint="--scores")
        rule = ScoreStrategy(
            BASE_CNN.threshold if threshold is None else threshold, exit_plan, name="scores"
        )
        try:
            return rule.signals(read_scores(scores))
        except (ValueError, pl.exceptions.PolarsError) as exc:
            raise typer.BadParameter(f"{scores}: {exc}", param_hint="--scores") from None
    assert universe is not None
    try:
        symbols = universe_symbols(universe, root)
    except FileNotFoundError as exc:
        raise _fail(f"{exc.filename} not found: the {universe} universe needs it") from None
    if not symbols:
        raise _fail(f"no 5m candles stored in {root} for the {universe} universe")
    _note(f"signals: {strategy} on {len(symbols)} symbols ({universe})")
    rule_ = RULES[strategy]
    try:
        run = compute_signals(
            rule_, symbols, root, until=until, cache_dir=cache_dir, workers=workers
        )
    except (ValueError, FileNotFoundError) as exc:
        raise _fail(str(exc)) from None
    _note(f"signals: {len(run.signals)} ({'cached' if run.cached else f'{run.seconds:.1f}s'})")
    return replace(run.signals, exit=exit_plan)


def _clean(value: Any) -> Any:
    """JSON-safe: NaN and infinity become null, datetimes ISO strings, timedeltas seconds."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    return value


def _sizing_dict(sizing: Sizing) -> dict[str, Any]:
    if isinstance(sizing, EqualSlots):
        return {"type": "equal_slots", "slots": sizing.slots}
    if isinstance(sizing, FixedStake):
        return {"type": "fixed_stake", "stake": sizing.usdt}
    return {"type": type(sizing).__name__}


def config_dict(config: BacktestConfig, source: ParquetCandleSource) -> dict[str, Any]:
    """The run's full configuration, as stored in `metrics.json`."""
    plan = config.exit
    return {
        "interval": config.interval.value,
        "fine_interval": config.fine_interval.value if config.use_fine else None,
        "use_fine": config.use_fine,
        "entry_delay": config.entry_delay,
        "sizing": _sizing_dict(config.sizing),
        "initial_capital": config.initial_capital,
        "max_positions": config.max_positions,
        "start": config.start,
        "end": config.end,
        "until": source.until,
        "data_end": source.data_end(config.interval),
        "exit": {
            "take_profit": plan.take_profit,
            "stop": plan.stop,
            "trailing": plan.trailing,
            "max_hold_seconds": plan.max_hold,
            "tp_fill": plan.tp_fill,
        },
        "costs": {
            "fee": config.costs.fee,
            "slippage": config.costs.slippage,
            "latency": config.costs.latency,
        },
    }


def write_results(out: Path, result: BacktestResult, document: dict[str, Any]) -> None:
    """Write the run's files into `out` (created if needed)."""
    out.mkdir(parents=True, exist_ok=True)
    result.trades.write_parquet(out / "trades.parquet")
    result.trades.write_csv(out / "trades.csv")
    result.equity.write_csv(out / "equity.csv")
    result.yearly.write_csv(out / "yearly.csv")
    (out / "metrics.json").write_text(json.dumps(_clean(document), indent=2) + "\n", "utf-8")


@dataclass(frozen=True, slots=True)
class Options:
    """The command line's options, as given (`None`: use the strategy's default)."""

    data_dir: Path | None
    universe: Universe | None
    until: str | None
    start: str | None
    end: str | None
    slots: int | None
    stake: float | None
    capital: float
    max_positions: int | None
    hold: str | None
    take_profit: float | None
    stop: float | None
    trailing: float | None
    tp_fill: TpFill | None
    fee: float
    slippage: float
    latency: float
    fine: bool
    entry_delay: int
    scores: Path | None
    threshold: float | None
    out: Path | None
    json_output: bool
    cache_dir: Path
    workers: int | None


def _prepare(
    options: Options,
    preset: Preset,
    *,
    interval: Interval = Interval.M5,
    require_exit: bool = False,
) -> tuple[BacktestConfig, datetime | None]:
    """The run's config from the options and the strategy's defaults, and the data end."""
    start = parse_when(options.start, option="--start") if options.start else None
    end = parse_when(options.end, option="--end") if options.end else preset.end
    until = parse_when(options.until, option="--until") if options.until else None
    exit_plan = build_exit(
        preset.exit,
        hold=options.hold,
        take_profit=options.take_profit,
        stop=options.stop,
        trailing=options.trailing,
        tp_fill=options.tp_fill,
    )
    if require_exit and exit_plan == replace(ExitPlan(), tp_fill=exit_plan.tp_fill):
        msg = "no exit plan: give at least one of --hold, --take-profit, --stop or --trailing"
        raise typer.BadParameter(msg, param_hint="--hold")
    sizing = build_sizing(preset, options.slots, options.stake)
    max_positions = options.max_positions
    if max_positions is None:
        max_positions = sizing.slots if isinstance(sizing, EqualSlots) else preset.max_positions
    config = _config_error(
        lambda: BacktestConfig(
            exit=exit_plan,
            sizing=sizing,
            max_positions=max_positions,
            initial_capital=options.capital,
            costs=Costs(options.fee, options.slippage, options.latency),
            start=start,
            end=end,
            interval=interval,
            use_fine=options.fine,
            entry_delay=options.entry_delay,
        )
    )
    return config, until


def _data_root(options: Options) -> Path:
    root = resolve_settings(options.data_dir).data_dir
    if not KlineStore(root).list_series():
        raise _fail(f"no candles stored in {root}; download some with `moonshot data sync`")
    return root


def _execute(
    label: str,
    options: Options,
    config: BacktestConfig,
    signals: Signals,
    root: Path,
    until: datetime | None,
    head: dict[str, Any],
) -> None:
    """Run the backtest, write the results and print the summary (or `--json`). `head` leads
    the metrics document."""
    source = ParquetCandleSource(root, until=until)
    _note(f"backtest: {len(signals)} signals")
    result = run_backtest(signals.frame, source, config)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = options.out or Path("results") / f"{label}-{stamp}"
    document: dict[str, Any] = {
        **head,
        "signals": len(signals),
        "created": datetime.now(UTC),
        "metrics": result.metrics,
        "skipped": result.skipped,
        "config": config_dict(config, source),
        "out": str(out),
    }
    write_results(out, result, document)
    if options.json_output:
        typer.echo(json.dumps(_clean(document)))
    else:
        _note(f"results written to {out}")
        print_backtest_summary(result.metrics, result.skipped, result.yearly, tty=stdout_is_tty())


def run_command(strategy: str, options: Options) -> None:
    """Everything behind `moonshot backtest <strategy>` once the options are collected."""
    preset = PRESETS[strategy]
    if strategy == "scores":
        if options.scores is None:
            raise typer.BadParameter("--scores FILE is required", param_hint="--scores")
        if options.universe is not None:
            raise typer.BadParameter("scores need no universe", param_hint="--universe")
    else:
        if options.scores is not None or options.threshold is not None:
            msg = f"{strategy} is a rule: --scores and --threshold are for `backtest scores`"
            raise typer.BadParameter(msg, param_hint="--scores")
    config, until = _prepare(options, preset)
    root = _data_root(options)
    universe = options.universe or preset.universe
    signals = _signals(
        strategy,
        universe,
        root,
        until,
        options.scores,
        options.threshold,
        config.exit,
        options.cache_dir,
        options.workers,
    )
    head = {
        "strategy": strategy,
        "universe": None if universe is None else universe.value,
        "scores": None if options.scores is None else str(options.scores),
        "threshold": (
            None
            if strategy != "scores"
            else BASE_CNN.threshold
            if options.threshold is None
            else options.threshold
        ),
    }
    _execute(strategy, options, config, signals, root, until, head)


# Option declarations shared by every run command.
DataDirOpt = Annotated[
    Path | None, typer.Option(help="Data folder (default: $MOONSHOT_DATA_DIR, else ./data).")
]
UntilOpt = Annotated[
    str | None,
    typer.Option(help="Data end, exclusive (YYYY-MM-DD or ISO datetime): ignore later."),
]
StartOpt = Annotated[str | None, typer.Option(help="Take signals from this open time (inclusive).")]
EndOpt = Annotated[str | None, typer.Option(help="Take signals before this open time (exclusive).")]
SlotsOpt = Annotated[
    int | None, typer.Option(min=1, help="Equal-slot compounding: stake = equity / SLOTS.")
]
StakeOpt = Annotated[float | None, typer.Option(min=0.0, help="Fixed stake per entry in USDT.")]
CapitalOpt = Annotated[float, typer.Option(min=0.0, help="Starting capital in USDT.")]
MaxPositionsOpt = Annotated[
    int | None,
    typer.Option(min=1, help="Most positions open at once (default: the slots, or 200)."),
]
HoldOpt = Annotated[
    str | None, typer.Option(help="Time exit, e.g. 1h, 90m, 3d, 2w; 'none' for no limit.")
]
TakeProfitOpt = Annotated[
    float | None, typer.Option(min=0.0, help="Take-profit as a fraction (0.2 = +20%).")
]
StopOpt = Annotated[
    float | None, typer.Option(min=0.0, max=0.999, help="Stop as a fraction (0 = off).")
]
TrailingOpt = Annotated[
    float | None, typer.Option(min=0.0, max=0.999, help="Trailing stop as a fraction (0 = off).")
]
TpFillOpt = Annotated[
    TpFill | None,
    typer.Option(help="Take profit fills when a candle touches the target, or closes."),
]
FeeOpt = Annotated[float, typer.Option(min=0.0, max=0.5, help="Fee per side.")]
SlippageOpt = Annotated[float, typer.Option(min=0.0, max=0.5, help="Slippage per side.")]
LatencyOpt = Annotated[float, typer.Option(min=0.0, max=0.5, help="Latency cost on market orders.")]
FineOpt = Annotated[
    bool, typer.Option("--fine/--no-fine", help="Fill exits on 1m candles when stored.")
]
EntryDelayOpt = Annotated[int, typer.Option(min=0, help="Extra candles between signal and entry.")]
OutOpt = Annotated[
    Path | None,
    typer.Option(help="Output folder (default: results/<strategy>-<UTC timestamp>/)."),
]
JsonOpt = Annotated[
    bool, typer.Option("--json", help="Print the results as one JSON object on stdout.")
]
CacheDirOpt = Annotated[Path, typer.Option(help="Where signal caches go (never the data folder).")]
WorkersOpt = Annotated[int | None, typer.Option(min=1, help="Processes for signal computation.")]


def _register(strategy: str, summary: str) -> None:
    def command(
        data_dir: DataDirOpt = None,
        universe: Annotated[
            Universe | None,
            typer.Option(
                help="Symbols to trade: pool (top 200 minus stablecoins and tokenised stocks) or "
                "all (plus delisted and outside the pool). Default: the strategy's own."
            ),
        ] = None,
        until: UntilOpt = None,
        start: StartOpt = None,
        end: EndOpt = None,
        slots: SlotsOpt = None,
        stake: StakeOpt = None,
        capital: CapitalOpt = DEFAULT_CAPITAL,
        max_positions: MaxPositionsOpt = None,
        hold: HoldOpt = None,
        take_profit: TakeProfitOpt = None,
        stop: StopOpt = None,
        trailing: TrailingOpt = None,
        tp_fill: TpFillOpt = None,
        fee: FeeOpt = 0.001,
        slippage: SlippageOpt = 0.0005,
        latency: LatencyOpt = 0.0,
        fine: FineOpt = False,
        entry_delay: EntryDelayOpt = 0,
        scores: Annotated[
            Path | None, typer.Option(help="Score file (Parquet); required for `scores`.")
        ] = None,
        threshold: Annotated[
            float | None,
            typer.Option(help=f"Enter where score >= this (default {BASE_CNN.threshold})."),
        ] = None,
        out: OutOpt = None,
        json_output: JsonOpt = False,
        cache_dir: CacheDirOpt = DEFAULT_CACHE_DIR,
        workers: WorkersOpt = None,
    ) -> None:
        run_command(
            strategy,
            Options(
                data_dir, universe, until, start, end, slots, stake, capital, max_positions,
                hold, take_profit, stop, trailing, tp_fill, fee, slippage, latency, fine,
                entry_delay, scores, threshold, out, json_output, cache_dir, workers,
            ),
        )  # fmt: skip

    command.__doc__ = summary
    backtest_app.command(strategy)(command)


_register(
    "f2",
    "Backtest F2, the deep liquid flush: 5 slots, 1 hour hold, pool universe.\n\n"
    "Defaults reproduce the golden F2 case with --until 2026-11-02. Signals are cached in "
    "--cache-dir.",
)
_register(
    "capitulation",
    "Backtest R2, the capitulation long shot: $50 stakes, +20% or 3 days, all coins.\n\n"
    "Signals before 2026-10-01 (--end) as the research. Add `--fine --tp-fill close` for the "
    "1m fill check.",
)
_register(
    "scores",
    "Backtest entries from a score file (--scores): top 2%, 5 slots, 1 hour hold.\n\n"
    "The defaults are the walk-forward base CNN as one continuous book (the research traded one "
    "book per year). Make a score file with `convert-scores`.",
)


def parse_params(items: list[str]) -> dict[str, object]:
    """`key=value` options as a dict; a value is JSON when it parses as JSON, else a string."""
    params: dict[str, object] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            msg = f"invalid param {item!r}: use key=value"
            raise typer.BadParameter(msg, param_hint="--param")
        try:
            params[key.strip()] = json.loads(raw)
        except ValueError:
            params[key.strip()] = raw
    return params


def _plugin_symbols(text: str, root: Path, interval: Interval) -> list[str]:
    if text.lower() in (Universe.POOL.value, Universe.ALL.value):
        try:
            symbols = universe_symbols(Universe(text.lower()), root, interval=interval)
        except FileNotFoundError as exc:
            raise _fail(f"{exc.filename} not found: the {text} universe needs it") from None
        if not symbols:
            raise _fail(f"no {interval.value} candles stored in {root} for the {text} universe")
        return symbols
    symbols = list(dict.fromkeys(s.strip().upper() for s in text.split(",") if s.strip()))
    if not symbols:
        msg = "give pool, all or a comma-separated list of symbols"
        raise typer.BadParameter(msg, param_hint="--symbols")
    return symbols


def _rolling(
    strategy: Strategy,
    symbols: list[str],
    root: Path,
    until: datetime | None,
    start: datetime | None,
    end: datetime | None,
) -> Signals:
    """Run `on_candles` over the stored candles (see `live.plugin.rolling_signals`)."""
    store = KlineStore(root)
    interval = strategy.interval
    stored = set(store.list_series())
    read = [s for s in symbols if (s, interval) in stored]
    if not read:
        raise _fail(f"none of the symbols has {interval.value} candles in {root}")
    first_us = None if start is None else start - strategy.warmup * interval.length
    series = {s: store.read_series(s, interval, end=until) for s in read}
    context = {
        s: store.read_series(s, interval, end=until)
        for s in strategy.context_symbols
        if (s, interval) in stored
    }
    if first_us is not None:
        series = {s: f.filter(pl.col("open_time") >= first_us) for s, f in series.items()}
        context = {s: f.filter(pl.col("open_time") >= first_us) for s, f in context.items()}
    firsts: list[datetime] = [f["open_time"][0] for f in series.values() if f.height]
    lasts: list[datetime] = [f["open_time"][-1] for f in series.values() if f.height]
    if not firsts:
        raise _fail("no candles in the requested range")
    first = start or min(firsts)
    last = end or max(lasts) + interval.length
    if until is not None:
        last = min(last, until)
    ticks = interval.count_open_times(first, last)
    _note(f"signals: running {strategy.name} on {len(read)} symbols over {ticks:,} candles")
    started = time.perf_counter()
    signals = rolling_signals(strategy, series, context, first, last)
    _note(f"signals: {len(signals)} ({time.perf_counter() - started:.1f}s)")
    return signals


def run_plugin(
    source: str, params: dict[str, object], name: str | None, symbols_text: str, options: Options
) -> None:
    """Everything behind `moonshot backtest plugin`."""
    label = name or source.rpartition(":")[2].rpartition(".")[2].lower() or "plugin"
    try:
        strategy = load_strategy(source, name=label, params=params)
    except PluginError as exc:
        raise _fail(str(exc)) from None
    if isinstance(strategy, RuleStrategy):
        strategy.cache_dir = options.cache_dir
        strategy.workers = options.workers
    preset = Preset(Universe.POOL, strategy.exit or ExitPlan(), 5, 5, None, None)
    config, until = _prepare(options, preset, interval=strategy.interval, require_exit=True)
    root = _data_root(options)
    symbols = _plugin_symbols(symbols_text, root, strategy.interval)
    try:
        if isinstance(strategy, BacktestSignalsProvider):
            _note(f"signals: {label} on {len(symbols)} symbols (backtest_signals)")
            signals = strategy.backtest_signals(root, symbols, until)
        else:
            signals = _rolling(
                strategy,
                symbols,
                root,
                until,
                config.start,
                config.end,
            )
    except (ValueError, FileNotFoundError) as exc:
        raise _fail(f"{label}: {exc}") from None
    signals = replace(signals, exit=config.exit, interval=strategy.interval)
    head: dict[str, Any] = {
        "strategy": label,
        "source": source,
        "params": params,
        "symbols": symbols_text if symbols_text in ("pool", "all") else symbols,
    }
    _execute(label, options, config, signals, root, until, head)


@backtest_app.command("plugin")
def plugin_command(
    source: Annotated[
        str,
        typer.Argument(
            metavar="SOURCE", help="Strategy plugin: package.module:attr or path/to/file.py:attr."
        ),
    ],
    param: Annotated[
        list[str] | None,
        typer.Option("--param", help="Plugin parameter key=value (JSON value, else text); repeat."),
    ] = None,
    symbols: Annotated[
        str, typer.Option(help="pool, all or SYM1,SYM2: the symbols to trade.")
    ] = "pool",
    name: Annotated[
        str | None, typer.Option(help="Strategy instance name (default: from SOURCE).")
    ] = None,
    data_dir: DataDirOpt = None,
    until: UntilOpt = None,
    start: StartOpt = None,
    end: EndOpt = None,
    slots: SlotsOpt = None,
    stake: StakeOpt = None,
    capital: CapitalOpt = DEFAULT_CAPITAL,
    max_positions: MaxPositionsOpt = None,
    hold: HoldOpt = None,
    take_profit: TakeProfitOpt = None,
    stop: StopOpt = None,
    trailing: TrailingOpt = None,
    tp_fill: TpFillOpt = None,
    fee: FeeOpt = 0.001,
    slippage: SlippageOpt = 0.0005,
    latency: LatencyOpt = 0.0,
    fine: FineOpt = False,
    entry_delay: EntryDelayOpt = 0,
    out: OutOpt = None,
    json_output: JsonOpt = False,
    cache_dir: CacheDirOpt = DEFAULT_CACHE_DIR,
    workers: WorkersOpt = None,
) -> None:
    """Backtest a strategy plugin: 5 slots unless --slots or --stake, the plugin's exit plan.

    Uses the plugin's `backtest_signals` when it has one, otherwise runs `on_candles` over
    rolling windows of the stored candles (slower, but exactly the live signals). The CLI's
    exit options override the plugin's exit plan.
    """
    options = Options(
        data_dir, None, until, start, end, slots, stake, capital, max_positions,
        hold, take_profit, stop, trailing, tp_fill, fee, slippage, latency, fine,
        entry_delay, None, None, out, json_output, cache_dir, workers,
    )  # fmt: skip
    run_plugin(source, parse_params(param or []), name, symbols, options)


@backtest_app.command("signals")
def signals_command(
    file: Annotated[
        Path,
        typer.Argument(
            metavar="FILE",
            help="Signal file (Parquet): symbol, open_time (UTC), priority, extra columns.",
        ),
    ],
    interval: Annotated[Interval, typer.Option(help="Interval of the signal candles.")] = (
        Interval.M5
    ),
    data_dir: DataDirOpt = None,
    until: UntilOpt = None,
    start: StartOpt = None,
    end: EndOpt = None,
    slots: SlotsOpt = None,
    stake: StakeOpt = None,
    capital: CapitalOpt = DEFAULT_CAPITAL,
    max_positions: MaxPositionsOpt = None,
    hold: HoldOpt = None,
    take_profit: TakeProfitOpt = None,
    stop: StopOpt = None,
    trailing: TrailingOpt = None,
    tp_fill: TpFillOpt = None,
    fee: FeeOpt = 0.001,
    slippage: SlippageOpt = 0.0005,
    latency: LatencyOpt = 0.0,
    fine: FineOpt = False,
    entry_delay: EntryDelayOpt = 0,
    out: OutOpt = None,
    json_output: JsonOpt = False,
    cache_dir: CacheDirOpt = DEFAULT_CACHE_DIR,
    workers: WorkersOpt = None,
) -> None:
    """Backtest a signal file produced anywhere (the columns of `Signals.frame`).

    The exit plan comes from the options (--hold, --take-profit, --stop, --trailing; at least
    one is required); 5 slots unless --slots or --stake.
    """
    if not file.is_file():
        raise typer.BadParameter(f"{file} is not a file", param_hint="FILE")
    options = Options(
        data_dir, None, until, start, end, slots, stake, capital, max_positions,
        hold, take_profit, stop, trailing, tp_fill, fee, slippage, latency, fine,
        entry_delay, None, None, out, json_output, cache_dir, workers,
    )  # fmt: skip
    preset = Preset(None, ExitPlan(), 5, 5, None, None)
    config, until_ = _prepare(options, preset, interval=interval, require_exit=True)
    try:
        signals = Signals(pl.read_parquet(file), config.exit, interval)
    except (ValueError, pl.exceptions.PolarsError, OSError) as exc:
        raise typer.BadParameter(f"{file}: {exc}") from None
    dtype = signals.frame.schema["open_time"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
        msg = f"{file}: open_time must be a timezone-aware datetime, got {dtype}"
        raise typer.BadParameter(msg, param_hint="FILE")
    root = _data_root(options)
    head: dict[str, Any] = {"strategy": file.stem, "signal_file": str(file)}
    _execute("signals", options, config, signals, root, until_, head)


@backtest_app.command("convert-scores")
def convert_scores(
    npz: Annotated[
        Path, typer.Argument(metavar="NPZ", help="Walk-forward .npz (wf/<variant>.npz).")
    ],
    meta: Annotated[Path, typer.Argument(metavar="META", help="Its seq2_meta.parquet.")],
    out: Annotated[Path, typer.Argument(metavar="OUT", help="Score file to write (Parquet).")],
    key: Annotated[str, typer.Option(help="The npz array that becomes the score.")] = "pct",
) -> None:
    """Convert a research walk-forward `.npz` and its metadata into a score file."""
    for path, option in ((npz, "NPZ"), (meta, "META")):
        if not path.is_file():
            raise typer.BadParameter(f"{path} is not a file", param_hint=option)
    try:
        frame = convert_wf_npz(npz, meta, out, key=key)
    except (ValueError, pl.exceptions.PolarsError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from None
    typer.echo(f"wrote {frame.height:,} scores to {out}")
