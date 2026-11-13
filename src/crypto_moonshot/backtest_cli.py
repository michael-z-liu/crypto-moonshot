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
import re
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
_HOLD_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([smhdw])", re.IGNORECASE)
_HOLD_UNITS = {
    "s": timedelta(seconds=1),
    "m": timedelta(minutes=1),
    "h": timedelta(hours=1),
    "d": timedelta(days=1),
    "w": timedelta(weeks=1),
}


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
    if text.strip().lower() in ("none", "off"):
        return None
    match = _HOLD_RE.fullmatch(text.strip())
    if match is None:
        msg = f"invalid duration {text!r}: use a number and a unit, for example 1h, 90m, 3d or 2w"
        raise typer.BadParameter(msg, param_hint="--hold")
    hold = float(match[1]) * _HOLD_UNITS[match[2].lower()]
    if hold <= timedelta(0):
        msg = f"invalid duration {text!r}: must be positive (or 'none')"
        raise typer.BadParameter(msg, param_hint="--hold")
    return hold


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
            use_fine=options.fine,
            entry_delay=options.entry_delay,
        )
    )
    root = resolve_settings(options.data_dir).data_dir
    if not KlineStore(root).list_series():
        raise _fail(f"no candles stored in {root}; download some with `moonshot data sync`")
    universe = options.universe or preset.universe
    signals = _signals(
        strategy,
        universe,
        root,
        until,
        options.scores,
        options.threshold,
        exit_plan,
        options.cache_dir,
        options.workers,
    )
    source = ParquetCandleSource(root, until=until)
    _note(f"backtest: {len(signals)} signals")
    result = run_backtest(signals.frame, source, config)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = options.out or Path("results") / f"{strategy}-{stamp}"
    document: dict[str, Any] = {
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


def _register(strategy: str, summary: str) -> None:
    def command(
        data_dir: Annotated[
            Path | None,
            typer.Option(help="Data folder (default: $MOONSHOT_DATA_DIR, else ./data)."),
        ] = None,
        universe: Annotated[
            Universe | None,
            typer.Option(
                help="Symbols to trade: pool (top 200 minus stablecoins and tokenised stocks) or "
                "all (plus delisted and outside the pool). Default: the strategy's own."
            ),
        ] = None,
        until: Annotated[
            str | None,
            typer.Option(help="Data end, exclusive (YYYY-MM-DD or ISO datetime): ignore later."),
        ] = None,
        start: Annotated[
            str | None, typer.Option(help="Take signals from this open time (inclusive).")
        ] = None,
        end: Annotated[
            str | None, typer.Option(help="Take signals before this open time (exclusive).")
        ] = None,
        slots: Annotated[
            int | None,
            typer.Option(min=1, help="Equal-slot compounding: stake = equity / SLOTS."),
        ] = None,
        stake: Annotated[
            float | None, typer.Option(min=0.0, help="Fixed stake per entry in USDT.")
        ] = None,
        capital: Annotated[
            float, typer.Option(min=0.0, help="Starting capital in USDT.")
        ] = DEFAULT_CAPITAL,
        max_positions: Annotated[
            int | None,
            typer.Option(min=1, help="Most positions open at once (default: the slots, or 200)."),
        ] = None,
        hold: Annotated[
            str | None, typer.Option(help="Time exit, e.g. 1h, 90m, 3d, 2w; 'none' for no limit.")
        ] = None,
        take_profit: Annotated[
            float | None, typer.Option(min=0.0, help="Take-profit as a fraction (0.2 = +20%).")
        ] = None,
        stop: Annotated[
            float | None, typer.Option(min=0.0, max=0.999, help="Stop as a fraction (0 = off).")
        ] = None,
        trailing: Annotated[
            float | None,
            typer.Option(min=0.0, max=0.999, help="Trailing stop as a fraction (0 = off)."),
        ] = None,
        tp_fill: Annotated[
            TpFill | None,
            typer.Option(help="Take profit fills when a candle touches the target, or closes."),
        ] = None,
        fee: Annotated[float, typer.Option(min=0.0, max=0.5, help="Fee per side.")] = 0.001,
        slippage: Annotated[
            float, typer.Option(min=0.0, max=0.5, help="Slippage per side.")
        ] = 0.0005,
        latency: Annotated[
            float, typer.Option(min=0.0, max=0.5, help="Latency cost on market orders.")
        ] = 0.0,
        fine: Annotated[
            bool, typer.Option("--fine/--no-fine", help="Fill exits on 1m candles when stored.")
        ] = False,
        entry_delay: Annotated[
            int, typer.Option(min=0, help="Extra candles between signal and entry.")
        ] = 0,
        scores: Annotated[
            Path | None, typer.Option(help="Score file (Parquet); required for `scores`.")
        ] = None,
        threshold: Annotated[
            float | None,
            typer.Option(help=f"Enter where score >= this (default {BASE_CNN.threshold})."),
        ] = None,
        out: Annotated[
            Path | None,
            typer.Option(help="Output folder (default: results/<strategy>-<UTC timestamp>/)."),
        ] = None,
        json_output: Annotated[
            bool, typer.Option("--json", help="Print the results as one JSON object on stdout.")
        ] = False,
        cache_dir: Annotated[
            Path, typer.Option(help="Where signal caches go (never the data folder).")
        ] = DEFAULT_CACHE_DIR,
        workers: Annotated[
            int | None, typer.Option(min=1, help="Processes for signal computation.")
        ] = None,
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
