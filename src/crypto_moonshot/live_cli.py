"""`moonshot live` and `moonshot provider`: run, replay and control the paper trading daemon.

Commands:

- `live run --config FILE`: the daemon (paper only; `MOONSHOT_API_TOKEN` is required).
- `live replay --config FILE --start --end`: the engine over stored candles; prints the metrics
  like `moonshot backtest`; `--serve` keeps the API up so external providers can trade it.
- `live status`, `live pause`, `live resume`, `live flatten`: talk to a running daemon through
  its API (`status --config` falls back to reading the state file when the daemon is down).
- `live backup --config FILE [--out DIR]`: an online backup of the state file (daemon running or
  not); `live restore BACKUP --config FILE`: replace the state file with a backup (daemon stopped).
- `provider run SOURCE --name NAME --url URL`: run a strategy plugin as an external provider.

Exit codes: 0 ok, 1 runtime failure (including a state file in use by another process), 2 usage
or configuration error (no token, bad config, a plugin that does not load), 3 the daemon's
watchdog stopped it (a stalled engine or a blocked event loop; a restart policy starts it
again), 130 interrupted. Errors go to stderr; results to stdout.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import typer

from crypto_moonshot.backtest.results import json_safe
from crypto_moonshot.backtest_cli import parse_params
from crypto_moonshot.cli_output import (
    format_time,
    print_backtest_summary,
    print_notice,
    print_table,
    stdout_is_tty,
)
from crypto_moonshot.config import Settings
from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.data_cli import parse_when, resolve_settings
from crypto_moonshot.live.backup import BackupError, make_backup, restore_backup
from crypto_moonshot.live.config import ConfigError, RunConfig, load_config
from crypto_moonshot.live.daemon import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DaemonError,
    DaemonInterrupted,
    install_stop_signals,
    require_token,
    run_daemon,
    serve_replay,
)
from crypto_moonshot.live.engine import BookStatus, to_json
from crypto_moonshot.live.plugin import PluginError, load_strategy
from crypto_moonshot.live.replay import ReplayResult, prepare_replay
from crypto_moonshot.live.state import StateError, StateStore
from crypto_moonshot.logs import LogFormat, LogLevel, configure_logging
from crypto_moonshot.provider import ProviderError, run_provider

live_app = typer.Typer(
    no_args_is_help=True,
    help="Run, replay and control the paper trading daemon (paper mode only).",
)
provider_app = typer.Typer(
    no_args_is_help=True, help="Run a strategy plugin as an external provider."
)

DEFAULT_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130
_TIMEOUT_S = 15.0

ConfigOpt = Annotated[
    Path, typer.Option("--config", help="The run config (TOML), see docs/architecture.md.")
]
UrlOpt = Annotated[
    str,
    typer.Option(
        "--url",
        envvar="MOONSHOT_URL",
        help="The daemon's base URL (token from MOONSHOT_API_TOKEN).",
    ),
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON on stdout.")]
DataDirOpt = Annotated[
    Path | None, typer.Option(help="Data folder (default: $MOONSHOT_DATA_DIR, else ./data).")
]
HostOpt = Annotated[str, typer.Option(help="Address the API listens on.")]
PortOpt = Annotated[int, typer.Option(min=0, max=65535, help="Port the API listens on.")]
LogFormatOpt = Annotated[
    LogFormat,
    typer.Option(
        help="text: lines on stderr; json: one JSON object per line on stdout (for Docker)."
    ),
]
LogLevelOpt = Annotated[LogLevel | None, typer.Option(help="Log level (default: info for run).")]


class SignalsMode(StrEnum):
    ROLLING = "rolling"
    BACKTEST = "backtest"


def _fail(text: str, code: int = EXIT_FAILURE) -> typer.Exit:
    typer.echo(f"error: {text}", err=True)
    return typer.Exit(code)


def _log_to_stderr(level: int = logging.INFO) -> None:
    configure_logging(LogFormat.TEXT, level)


def _logging(fmt: LogFormat, level: LogLevel | None, default: LogLevel, secrets: str = "") -> None:
    """Set up logging for a long-running command; `secrets` are masked in every line."""
    configure_logging(fmt, level or default, secrets=[secrets])


def _config(path: Path) -> RunConfig:
    try:
        return load_config(path)
    except ConfigError as exc:
        raise _fail(str(exc), EXIT_USAGE) from None


def _token(settings: Settings | None = None) -> str:
    try:
        return require_token(settings or Settings()).get_secret_value()
    except DaemonError as exc:
        raise _fail(str(exc), EXIT_USAGE) from None


# -- run ----------------------------------------------------------------------------------------


@live_app.command("run")
def run_command(
    config: ConfigOpt,
    host: HostOpt = DEFAULT_HOST,
    port: PortOpt = DEFAULT_PORT,
    data_dir: DataDirOpt = None,
    log_format: LogFormatOpt = LogFormat.TEXT,
    log_level: LogLevelOpt = None,
) -> None:
    """Run the paper trading daemon until SIGINT or SIGTERM.

    Follows live Binance candles (public market data, no keys), runs the config's strategies
    (in-process plugins, and external providers that connect to the API), keeps its state in the
    config's `state` file and serves the API on HOST:PORT. Needs MOONSHOT_API_TOKEN. Only one
    daemon (or replay) can use a state file at a time. Exits with 3 when its watchdog stops it.
    """
    run = _config(config)
    settings = resolve_settings(data_dir)
    token = _token(settings)
    try:
        run.require_runnable()
    except ConfigError as exc:
        raise _fail(str(exc), EXIT_USAGE) from None
    _logging(log_format, log_level, LogLevel.INFO, token)
    try:
        asyncio.run(run_daemon(run, settings, host=host, port=port))
    except (ConfigError, PluginError) as exc:
        raise _fail(str(exc), EXIT_USAGE) from None
    except DaemonError as exc:
        raise _fail(str(exc), exc.code or (EXIT_USAGE if exc.usage else EXIT_FAILURE)) from None
    except StateError as exc:
        raise _fail(str(exc)) from None
    except KeyboardInterrupt:
        raise typer.Exit(EXIT_INTERRUPTED) from None
    except Exception as exc:
        raise _fail(f"the daemon stopped: {type(exc).__name__}: {exc}") from None


# -- replay -------------------------------------------------------------------------------------


def _print_replay(result: ReplayResult, out: Path) -> None:
    typer.echo(f"results written to {out}", err=True)
    outcomes = {k: v for k, v in result.document["signals"].items() if k != "placed"}
    print_backtest_summary(result.metrics, outcomes, result.yearly, tty=stdout_is_tty())
    if result.open_positions:
        print_notice(
            f"{result.open_positions} position(s) still open at the end (not in the trade list)",
            tty=stdout_is_tty(),
            style="",
        )


@live_app.command("replay")
def replay_command(
    config: ConfigOpt,
    start: Annotated[str, typer.Option(help="First candle open time (YYYY-MM-DD or ISO).")],
    end: Annotated[str, typer.Option(help="End, exclusive (YYYY-MM-DD or ISO datetime).")],
    data_dir: DataDirOpt = None,
    out: Annotated[
        Path | None, typer.Option(help="Output folder (default: results/replay-<UTC timestamp>/).")
    ] = None,
    state: Annotated[
        Path | None,
        typer.Option(
            help="State file for this replay (default: the config's `state`). An existing file "
            "with processed ticks resumes after them."
        ),
    ] = None,
    signals: Annotated[
        SignalsMode,
        typer.Option(
            help="rolling: call on_candles on every candle (the live signals); backtest: use "
            "the plugin's backtest_signals when it has one (faster)."
        ),
    ] = SignalsMode.ROLLING,
    serve: Annotated[
        bool, typer.Option("--serve", help="Serve the API so external providers can trade it.")
    ] = False,
    host: HostOpt = DEFAULT_HOST,
    port: PortOpt = DEFAULT_PORT,
    provider_timeout: Annotated[
        float, typer.Option(min=0.0, help="With --serve: seconds to wait for the providers.")
    ] = 60.0,
    json_output: JsonOpt = False,
    log_format: LogFormatOpt = LogFormat.TEXT,
    log_level: LogLevelOpt = None,
) -> None:
    """Replay the engine over stored candles and print the metrics.

    Runs the same engine as `live run`, on stored candles with a simulated clock; its trades
    match the backtest's on the same signals. External strategies need `--serve` and a
    provider (`moonshot provider run`); the replay waits for them before the first candle.
    """
    run = _config(config)
    if state is not None:
        run = run.model_copy(update={"state": state})
    first, last = parse_when(start, option="--start"), parse_when(end, option="--end")
    if last <= first:
        raise typer.BadParameter("--end must be after --start", param_hint="--end")
    settings = resolve_settings(data_dir)
    root = settings.data_dir
    if not KlineStore(root).list_series():
        raise _fail(f"no candles stored in {root}; download some with `moonshot data sync`")
    token = _token(settings) if serve else ""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out or Path("results") / f"replay-{stamp}"
    _logging(log_format, log_level, LogLevel.WARNING, token)
    try:
        prepared = prepare_replay(
            run,
            first,
            last,
            root,
            out_dir,
            signals="backtest" if signals is SignalsMode.BACKTEST else "on_candles",
        )
    except (ConfigError, PluginError) as exc:
        raise _fail(str(exc), EXIT_USAGE) from None
    except StateError as exc:
        raise _fail(str(exc)) from None
    try:
        typer.echo(f"replay: {prepared.feed.tick_count:,} candles to process", err=True)
        if serve:
            result = asyncio.run(
                serve_replay(
                    prepared, settings, host=host, port=port, provider_timeout=provider_timeout
                )
            )
        else:
            result = asyncio.run(prepared.run())
    except (ConfigError, PluginError) as exc:
        raise _fail(str(exc), EXIT_USAGE) from None
    except DaemonInterrupted as exc:
        raise _fail(str(exc), EXIT_INTERRUPTED) from None
    except DaemonError as exc:
        raise _fail(str(exc), EXIT_USAGE if exc.usage else EXIT_FAILURE) from None
    except KeyboardInterrupt:
        raise _fail("interrupted", EXIT_INTERRUPTED) from None
    except Exception as exc:
        raise _fail(f"the replay failed: {type(exc).__name__}: {exc}") from None
    finally:
        prepared.close()
    if json_output:
        typer.echo(json.dumps(json_safe({**result.document, "out": str(out_dir)})))
    else:
        _print_replay(result, out_dir)


# -- backups ------------------------------------------------------------------------------------


@live_app.command("backup")
def backup_command(
    config: ConfigOpt,
    out: Annotated[
        Path | None,
        typer.Option(help="Folder for the backup (default: backups/ next to the state file)."),
    ] = None,
    json_output: JsonOpt = False,
) -> None:
    """Back up the config's state file (safe while the daemon runs).

    Uses SQLite's online backup, so the copy is one consistent snapshot, and checks it
    (`PRAGMA integrity_check`) before it gets its name `<state>-<UTC timestamp>.sqlite`. The
    daemon also backs up by itself at start and every `health.backup_every`, keeping the newest
    `health.backup_keep`; this command never deletes anything.
    """
    run = _config(config)
    try:
        result = make_backup(run.state, out)
    except StateError as exc:
        raise _fail(str(exc)) from None
    except OSError as exc:
        raise _fail(f"cannot write the backup: {exc.strerror or exc}") from None
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "path": str(result.path),
                    "size": result.size,
                    "schema_version": result.schema_version,
                    "created_at": result.created_at.isoformat(),
                }
            )
        )
    else:
        typer.echo(f"backup written to {result.path} ({result.size:,} bytes, verified)")


@live_app.command("restore")
def restore_command(
    backup: Annotated[
        Path, typer.Argument(metavar="BACKUP", help="A backup made by `moonshot live backup`.")
    ],
    config: ConfigOpt,
) -> None:
    """Replace the config's state file with BACKUP (the daemon must be stopped).

    Checks the backup first (integrity and schema version) and refuses while a daemon or replay
    uses the state file. The state file it replaces is kept next to it as
    `<state>.before-restore-<UTC timestamp>`.
    """
    run = _config(config)
    try:
        result = restore_backup(backup, run.state)
    except (BackupError, StateError) as exc:
        raise _fail(str(exc)) from None
    except OSError as exc:
        raise _fail(f"cannot restore: {exc.strerror or exc}") from None
    typer.echo(f"restored {result.state} from {result.restored_from}")
    if result.replaced is not None:
        typer.echo(f"the previous state file is kept as {result.replaced}")


# -- talking to a daemon -------------------------------------------------------------------------


def _api(url: str, token: str) -> httpx.Client:
    return httpx.Client(
        base_url=url.rstrip("/") + "/api/v1",
        headers={"Authorization": f"Bearer {token}"},
        timeout=_TIMEOUT_S,
    )


class _ApiError(Exception):
    """An API call failed; `unreachable` means no connection (as opposed to an HTTP error)."""

    def __init__(self, message: str, *, unreachable: bool = False) -> None:
        super().__init__(message)
        self.unreachable = unreachable


def _call(client: httpx.Client, method: str, path: str, **kwargs: Any) -> Any:
    """One API call; failures raise `_ApiError` with a message that never holds the token."""
    try:
        response = client.request(method, path, **kwargs)
    except httpx.TransportError as exc:
        msg = f"cannot reach the daemon at {client.base_url}: {type(exc).__name__}"
        raise _ApiError(msg, unreachable=True) from None
    if response.status_code == 401:
        msg = "the daemon refused the API token (401); check MOONSHOT_API_TOKEN"
        raise _ApiError(msg)
    if response.status_code >= 400:
        try:
            reason = response.json().get("reason", response.text)
        except ValueError:
            reason = response.text[:200]
        msg = f"{method} {path}: HTTP {response.status_code}: {reason}"
        raise _ApiError(msg)
    return response.json()


def _event_json(event: object) -> dict[str, Any]:
    out = to_json(event)
    assert isinstance(out, dict)
    return out


def status_from_state(store: StateStore, config: RunConfig, now: datetime) -> dict[str, Any]:
    """The `GET /status` document rebuilt from a state file (no feed: `running` is false)."""
    books: dict[str, BookStatus] = {}
    for record in store.strategies():
        book = store.derive_book(record.name, record.capital, now)
        books[record.name] = BookStatus(
            strategy=record.name,
            capital=record.capital,
            cash=book.cash,
            equity=book.equity,
            open_cost=book.open_cost,
            open_positions=book.open_positions,
            day_pnl=book.day_pnl,
            day_entries=book.day_entries,
        )
    document = {
        "mode": config.mode,
        "running": False,
        "controls": store.controls(),
        "books": books,
        "positions": store.live_positions(),
        "open_orders": store.pending_orders(),
        "stale_symbols": {},
        "last_ticks": {i.value: t for i, t in store.last_ticks().items()},
        "providers": store.providers(),
        "health": {"source": "state_file", "path": str(store.path)},
    }
    out = to_json(document)
    assert isinstance(out, dict)
    return out


def _time(text: object) -> str:
    return "-" if not text else format_time(datetime.fromisoformat(str(text)))


def _num(value: object, spec: str = ",.2f") -> str:
    return "-" if value is None else format(float(str(value)), spec)


def print_status(status: dict[str, Any], events: list[dict[str, Any]], *, source: str) -> None:
    """The status document as tables (see `GET /status`)."""
    tty = stdout_is_tty()
    controls = status["controls"]
    print_notice(
        f"{status['mode']} ({source}): "
        f"{'running' if status['running'] else 'not running'}, "
        f"entries {'PAUSED' if controls['paused'] else 'on'}"
        + (", flatten requested" if controls["flatten_requested"] else ""),
        tty=tty,
        style="bold",
    )
    ticks = ", ".join(f"{i} {_time(t)}" for i, t in status["last_ticks"].items())
    print_notice(f"last candle processed: {ticks or 'none'}", tty=tty, style="")
    health = status["health"]
    if health.get("error"):
        print_notice(f"engine error: {health['error']}", tty=tty)
    stale = {i: s for i, s in status["stale_symbols"].items() if s}
    for interval, symbols in stale.items():
        print_notice(f"stale at {interval}: {', '.join(symbols[:10])}", tty=tty)
    print_table(
        ["STRATEGY", "CAPITAL", "CASH", "EQUITY", "OPEN COST", "OPEN", "DAY P&L"],
        [
            [
                b["strategy"],
                _num(b["capital"], ",.0f"),
                _num(b["cash"]),
                _num(b["equity"]),
                _num(b["open_cost"]),
                str(b["open_positions"]),
                _num(b["day_pnl"], "+,.2f"),
            ]
            for b in status["books"].values()
        ],
        tty=tty,
        right={1, 2, 3, 4, 5, 6},
    )
    print_table(
        ["STRATEGY", "SYMBOL", "STATUS", "ENTRY", "PRICE", "COST"],
        [
            [
                p["strategy"],
                p["symbol"],
                p["status"],
                _time(p["entry_time"] or p["placed_at"]),
                _num(p["entry_price"], ",.6g"),
                _num(p["cost"]),
            ]
            for p in status["positions"]
        ],
        tty=tty,
        right={4, 5},
    )
    if status["providers"]:
        print_table(
            ["PROVIDER", "LAST SEEN", "LAST BATCH"],
            [
                [p["name"], _time(p["last_seen"]), _time(p["last_batch_time"])]
                for p in status["providers"]
            ],
            tty=tty,
        )
    print_table(
        ["TIME", "LEVEL", "EVENT", "MESSAGE"],
        [[_time(e["time"]), e["level"], e["kind"], str(e["message"])[:100]] for e in events],
        tty=tty,
    )


@live_app.command("status")
def status_command(
    url: UrlOpt = DEFAULT_URL,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="The run config: when the daemon is not reachable, read its state file "
            "(read-only) instead.",
        ),
    ] = None,
    json_output: JsonOpt = False,
    events: Annotated[int, typer.Option(min=1, max=1000, help="Recent events to show.")] = 10,
) -> None:
    """Show the daemon's mode, controls, books, positions, providers and recent events.

    Asks the running daemon (--url, token from MOONSHOT_API_TOKEN). With --config and no daemon
    reachable, reads the config's state file instead (read-only; safe while the daemon runs).
    """
    run = _config(config) if config is not None else None
    token = Settings().api_token
    document: dict[str, Any] | None = None
    recent: list[dict[str, Any]] = []
    source = "daemon"
    if token is not None and token.get_secret_value():
        try:
            with _api(url, token.get_secret_value()) as client:
                document = _call(client, "GET", "/status")
                recent = _call(client, "GET", "/events", params={"limit": events})["events"]
        except _ApiError as exc:
            if run is None or not exc.unreachable:
                raise _fail(str(exc)) from None
            typer.echo("note: the daemon is not reachable; reading the state file", err=True)
    elif run is None:
        raise _fail(
            "MOONSHOT_API_TOKEN is not set (needed to ask the daemon); or pass --config to "
            "read the state file",
            EXIT_USAGE,
        )
    else:
        typer.echo("note: no MOONSHOT_API_TOKEN; reading the state file", err=True)
    if document is None:
        assert run is not None
        try:
            with StateStore.open_readonly(run.state) as store:
                document = status_from_state(store, run, datetime.now(UTC))
                recent = [_event_json(e) for e in store.recent_events(events)]
        except StateError as exc:
            raise _fail(str(exc)) from None
        source = "state file"
    if json_output:
        typer.echo(json.dumps(json_safe({**document, "events": recent, "source": source})))
    else:
        print_status(document, recent, source=source)


def _control(action: Literal["pause", "resume", "flatten"], url: str) -> None:
    token = _token()
    try:
        with _api(url, token) as client:
            result = _call(client, "POST", f"/control/{action}")
    except _ApiError as exc:
        raise _fail(str(exc)) from None
    controls = result["controls"]
    typer.echo(
        f"{action}: ok (entries {'paused' if controls['paused'] else 'on'}"
        + (", flatten requested" if controls["flatten_requested"] else "")
        + ")"
    )


@live_app.command("pause")
def pause_command(url: UrlOpt = DEFAULT_URL) -> None:
    """Stop taking new entries (exits keep working)."""
    _control("pause", url)


@live_app.command("resume")
def resume_command(url: UrlOpt = DEFAULT_URL) -> None:
    """Take entries again."""
    _control("resume", url)


@live_app.command("flatten")
def flatten_command(
    url: UrlOpt = DEFAULT_URL,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
) -> None:
    """At the next candle: sell every open position at market, cancel pending entries, pause."""
    if not yes and not typer.confirm(
        "Sell every open position at the next candle, cancel pending entries and pause?"
    ):
        raise _fail("aborted")
    _control("flatten", url)


# -- provider -----------------------------------------------------------------------------------


@provider_app.command("run")
def provider_run_command(
    source: Annotated[
        str,
        typer.Argument(
            metavar="SOURCE", help="Strategy plugin: package.module:attr or path/to/file.py:attr."
        ),
    ],
    name: Annotated[
        str,
        typer.Option(help="The strategy's name in the daemon's run config (source = external)."),
    ],
    url: UrlOpt = DEFAULT_URL,
    param: Annotated[
        list[str] | None,
        typer.Option("--param", help="Plugin parameter key=value (JSON value, else text); repeat."),
    ] = None,
    keep_going: Annotated[
        bool,
        typer.Option(
            "--keep-going", help="If on_candles fails, log it and miss that candle (default: stop)."
        ),
    ] = False,
) -> None:
    """Run a plugin as the external provider of a strategy of the daemon.

    The same plugin file that runs inside the daemon (`source = "path/file.py:attr"`) runs here
    unchanged; the daemon's config lists the strategy with `source = "external"`. Needs
    MOONSHOT_API_TOKEN. Reconnects when the daemon goes away; ends when a replay finishes.
    """
    token = _token()
    try:
        strategy = load_strategy(source, name=name, params=parse_params(param or []))
    except PluginError as exc:
        raise _fail(str(exc), EXIT_USAGE) from None
    _log_to_stderr()

    async def main() -> bool:
        stop = asyncio.Event()
        remove_signals = install_stop_signals(stop)
        try:
            await run_provider(
                strategy, url=url, token=token, name=name, keep_going=keep_going, stop=stop
            )
        finally:
            remove_signals()
        return stop.is_set()

    try:
        interrupted = asyncio.run(main())
    except ProviderError as exc:
        raise _fail(str(exc)) from None
    except KeyboardInterrupt:
        raise typer.Exit(EXIT_INTERRUPTED) from None
    if interrupted:
        raise typer.Exit(EXIT_INTERRUPTED)
