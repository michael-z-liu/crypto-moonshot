"""The daemon: wires the config, state file, engine, feed and API into one process.

`run_daemon` is `moonshot live run` (paper only): it loads the strategies, opens the state file,
builds the engine and a `LiveFeed` over Binance's public market data (no keys), serves the API on
uvicorn in the same event loop and runs until SIGINT/SIGTERM (or its `stop` event). Shutdown is
graceful: the engine stops after the tick in progress (a tick still collecting is decided again
on the next start), the feed and the API close, then the state file.

`serve_replay` is `moonshot live replay --serve`: a replay over stored candles with the API up,
so external providers (separate processes) can take part. It waits for the providers to connect
before the first tick, and tells them when the replay is over.

The REST client is built for a process that runs for weeks: it never aborts for good after a few
HTTP 429 responses (it keeps waiting out each `Retry-After`), unlike a one-off `data sync`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import socket
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass

import uvicorn
from pydantic import SecretStr

from crypto_moonshot.config import Settings
from crypto_moonshot.data.binance import BinanceClient
from crypto_moonshot.data.stream import StreamConfig
from crypto_moonshot.live.api import TickHub, create_app
from crypto_moonshot.live.config import RunConfig
from crypto_moonshot.live.engine import (
    Clock,
    Engine,
    build_specs,
    feed_subscriptions,
    feed_warmups,
)
from crypto_moonshot.live.feed import KlineFetcher, LiveFeed
from crypto_moonshot.live.replay import ReplayResult, ReplayRun
from crypto_moonshot.live.state import EventLevel, StateStore
from crypto_moonshot.models import Interval

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
RATE_LIMIT_HITS = 1_000_000
"""`max_rate_limit_hits` of the daemon's REST client: effectively never abort for good."""
TOKEN_HELP = (
    "MOONSHOT_API_TOKEN is not set. The API needs a secret token, e.g. "
    "`export MOONSHOT_API_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')`"
)


class DaemonError(Exception):
    """The daemon cannot start or had to stop; the message says why (never a secret).
    `usage` marks a configuration problem (the CLI exits with 2), not a runtime failure."""

    def __init__(self, message: str, *, usage: bool = False) -> None:
        super().__init__(message)
        self.usage = usage


class DaemonInterrupted(Exception):
    """A replay was stopped by a signal before it finished (the state file allows a resume)."""


def require_token(settings: Settings) -> SecretStr:
    """The API token from the settings, or a `DaemonError` that says how to set it."""
    token = settings.api_token
    if token is None or not token.get_secret_value().strip():
        raise DaemonError(TOKEN_HELP, usage=True)
    return token


class _Server(uvicorn.Server):
    """A uvicorn server that leaves signal handling to the daemon (stock uvicorn installs its
    own handlers and re-raises the signal when it has shut down)."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def bind(host: str, port: int) -> socket.socket:
    """A listening socket (so a taken port fails before anything else starts)."""
    try:
        return socket.create_server((host, port), reuse_port=False)
    except OSError as exc:
        msg = f"cannot listen on {host}:{port}: {exc.strerror or exc}"
        raise DaemonError(msg) from None


@contextlib.asynccontextmanager
async def serve_app(app: object, host: str, port: int) -> AsyncIterator[int]:
    """Serve an ASGI `app` with uvicorn in this event loop; yields the bound port. Access logs
    are off (nothing about a request is logged), and leaving the block shuts the server down."""
    sock = bind(host, port)
    config = uvicorn.Config(
        app,  # type: ignore[arg-type]
        log_config=None,
        access_log=False,
        server_header=False,
        lifespan="off",
        timeout_graceful_shutdown=5,
    )
    server = _Server(config)
    task = asyncio.create_task(server.serve(sockets=[sock]), name="api-server")
    try:
        while not server.started:
            if task.done():
                msg = f"the API server on {host}:{port} did not start"
                raise DaemonError(msg)
            await asyncio.sleep(0.01)
        yield sock.getsockname()[1]
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 10.0)
        except (TimeoutError, SystemExit, asyncio.CancelledError):
            server.force_exit = True
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        sock.close()


@dataclass(frozen=True)
class _Signals:
    stop: asyncio.Event
    installed: list[signal.Signals]

    def remove(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in self.installed:
            loop.remove_signal_handler(sig)


def _install_signals(stop: asyncio.Event) -> _Signals:
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError):  # not the main thread
            continue
        installed.append(sig)
    return _Signals(stop, installed)


async def run_daemon(
    config: RunConfig,
    settings: Settings,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    client: KlineFetcher | None = None,
    stream_config: StreamConfig | None = None,
    clock: Clock | None = None,
    stop: asyncio.Event | None = None,
    install_signals: bool = True,
    on_ready: Callable[[int], None] | None = None,
) -> None:
    """Run the live paper daemon until SIGINT/SIGTERM or `stop` is set.

    `client` replaces the Binance REST client and `stream_config` the WebSocket settings (the
    tests point them at fakes); `on_ready` is called with the API's port once it listens.
    Raises `DaemonError` (no token, port taken), `ConfigError` or `PluginError` before anything
    runs, and re-raises whatever stopped the engine after shutting everything down.
    """
    token = require_token(settings)
    config.require_runnable()
    data_root = settings.data_dir if settings.data_dir.is_dir() else None
    specs = build_specs(config, data_root)
    config.state.parent.mkdir(parents=True, exist_ok=True)
    stop = stop if stop is not None else asyncio.Event()
    own_client: BinanceClient | None = None
    if client is None:
        own_client = BinanceClient.from_settings(settings, max_rate_limit_hits=RATE_LIMIT_HITS)
        client = own_client
    store = StateStore.open(config.state)
    handlers = _install_signals(stop) if install_signals else None
    try:
        engine = (
            Engine(config, specs, store, history=client)
            if clock is None
            else Engine(config, specs, store, clock=clock, history=client)
        )
        engine.start()
        feed = LiveFeed(
            feed_subscriptions(specs, store.live_positions()),
            feed_warmups(specs),
            client=client,
            stream_config=stream_config or StreamConfig(base_url=settings.stream_url),
            data_root=data_root,
            tick_timeout=config.engine.tick_timeout.total_seconds(),
            on_event=engine.on_feed_event,
            stale_after=config.account.stale_after,
        )
        app = create_app(engine, feed, token)
        async with serve_app(app, host, port) as bound:
            logger.info("API listening on http://%s:%d/api/v1", host, bound)
            engine.record_event(
                EventLevel.INFO,
                "daemon_started",
                f"daemon started ({config.mode}, {len(specs)} strategies)",
            )
            if on_ready is not None:
                on_ready(bound)
            error = await _run_until_stopped(engine, feed, stop)
            engine.record_event(EventLevel.INFO, "daemon_stopped", "daemon stopped")
        if error is not None:
            raise error
    finally:
        if handlers is not None:
            handlers.remove()
        if own_client is not None:
            await own_client.aclose()
        store.close()


async def _run_until_stopped(
    engine: Engine, feed: LiveFeed, stop: asyncio.Event
) -> Exception | None:
    """Run the engine until it ends or `stop` is set; the exception that ended it, if any."""
    runner = asyncio.create_task(engine.run(feed), name="engine")
    waiter = asyncio.create_task(stop.wait())
    try:
        await asyncio.wait({runner, waiter}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiter.cancel()
    if not runner.done():
        logger.info("stopping")
        await engine.stop()
    results = await asyncio.gather(runner, return_exceptions=True)
    result = results[0]
    if isinstance(result, Exception):
        logger.error("the engine stopped: %s: %s", type(result).__name__, result)
        return result
    if isinstance(result, BaseException):
        raise result
    return None


async def _wait_for_providers(
    run: ReplayRun, hub: TickHub, timeout: float, stop: asyncio.Event
) -> None:
    """Wait until every external strategy has a tick client at its interval."""
    needed: dict[Interval, int] = {}
    for spec in run.specs:
        if spec.external:
            needed[spec.interval] = needed.get(spec.interval, 0) + 1
    if not needed:
        return

    def missing() -> list[str]:
        return [
            f"{n - hub.clients_at(i)} provider(s) for {i.value}"
            for i, n in needed.items()
            if hub.clients_at(i) < n
        ]

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while (waiting := missing()) and not stop.is_set():
        remaining = deadline - loop.time()
        if remaining <= 0:
            msg = f"no provider connected within {timeout:g}s: waiting for {', '.join(waiting)}"
            raise DaemonError(msg)
        hub.changed.clear()
        if not missing():
            break
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(min(remaining, 0.5)):
                await hub.changed.wait()


async def serve_replay(
    run: ReplayRun,
    settings: Settings,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    provider_timeout: float = 60.0,
    stop: asyncio.Event | None = None,
    install_signals: bool = True,
    on_ready: Callable[[int], None] | None = None,
) -> ReplayResult:
    """Run a prepared replay with the API up for external providers.

    Waits (up to `provider_timeout`) for every external strategy's provider to connect, runs the
    replay, tells the providers it is over and shuts the API down. Raises `DaemonInterrupted` if
    a signal (or `stop`) ended it early; the state file then allows a resume.
    """
    token = require_token(settings)
    stop = stop if stop is not None else asyncio.Event()
    handlers = _install_signals(stop) if install_signals else None
    app = create_app(run.engine, run.feed, token)
    hub: TickHub = app.state.hub
    try:
        run.engine.start()
        async with serve_app(app, host, port) as bound:
            logger.info("API listening on http://%s:%d/api/v1", host, bound)
            if on_ready is not None:
                on_ready(bound)
            await _wait_for_providers(run, hub, provider_timeout, stop)
            runner = asyncio.create_task(run.run(), name="replay")
            waiter = asyncio.create_task(stop.wait())
            try:
                await asyncio.wait({runner, waiter}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                waiter.cancel()
            interrupted = not runner.done()
            if interrupted:
                await run.engine.stop()
            try:
                result = await runner
            finally:
                hub.finish("replay finished")
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(5.0):
                        while hub.client_count:
                            hub.changed.clear()
                            await hub.changed.wait()
        if interrupted:
            msg = "the replay was interrupted; run it again to resume from the state file"
            raise DaemonInterrupted(msg)
        return result
    finally:
        if handlers is not None:
            handlers.remove()
