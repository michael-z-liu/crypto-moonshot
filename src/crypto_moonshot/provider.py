"""The provider SDK: run a strategy plugin as an external signal provider.

`run_provider(strategy, url=..., token=..., name=...)` makes a `Strategy` (the same class that
runs in the daemon's process) work from another process, or another machine: it reads the
strategy's settings from the daemon (`GET /strategies/{name}`), fetches warm-up windows
(`GET /candles`), follows the tick stream (`WS /ticks`), keeps its own windows, calls
`Strategy.on_candles` with exactly the inputs the engine would give an in-process plugin
(`Engine._plugin_inputs`: traded windows, minus symbols whose latest candle is more than
`stale_after` intervals old, and the context windows) and posts one batch per tick
(`POST /signals`), even an empty one. The daemon must list the strategy with
`source = "external"` in its run config.

Behaviour worth knowing:

- `on_candles` runs in a worker thread, so the connection's pings and heartbeats keep working
  while a model computes.
- A dropped connection (or a daemon that is not up yet) is retried with backoff; every
  reconnect fetches the windows again, and a tick the engine is still waiting for is answered
  from them. A wrong token, an unknown strategy or an interval or contract mismatch is fatal.
- A plugin that raises, or returns a frame that breaks the signal contract, stops the provider
  with `ProviderError` unless `keep_going` is set (then that tick gets no batch and the daemon
  records it as missed). Batches the daemon refuses are logged as errors and counted.
- Known difference from in-process: candles that arrive late for an earlier tick reach the
  daemon's windows but not this provider's, until it reconnects.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
import polars as pl
from pydantic import SecretStr
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from crypto_moonshot.live.api import API_PREFIX
from crypto_moonshot.live.contract import CONTRACT_VERSION, SignalBatch, frame_to_batch
from crypto_moonshot.live.feed import HISTORY_SLACK
from crypto_moonshot.live.plugin import Strategy
from crypto_moonshot.models import KLINE_SCHEMA, Interval, as_utc

logger = logging.getLogger(__name__)

_EMPTY_WINDOW = pl.DataFrame(schema=KLINE_SCHEMA)
BACKOFF_START_S = 1.0
BACKOFF_MAX_S = 30.0
POST_ATTEMPTS = 3


class ProviderError(Exception):
    """The provider cannot continue (fatal setup problem, or a failing plugin). The message
    never contains the token."""


@dataclass
class ProviderStats:
    """What a provider has done so far."""

    ticks: int = 0
    posted: int = 0
    late: int = 0
    duplicates: int = 0
    refused: list[str] = field(default_factory=list)
    reconnects: int = 0


@dataclass(frozen=True)
class StrategyInfo:
    """What `GET /strategies/{name}` says."""

    name: str
    interval: Interval
    symbols: tuple[str, ...]
    warmup: int
    context_symbols: tuple[str, ...]
    stale_after: int
    contract: int
    external: bool


def _ws_url(url: str) -> str:
    base = url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base.removeprefix("https://")
    if base.startswith("http://"):
        return "ws://" + base.removeprefix("http://")
    msg = f"the daemon URL must start with http:// or https://, got {url!r}"
    raise ProviderError(msg)


def _parse_time(text: str) -> datetime:
    return as_utc(datetime.fromisoformat(text))


class Provider:
    """One strategy's provider loop; `run_provider` is the usual entry point.

    Args:
        strategy: the plugin; its `interval` must match the daemon's, its `context_symbols`
            must be among the ones the daemon subscribes for `name`.
        url: the daemon's base URL (`http://host:8765`).
        token: the API token.
        name: the strategy's instance name in the daemon's run config.
        keep_going: log a failing `on_candles` and carry on instead of stopping.
        stop: set it to stop the provider cleanly.
        backoff_start / backoff_max: reconnect delays in seconds.
    """

    def __init__(
        self,
        strategy: Strategy,
        *,
        url: str,
        token: str | SecretStr,
        name: str,
        keep_going: bool = False,
        stop: asyncio.Event | None = None,
        backoff_start: float = BACKOFF_START_S,
        backoff_max: float = BACKOFF_MAX_S,
    ) -> None:
        self.strategy = strategy
        self.name = name
        self.keep_going = keep_going
        self.stop = stop if stop is not None else asyncio.Event()
        self.stats = ProviderStats()
        self._base = url.rstrip("/")
        self._ws_base = _ws_url(url)
        secret = token.get_secret_value() if isinstance(token, SecretStr) else token
        if not secret:
            msg = "the API token is empty"
            raise ProviderError(msg)
        self._headers = {"Authorization": f"Bearer {secret}"}
        self._backoff = (backoff_start, backoff_max)
        self._http: httpx.AsyncClient | None = None
        self._info: StrategyInfo | None = None
        self._windows: dict[str, pl.DataFrame] = {}
        self._handled: datetime | None = None
        self._finished = False

    # -- running ------------------------------------------------------------------------------

    async def run(self) -> None:
        """Serve ticks until `stop` is set or the daemon ends the run (a finished replay)."""
        delay = self._backoff[0]
        async with httpx.AsyncClient(
            base_url=self._base + API_PREFIX, headers=self._headers, timeout=30.0
        ) as http:
            self._http = http
            while not self.stop.is_set() and not self._finished:
                try:
                    progressed = await self._session()
                except (httpx.TransportError, OSError, WebSocketException) as exc:
                    progressed = False
                    logger.warning(
                        "provider %s: connection to the daemon failed (%s); retrying in %.0fs",
                        self.name,
                        type(exc).__name__,
                        delay,
                    )
                else:
                    if not (self.stop.is_set() or self._finished):
                        logger.warning(
                            "provider %s: connection closed; reconnecting in %.0fs",
                            self.name,
                            delay,
                        )
                if self.stop.is_set() or self._finished:
                    break
                self.stats.reconnects += 1
                await self._sleep(delay)
                delay = self._backoff[0] if progressed else min(delay * 2, self._backoff[1])

    async def _sleep(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                await self.stop.wait()

    async def _session(self) -> bool:
        """One connection: setup, windows, ticks. True when it got as far as handling a tick."""
        info = await self._fetch_info()
        self._info = info
        uri = f"{self._ws_base}{API_PREFIX}/ticks?interval={info.interval.value}"
        try:
            async with connect(
                uri, additional_headers=self._headers, max_size=64 * 1024 * 1024
            ) as ws:
                hello = json.loads(await ws.recv())
                await self._load_windows(info)
                collecting = hello.get("collecting")
                if collecting is not None:
                    await self._answer_collecting(info, _parse_time(collecting))
                progressed = False
                async for raw in ws:
                    message = json.loads(raw)
                    kind = message.get("type")
                    if kind == "tick":
                        progressed = True
                        await self._on_tick(info, message)
                    elif kind == "end":
                        logger.info(
                            "provider %s: the run is over (%s)", self.name, message["reason"]
                        )
                        self._finished = True
                        return progressed
                    if self.stop.is_set():
                        return progressed
                return progressed
        except InvalidStatus as exc:
            code = exc.response.status_code
            if code == 401:
                msg = "the daemon refused the API token (401); check MOONSHOT_API_TOKEN"
                raise ProviderError(msg) from None
            raise
        except ConnectionClosed:
            return False

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        assert self._http is not None
        response = await self._http.request(method, path, **kwargs)
        if response.status_code == 401:
            msg = "the daemon refused the API token (401); check MOONSHOT_API_TOKEN"
            raise ProviderError(msg)
        return response

    @staticmethod
    def _ok(response: httpx.Response) -> None:
        if response.status_code >= 400:
            msg = (
                f"{response.request.method} {response.request.url.path}: "
                f"HTTP {response.status_code}: {_reason(response)}"
            )
            raise ProviderError(msg)

    async def _fetch_info(self) -> StrategyInfo:
        response = await self._request("GET", f"/strategies/{self.name}")
        if response.status_code == 404:
            msg = f"the daemon has no strategy named {self.name!r} (check the run config)"
            raise ProviderError(msg)
        self._ok(response)
        body = response.json()
        info = StrategyInfo(
            name=body["name"],
            interval=Interval(body["interval"]),
            symbols=tuple(body["symbols"]),
            warmup=int(body["warmup"]),
            context_symbols=tuple(body["context_symbols"]),
            stale_after=int(body["stale_after"]),
            contract=int(body["contract"]),
            external=bool(body["external"]),
        )
        self._check(info)
        return info

    def _check(self, info: StrategyInfo) -> None:
        strategy = self.strategy
        if not info.external:
            msg = (
                f"strategy {info.name!r} runs inside the daemon; give it "
                'source = "external" in the run config to provide it from here'
            )
            raise ProviderError(msg)
        if info.contract != CONTRACT_VERSION:
            msg = (
                f"the daemon speaks signal contract {info.contract}, this build {CONTRACT_VERSION}"
            )
            raise ProviderError(msg)
        if info.interval is not strategy.interval:
            msg = (
                f"strategy {info.name!r} trades {info.interval.value} in the daemon's config, "
                f"but the plugin works on {strategy.interval.value}"
            )
            raise ProviderError(msg)
        missing = [s for s in strategy.context_symbols if s not in info.context_symbols]
        if missing:
            msg = (
                f"the plugin reads context symbols {missing} that the daemon does not subscribe: "
                f"add context_symbols = {list(strategy.context_symbols)} to strategy "
                f"{info.name!r} in the run config"
            )
            raise ProviderError(msg)

    async def _load_windows(self, info: StrategyInfo) -> None:
        symbols = list(dict.fromkeys((*info.symbols, *self.strategy.context_symbols)))
        response = await self._request(
            "GET",
            "/candles",
            params={
                "interval": info.interval.value,
                "symbols": ",".join(symbols),
                "limit": info.warmup + HISTORY_SLACK,
                "format": "arrow",
            },
        )
        self._ok(response)
        table = pl.read_ipc_stream(io.BytesIO(response.content))
        self._windows = {
            str(key[0]): part.drop("symbol").select(list(KLINE_SCHEMA))
            for key, part in table.partition_by("symbol", as_dict=True, maintain_order=True).items()
        }

    # -- ticks --------------------------------------------------------------------------------

    def _append(self, symbol: str, rows: list[Any], cap: int) -> None:
        new = pl.DataFrame([rows], schema=KLINE_SCHEMA, orient="row")
        old = self._windows.get(symbol)
        if old is None or old.height == 0:
            self._windows[symbol] = new
            return
        if new["open_time"][0] > old["open_time"][-1]:
            self._windows[symbol] = pl.concat([old, new]).tail(cap)
            return
        merged = pl.concat([old, new]).unique("open_time", keep="last", maintain_order=True)
        self._windows[symbol] = merged.sort("open_time").tail(cap)

    async def _on_tick(self, info: StrategyInfo, message: Mapping[str, Any]) -> None:
        time = _parse_time(message["candle_time"])
        if self._handled is not None and time <= self._handled:
            return
        columns: list[str] = message["columns"]
        assert columns == list(KLINE_SCHEMA)
        cap = info.warmup + HISTORY_SLACK
        traded = set(info.symbols)
        context = set(self.strategy.context_symbols)
        closed: set[str] = set()
        for symbol, values in message["candles"].items():
            if symbol in traded or symbol in context:
                self._append(symbol, [_parse_time(values[0]), *values[1:]], cap)
            if symbol in traded:
                closed.add(symbol)
        await self._answer(info, time, bool(closed))

    async def _answer_collecting(self, info: StrategyInfo, time: datetime) -> None:
        """The engine is waiting for this candle's batch and its tick was published before this
        client connected: rebuild the tick from the windows."""
        if self._handled is not None and time <= self._handled:
            return
        closed = any(
            w.height and w["open_time"][-1] == time
            for s, w in self._windows.items()
            if s in set(info.symbols)
        )
        await self._answer(info, time, closed)

    def _inputs(
        self, info: StrategyInfo, time: datetime
    ) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
        """The plugin's inputs, as `Engine._plugin_inputs` builds them."""
        length = info.interval.length
        limit = time - info.stale_after * length

        def view(symbol: str) -> pl.DataFrame | None:
            frame = self._windows.get(symbol)
            if frame is None:
                return None
            frame = frame.filter(pl.col("open_time") <= time).tail(info.warmup)
            return frame if frame.height else None

        windows: dict[str, pl.DataFrame] = {}
        for symbol in info.symbols:
            window = view(symbol)
            if window is not None and window["open_time"][-1] >= limit:
                windows[symbol] = window
        context = {s: view(s) or _EMPTY_WINDOW for s in self.strategy.context_symbols}
        return windows, context

    async def _answer(self, info: StrategyInfo, time: datetime, any_traded: bool) -> None:
        self._handled = time
        self.stats.ticks += 1
        if not any_traded:  # as in the engine: nothing to trade, an empty batch, no call
            batch = SignalBatch(
                contract=CONTRACT_VERSION,
                strategy=self.name,
                interval=info.interval,
                candle_time=time,
            )
        else:
            windows, context = self._inputs(info, time)
            try:
                frame = await asyncio.to_thread(self.strategy.on_candles, windows, time, context)
                batch = frame_to_batch(
                    frame, strategy=self.name, interval=info.interval, candle_time=time
                )
            except Exception as exc:
                msg = f"on_candles for {time.isoformat()} failed: {type(exc).__name__}: {exc}"
                if self.keep_going:
                    logger.error("provider %s: %s; no batch sent for this candle", self.name, msg)
                    return
                raise ProviderError(msg) from exc
        await self._post(batch)

    async def _post(self, batch: SignalBatch) -> None:
        body = batch.model_dump(mode="json")
        label = batch.candle_time.isoformat()
        for attempt in range(1, POST_ATTEMPTS + 1):
            try:
                response = await self._request("POST", "/signals", json=body)
            except httpx.TransportError as exc:
                if attempt == POST_ATTEMPTS:
                    logger.error(
                        "provider %s: could not post the batch for %s (%s)",
                        self.name,
                        label,
                        type(exc).__name__,
                    )
                    return
                await asyncio.sleep(0.2 * attempt)
                continue
            break
        if response.status_code == 202:
            status = response.json().get("status")
            self.stats.posted += 1
            if status == "late":
                self.stats.late += 1
                logger.warning("provider %s: the batch for %s was LATE", self.name, label)
        elif response.status_code == 409:
            self.stats.duplicates += 1
            logger.info("provider %s: the batch for %s was a duplicate", self.name, label)
        else:
            reason = _reason(response)
            self.stats.refused.append(f"{label}: {reason}")
            logger.error(
                "provider %s: the daemon REFUSED the batch for %s (HTTP %d): %s",
                self.name,
                label,
                response.status_code,
                reason,
            )


def _reason(response: httpx.Response) -> str:
    try:
        body = response.json()
        return str(body.get("reason", body))
    except ValueError:
        return response.text[:200]


async def run_provider(
    strategy: Strategy,
    *,
    url: str,
    token: str | SecretStr,
    name: str,
    keep_going: bool = False,
    stop: asyncio.Event | None = None,
) -> ProviderStats:
    """Run `strategy` as the external provider of the daemon's strategy `name` until `stop` is
    set or the daemon says the run is over. Raises `ProviderError` for a fatal problem. See the
    module docstring."""
    provider = Provider(strategy, url=url, token=token, name=name, keep_going=keep_going, stop=stop)
    await provider.run()
    return provider.stats


def run_provider_sync(
    strategy: Strategy,
    *,
    url: str,
    token: str | SecretStr,
    name: str,
    keep_going: bool = False,
) -> ProviderStats:
    """`run_provider` for a script: its own event loop; Ctrl-C raises `KeyboardInterrupt`."""
    return asyncio.run(
        run_provider(strategy, url=url, token=token, name=name, keep_going=keep_going)
    )
