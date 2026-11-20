"""The daemon's HTTP and WebSocket API (FastAPI), versioned under `/api/v1`.

Served by the daemon on uvicorn in the engine's event loop (`live.daemon`), so every handler runs
on the loop thread that owns the `StateStore` and the feed. Every endpoint except `GET /health`
needs `Authorization: Bearer <token>`, the WebSocket included (checked before the handshake, so
a missing or wrong token gets a plain HTTP 401 and the token never travels in a URL). Tokens are
compared with `hmac.compare_digest` and never logged or echoed.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | alive, mode, feed connection and lag, last tick per interval (no auth) |
| `GET /status` | `Engine.status().to_json()` |
| `GET /strategies/{name}` | what a provider needs: interval, symbols, warm-up, context, exit |
| `GET /candles?interval=&symbols=&limit=&format=arrow\\|json` | windows for a warm-up |
| `WS /ticks?interval=` | one message per tick of that interval, with its closed candles |
| `POST /signals` | a `SignalBatch`: 202 on time or late, 409 duplicate, 422 invalid or refused |
| `POST /control/{pause,resume,flatten}`, `GET /events?since=&limit=` | controls, event feed |
| `GET /trades`, `GET /equity`, `GET /signals`, `GET /providers` | the dashboard's data (below) |
| `GET /alerts`, `POST /alerts/{key}/ack` | active alerts and their history; acknowledge one |

Ticks are fanned out from the engine's tick listener into one bounded queue per WebSocket
client. The listener only does `put_nowait`, so a slow or stuck client can never stall the
engine: when its queue is full it is dropped (its socket closed with code 1013) and a
`ws_client_dropped` event is recorded. A client sees a `hello` message first
(`{"type": "hello", "interval", "last_tick", "contract"}`), then `{"type": "tick", "interval",
"candle_time", "columns", "candles": {symbol: [values in `columns` order]}, "missing"}`.
`hello.collecting` is the candle the engine is waiting for batches on right now, if any: a client
that connected after that tick was published builds it from `/candles` and answers it.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Annotated, Any, Literal

import polars as pl
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import SecretStr

from crypto_moonshot.live.contract import API_PREFIX, CONTRACT_VERSION, ExitPlanIn, SignalBatch
from crypto_moonshot.live.engine import Engine, to_json
from crypto_moonshot.live.feed import Feed, Tick
from crypto_moonshot.live.health import ALERT_KIND
from crypto_moonshot.live.state import AlertRecord, EventLevel, EventRecord, PositionRecord
from crypto_moonshot.models import KLINE_SCHEMA, Interval

logger = logging.getLogger(__name__)

ARROW_MEDIA_TYPE = "application/vnd.apache.arrow.stream"
CANDLE_COLUMNS = list(KLINE_SCHEMA)
MAX_CANDLES = 5000
TICK_QUEUE_SIZE = 64
"""Ticks a WebSocket client may fall behind before it is dropped."""
_DROPPED = None
_CLOSE_TRY_AGAIN = 1013
_CLOSE_POLICY = 1008


def token_matches(header: str | None, token: SecretStr) -> bool:
    """Whether an `Authorization` header carries the bearer `token` (constant-time compare)."""
    if header is None:
        return False
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(value.strip().encode(), token.get_secret_value().encode())


def _candle_values(candle: tuple[Any, ...]) -> list[Any]:
    return [candle[0].isoformat(), *candle[1:]]


def tick_message(tick: Tick) -> str:
    """The JSON text a tick is broadcast as."""
    return json.dumps(
        {
            "type": "tick",
            "interval": tick.interval.value,
            "candle_time": tick.open_time.isoformat(),
            "columns": CANDLE_COLUMNS,
            "candles": {s: _candle_values(tuple(c)) for s, c in tick.candles.items()},
            "missing": list(tick.missing),
        },
        separators=(",", ":"),
    )


@dataclass(eq=False)
class _Client:
    interval: Interval
    queue: asyncio.Queue[str | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=TICK_QUEUE_SIZE)
    )
    dropped: asyncio.Event = field(default_factory=asyncio.Event)


class TickHub:
    """Fans the engine's ticks out to WebSocket clients without ever blocking the engine."""

    def __init__(self, engine: Engine, queue_size: int = TICK_QUEUE_SIZE) -> None:
        self._engine = engine
        self._queue_size = queue_size
        self._clients: set[_Client] = set()
        self.changed = asyncio.Event()
        """Set whenever a client connects or leaves (the replay waits for its providers)."""

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def clients_at(self, interval: Interval) -> int:
        return sum(1 for c in self._clients if c.interval is interval)

    def add(self, interval: Interval) -> _Client:
        client = _Client(interval, asyncio.Queue(maxsize=self._queue_size))
        self._clients.add(client)
        self.changed.set()
        return client

    def remove(self, client: _Client) -> None:
        self._clients.discard(client)
        self.changed.set()

    def finish(self, reason: str) -> None:
        """Tell every client the run is over (`{"type": "end"}`) and close their sockets: a
        provider that gets it exits instead of reconnecting. For the end of a replay."""
        text = json.dumps({"type": "end", "reason": reason})
        for client in list(self._clients):
            try:
                client.queue.put_nowait(text)
                client.queue.put_nowait(_DROPPED)
            except asyncio.QueueFull:
                client.dropped.set()

    def publish(self, tick: Tick) -> None:
        """The engine's tick listener: queue the tick for every client of its interval."""
        targets = [c for c in self._clients if c.interval is tick.interval]
        if not targets:
            return
        text = tick_message(tick)
        for client in targets:
            try:
                client.queue.put_nowait(text)
            except asyncio.QueueFull:
                self._clients.discard(client)
                client.dropped.set()
                self._engine.record_event(
                    EventLevel.WARNING,
                    "ws_client_dropped",
                    f"a {tick.interval.value} tick client fell {self._queue_size} ticks "
                    "behind and was disconnected",
                    {"interval": tick.interval.value, "candle_time": tick.open_time.isoformat()},
                )


def _frame_json(frames: Mapping[str, pl.DataFrame]) -> dict[str, Any]:
    return {
        "columns": CANDLE_COLUMNS,
        "windows": {
            symbol: [
                [row[0].isoformat(), *row[1:]] for row in frame.select(CANDLE_COLUMNS).iter_rows()
            ]
            for symbol, frame in frames.items()
        },
    }


def _frames_arrow(frames: Mapping[str, pl.DataFrame]) -> bytes:
    parts = [
        frame.select(CANDLE_COLUMNS).select(pl.lit(symbol).alias("symbol"), pl.all())
        for symbol, frame in frames.items()
    ]
    table = (
        pl.concat(parts)
        if parts
        else pl.DataFrame(schema=KLINE_SCHEMA).select(
            pl.lit(None, dtype=pl.String).alias("symbol"), pl.all()
        )
    )
    return table.write_ipc_stream(None).getvalue()


PROVIDER_WINDOW = timedelta(hours=24)
MAX_LIST = 1000
MAX_POINTS = 10_000


def trade_json(p: PositionRecord) -> dict[str, Any]:
    """A closed position as the trade list reports it (`GET /trades`)."""
    assert p.entry_time is not None
    assert p.exit_time is not None
    return {
        "id": p.id,
        "strategy": p.strategy,
        "symbol": p.symbol,
        "interval": p.interval.value,
        "entry_time": p.entry_time.isoformat(),
        "entry_price": p.entry_price,
        "exit_time": p.exit_time.isoformat(),
        "exit_price": p.exit_price,
        "exit_reason": p.exit_reason,
        "quantity": p.quantity,
        "cost": p.cost,
        "proceeds": p.proceeds,
        "pnl": p.pnl,
        "return_fraction": None if not p.cost or p.pnl is None else p.pnl / p.cost,
        "hold_seconds": (p.exit_time - p.entry_time).total_seconds(),
        "signal_info": p.signal_info,
    }


def alert_json(alert: AlertRecord) -> dict[str, Any]:
    return {
        "key": alert.key,
        "rule": alert.rule,
        "level": alert.level.value,
        "detail": alert.detail,
        "fired_at": alert.fired_at.isoformat(),
        "acknowledged": alert.acknowledged,
        "acknowledged_at": None
        if alert.acknowledged_at is None
        else alert.acknowledged_at.isoformat(),
    }


def alert_event_json(event: EventRecord) -> dict[str, Any]:
    data = event.data or {}
    return {
        "id": event.id,
        "time": event.time.isoformat(),
        "level": event.level.value,
        "key": data.get("key"),
        "rule": data.get("rule"),
        "state": data.get("state"),
        "detail": data.get("detail"),
    }


def _problem(code: int, reason: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"reason": reason, **extra}, status_code=code)


def create_app(
    engine: Engine,
    feed: Feed,
    token: SecretStr,
    *,
    queue_size: int = TICK_QUEUE_SIZE,
) -> FastAPI:
    """The API over `engine` and `feed`, guarded by the bearer `token`. Registers the tick
    broadcast with `engine.add_tick_listener`; `app.state.hub` is the `TickHub`."""
    if not token.get_secret_value():
        msg = "the API token is empty"
        raise ValueError(msg)
    app = FastAPI(
        title="crypto-moonshot daemon",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    hub = TickHub(engine, queue_size)
    engine.add_tick_listener(hub.publish)
    app.state.hub = hub

    def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        if not token_matches(authorization, token):
            raise HTTPException(
                status_code=401,
                detail="missing or wrong API token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    open_router = APIRouter(prefix=API_PREFIX)
    router = APIRouter(prefix=API_PREFIX, dependencies=[Depends(require_token)])

    def subscribed(interval: Interval) -> list[str]:
        """Every symbol some strategy trades or reads at `interval`, in a stable order."""
        found: dict[str, None] = {}
        for spec in engine.specs.values():
            if spec.interval is interval:
                for symbol in (*spec.symbols, *spec.context_symbols):
                    found[symbol] = None
        return list(found)

    def traded_intervals() -> list[Interval]:
        return sorted({s.interval for s in engine.specs.values()}, key=lambda i: i.length)

    # -- plain endpoints ----------------------------------------------------------------------

    @open_router.get("/health")
    async def health() -> dict[str, Any]:
        now = engine.clock()
        info = feed.health()
        lag = None if info.last_message_at is None else (now - info.last_message_at).total_seconds()
        return {
            "alive": True,
            "mode": "replay" if engine.replay else engine.config.mode,
            "running": engine.running,
            "feed": {
                "connected": info.connected,
                "connections": info.connections,
                "last_message_at": None
                if info.last_message_at is None
                else info.last_message_at.isoformat(),
                "lag_seconds": lag,
                "reconnects": info.reconnects,
            },
            "last_ticks": {i.value: t.isoformat() for i, t in info.last_tick.items()},
            "last_tick_at": None if info.last_tick_at is None else info.last_tick_at.isoformat(),
        }

    @router.get("/status")
    async def status() -> dict[str, Any]:
        return engine.status().to_json()

    @router.get("/openapi.json")
    async def openapi() -> dict[str, Any]:
        return app.openapi()

    @router.get("/strategies/{name}")
    async def strategy(name: str) -> dict[str, Any]:
        spec = engine.specs.get(name)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"unknown strategy {name!r}")
        exit_plan = None
        if spec.exit is not None:
            exit_plan = ExitPlanIn.from_plan(spec.exit).model_dump(exclude_none=True)
        return {
            "name": spec.name,
            "external": spec.external,
            "contract": CONTRACT_VERSION,
            "interval": spec.interval.value,
            "symbols": list(spec.symbols),
            "warmup": spec.warmup,
            "context_symbols": list(spec.context_symbols),
            "stale_after": engine.config.account.stale_after,
            "exit": exit_plan,
        }

    @router.get("/candles", response_model=None)
    async def candles(
        interval: Interval,
        symbols: str | None = None,
        limit: Annotated[int, Query(ge=1, le=MAX_CANDLES)] = 500,
        format: Literal["arrow", "json"] = "arrow",
    ) -> Response:
        allowed = subscribed(interval)
        if not allowed:
            return _problem(
                422,
                f"no strategy trades {interval.value}",
                intervals=[i.value for i in traded_intervals()],
            )
        if symbols is None:
            wanted = allowed
        else:
            wanted = list(dict.fromkeys(s.strip().upper() for s in symbols.split(",") if s.strip()))
            unknown = [s for s in wanted if s not in allowed]
            if unknown:
                return _problem(422, f"symbols not subscribed at {interval.value}: {unknown[:5]}")
        time = interval.floor(engine.clock()) - interval.length
        frames = feed.windows(interval, wanted, time, limit)
        if format == "json":
            return JSONResponse(
                {"interval": interval.value, "time": time.isoformat(), **_frame_json(frames)}
            )
        return Response(
            _frames_arrow(frames),
            media_type=ARROW_MEDIA_TYPE,
            headers={"X-Moonshot-Time": time.isoformat()},
        )

    @router.post("/signals", status_code=202, response_model=None)
    async def signals(batch: SignalBatch) -> Response:
        outcome = await engine.submit_batch(batch)
        if outcome.accepted:
            return JSONResponse(
                {
                    "status": outcome.status,
                    "batch_id": outcome.batch_id,
                    "lateness_seconds": None
                    if outcome.lateness is None
                    else outcome.lateness.total_seconds(),
                },
                status_code=202,
            )
        code = 409 if outcome.status == "duplicate" else 422
        return _problem(code, outcome.reason or outcome.status, status=outcome.status)

    @router.post("/control/{action}")
    async def control(action: Literal["pause", "resume", "flatten"]) -> dict[str, Any]:
        if action == "pause":
            engine.pause()
        elif action == "resume":
            engine.resume()
        else:
            engine.flatten()
        controls = to_json(engine.store.controls())
        assert isinstance(controls, dict)
        return {"status": "ok", "action": action, "controls": controls}

    @router.get("/events")
    async def events(
        since: Annotated[int | None, Query(ge=0)] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ) -> dict[str, Any]:
        """Events with an id above `since`, oldest first (without `since`: the latest ones)."""
        store = engine.store
        records = store.recent_events(limit) if since is None else store.events_since(since, limit)
        return {
            "events": [to_json(r) for r in records],
            "last_id": records[-1].id if records else (since or 0),
        }

    # -- the dashboard's data -----------------------------------------------------------------

    def known_strategy(name: str | None) -> None:
        if name is not None and name not in engine.specs:
            raise HTTPException(status_code=404, detail=f"unknown strategy {name!r}")

    @router.get("/trades")
    async def trades(
        strategy: str | None = None,
        limit: Annotated[int, Query(ge=1, le=MAX_LIST)] = 100,
    ) -> dict[str, Any]:
        """Closed positions, newest exit first."""
        known_strategy(strategy)
        rows = [trade_json(p) for p in engine.store.recent_closed_positions(limit, strategy)]
        return {"strategy": strategy, "limit": limit, "count": len(rows), "trades": rows}

    @router.get("/equity")
    async def equity(
        strategy: str | None = None,
        limit: Annotated[int, Query(ge=1, le=MAX_POINTS)] = 2000,
    ) -> dict[str, Any]:
        """Realised equity per strategy (a point per closed position, the latest `limit`) and
        the current mark-to-market: open positions valued at the feed's last close (their
        entry price when it has none), before the exit fee."""
        known_strategy(strategy)
        store = engine.store
        out: dict[str, Any] = {}
        for name, spec in engine.specs.items():
            if strategy is not None and name != strategy:
                continue
            realised = store.realised_pnl(name)
            unrealised = 0.0
            open_count = unpriced = 0
            for position in store.live_positions(name):
                if position.status.value != "open":
                    continue
                assert position.quantity is not None
                assert position.entry_price is not None
                candle = feed.last_candle(position.interval, position.symbol)
                if candle is None:
                    unpriced += 1
                price = position.entry_price if candle is None else candle.close
                unrealised += position.quantity * price - position.cost
                open_count += 1
            points = store.equity_points(name, spec.capital, limit)
            out[name] = {
                "capital": spec.capital,
                "realised_pnl": realised,
                "realised_equity": spec.capital + realised,
                "unrealised_pnl": unrealised,
                "equity": spec.capital + realised + unrealised,
                "open_positions": open_count,
                "unpriced_positions": unpriced,
                "points": [{"time": t.isoformat(), "equity": v} for t, v in points],
            }
        return {"time": engine.clock().isoformat(), "limit": limit, "strategies": out}

    @router.get("/signals")
    async def signals_list(
        strategy: str | None = None,
        limit: Annotated[int, Query(ge=1, le=MAX_LIST)] = 100,
    ) -> dict[str, Any]:
        """Recent signals, newest first, with their outcome and reason."""
        known_strategy(strategy)
        rows = [
            {
                "id": item.signal.id,
                "strategy": item.strategy,
                "interval": item.interval.value,
                "candle_time": item.candle_time.isoformat(),
                "batch_status": item.batch_status.value,
                "symbol": item.signal.symbol,
                "action": item.signal.action,
                "priority": item.signal.priority,
                "weight": item.signal.weight,
                "outcome": None if item.signal.outcome is None else item.signal.outcome.value,
                "reason": item.signal.reason,
                "info": item.signal.info,
            }
            for item in engine.store.recent_signals(limit, strategy)
        ]
        return {"strategy": strategy, "limit": limit, "count": len(rows), "signals": rows}

    @router.get("/providers")
    async def providers() -> dict[str, Any]:
        """Per strategy, over the last 24 hours (by candle time): batches on time, late and
        missed, the median and worst lateness and the last batch received."""
        store = engine.store
        since = engine.clock() - PROVIDER_WINDOW
        seen = {p.name: p for p in store.providers()}
        stats = {s.strategy: s for s in store.batch_stats(since)}
        rows = []
        for name in sorted(engine.specs):
            spec = engine.specs[name]
            stat = stats[name]
            heartbeat = seen.get(name)
            rows.append(
                {
                    "strategy": name,
                    "external": spec.external,
                    "interval": spec.interval.value,
                    "on_time": stat.on_time,
                    "late": stat.late,
                    "missed": stat.missed,
                    "median_lateness_seconds": stat.median_lateness,
                    "worst_lateness_seconds": stat.worst_lateness,
                    "last_batch_time": None
                    if stat.last_batch_time is None
                    else stat.last_batch_time.isoformat(),
                    "last_seen": None if heartbeat is None else heartbeat.last_seen.isoformat(),
                }
            )
        return {
            "window_hours": PROVIDER_WINDOW.total_seconds() / 3600,
            "since": since.isoformat(),
            "providers": rows,
        }

    @router.get("/alerts")
    async def alerts(
        history: Annotated[int, Query(ge=0, le=500)] = 50,
    ) -> dict[str, Any]:
        """Active alerts (oldest first, with their acknowledged flag) and the latest alert
        events (newest first)."""
        store = engine.store
        active = store.alerts()
        events = store.recent_events_of_kind(ALERT_KIND, history) if history else []
        return {
            "active": [alert_json(a) for a in active],
            "unacknowledged": sum(1 for a in active if not a.acknowledged),
            "history": [alert_event_json(e) for e in events],
        }

    @router.post("/alerts/{key}/ack")
    async def acknowledge(key: str) -> dict[str, Any]:
        """Acknowledge an active alert: it stays active but stops counting as unacknowledged,
        until it resolves and fires again."""
        store = engine.store
        if not store.acknowledge_alert(key, engine.clock()):
            raise HTTPException(status_code=404, detail=f"no active alert {key!r}")
        engine.record_event(
            EventLevel.INFO, "alert_acknowledged", f"{key} acknowledged", {"key": key}
        )
        alert = store.alert(key)
        assert alert is not None
        return {"status": "acknowledged", "alert": alert_json(alert)}

    # -- the tick stream ----------------------------------------------------------------------

    @app.websocket(f"{API_PREFIX}/ticks")
    async def ticks(websocket: WebSocket, interval: str = "") -> None:
        if not token_matches(websocket.headers.get("authorization"), token):
            await websocket.send_denial_response(
                Response("missing or wrong API token", status_code=401)
            )
            return
        try:
            wanted = Interval(interval)
        except ValueError:
            wanted = None
        if wanted is None or not subscribed(wanted):
            await websocket.accept()
            await websocket.close(
                code=_CLOSE_POLICY,
                reason=f"interval must be one of {[i.value for i in traded_intervals()]}",
            )
            return
        await websocket.accept()
        client = hub.add(wanted)
        last = engine.store.last_tick(wanted)
        pending = engine.collecting(wanted)
        try:
            await websocket.send_json(
                {
                    "type": "hello",
                    "interval": wanted.value,
                    "last_tick": None if last is None else last.isoformat(),
                    "collecting": None if pending is None else pending.isoformat(),
                    "contract": CONTRACT_VERSION,
                }
            )
            await _pump(websocket, client)
        finally:
            hub.remove(client)

    app.include_router(open_router)
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"where": ".".join(str(p) for p in e["loc"]), "reason": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        first = errors[0] if errors else {"where": "request", "reason": "invalid"}
        return JSONResponse(
            {
                "status": "invalid",
                "reason": f"{first['where']}: {first['reason']}",
                "errors": errors,
            },
            status_code=422,
        )

    return app


async def _pump(websocket: WebSocket, client: _Client) -> None:
    """Send queued ticks until the client disconnects or is dropped."""

    async def send() -> None:
        while True:
            text = await client.queue.get()
            if text is _DROPPED:
                return
            await websocket.send_text(text)

    async def receive() -> None:
        while True:  # anything the client says is ignored; this notices the disconnect
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

    sender = asyncio.create_task(send())
    receiver = asyncio.create_task(receive())
    dropped = asyncio.create_task(client.dropped.wait())
    tasks = (sender, receiver, dropped)
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    if receiver in done or (sender in done and sender.exception() is not None):
        return  # the client went away
    code, reason = (
        (_CLOSE_TRY_AGAIN, "too slow: ticks queued up") if dropped in done else (1000, "")
    )
    try:
        async with asyncio.timeout(2.0):
            await websocket.close(code=code, reason=reason)
    except (RuntimeError, TimeoutError):  # the socket is gone or stuck
        logger.debug("tick client did not close cleanly")
