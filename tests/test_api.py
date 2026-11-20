"""The daemon API: auth, batches, candles, the tick stream, controls and events."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from datetime import timedelta
from pathlib import Path

import httpx
import polars as pl
import pytest
from api_fakes import CONTEXT, SYMBOLS, T0, TOKEN, ApiRig, at, batch
from engine_fakes import until
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from crypto_moonshot.data.stream import Candle
from crypto_moonshot.live.api import API_PREFIX, TickHub, tick_message
from crypto_moonshot.live.contract import SignalIn
from crypto_moonshot.live.feed import Tick
from crypto_moonshot.live.state import EventLevel
from crypto_moonshot.models import KLINE_SCHEMA, Interval

pytestmark = pytest.mark.anyio

AUTH = {"Authorization": f"Bearer {TOKEN}"}


async def post(rig: ApiRig, *signals: SignalIn, k: int = 0, name: str = "ext") -> httpx.Response:
    body = batch(name, k, *signals).model_dump(mode="json")
    return await rig.http.post("/signals", json=body)


async def closed_by_server(ws: ClientConnection) -> ConnectionClosed:
    """Read until the server closes the socket; the exception says how."""
    try:
        async with asyncio.timeout(5):
            while True:
                await ws.recv()
    except ConnectionClosed as exc:
        return exc


# -- auth ---------------------------------------------------------------------------------------


async def test_every_endpoint_but_health_needs_the_token(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        routes = [
            (method.upper(), path)
            for path, item in rig.app.openapi()["paths"].items()
            for method in item
        ]
        assert len(routes) == 8
        bare = httpx.AsyncClient(base_url=rig.url)
        for method, path in routes:
            path = path.replace("{name}", "ext").replace("{action}", "pause")
            for headers in (
                {},
                {"Authorization": "Bearer wrong"},
                {"Authorization": f"Basic {TOKEN}"},
                {"Authorization": TOKEN},
                {"Authorization": f"Bearer {TOKEN}x"},
            ):
                response = await bare.request(method, path, headers=headers, json={})
                if path.endswith("/health"):
                    assert response.status_code == 200
                else:
                    assert response.status_code == 401, (method, path, headers)
                    assert response.headers["www-authenticate"] == "Bearer"
            if not path.endswith("/health"):
                good = await bare.request(method, path, headers=AUTH, params={"interval": "1h"})
                assert good.status_code != 401, (method, path)
        await bare.aclose()


async def test_the_websocket_needs_the_token_too(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        uri = f"ws://127.0.0.1:{rig.port}{API_PREFIX}/ticks?interval=1h"
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            with pytest.raises(InvalidStatus) as caught:
                async with connect(uri, additional_headers=headers):
                    pass
            assert caught.value.response.status_code == 401
        # a token in the query string is not accepted
        with pytest.raises(InvalidStatus) as caught:
            async with connect(uri + f"&token={TOKEN}"):
                pass
        assert caught.value.response.status_code == 401
        async with connect(uri, additional_headers=AUTH) as ws:
            hello = json.loads(await ws.recv())
            assert hello["type"] == "hello"
            assert hello["interval"] == "1h"


async def test_the_token_never_appears_in_logs_events_status_or_errors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    caplog.set_level(logging.DEBUG)
    async with ApiRig(tmp_path) as rig:
        seen: list[str] = []
        for headers in ({}, {"Authorization": "Bearer wrong"}, AUTH):
            for path in ("/status", "/events", "/strategies/ext", "/candles?interval=1h"):
                seen.append(
                    (
                        await httpx.AsyncClient(base_url=rig.url + API_PREFIX).get(
                            path, headers=headers
                        )
                    ).text
                )
        seen.append((await rig.http.post("/signals", content=b"{not json")).text)
        seen.append((await rig.http.get("/health")).text)
        await rig.tick(0)
        seen.append(json.dumps(rig.engine.status().to_json()))
        seen.append(repr(rig.app) + repr(rig.engine.config) + repr(rig.config))
        seen.extend(e.message + json.dumps(e.data) for e in rig.store.events_since())
    out = capsys.readouterr()
    everything = "\n".join([*seen, caplog.text, out.out, out.err])
    assert TOKEN not in everything


# -- strategies, status, health --------------------------------------------------------------


async def test_strategy_status_and_health(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        info = (await rig.http.get("/strategies/ext")).json()
        assert info == {
            "name": "ext",
            "external": True,
            "contract": 1,
            "interval": "1h",
            "symbols": list(SYMBOLS),
            "warmup": 3,
            "context_symbols": [CONTEXT],
            "stale_after": 2,
            "exit": {"max_hold": "10h", "tp_fill": "touch"},
        }
        inproc = (await rig.http.get("/strategies/inproc")).json()
        assert inproc["external"] is False
        assert (await rig.http.get("/strategies/nope")).status_code == 404
        status = (await rig.http.get("/status")).json()
        assert status == rig.engine.status().to_json()
        assert set(status["books"]) == {"ext", "inproc"}
        await rig.tick(0)
        health = (await rig.http.get("/health", headers={"Authorization": ""})).json()
        assert health["alive"] is True
        assert health["mode"] == "paper"
        assert health["running"] is True
        assert health["feed"]["connected"] is True


# -- batches ---------------------------------------------------------------------------------


async def test_batch_status_codes(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        rig.clock.now = at(1) + timedelta(seconds=1)
        on_time = await post(rig, SignalIn(symbol="AAAUSDT"), k=0)
        assert on_time.status_code == 202
        assert on_time.json()["status"] == "on_time"
        assert on_time.json()["lateness_seconds"] == 1.0
        duplicate = await post(rig, k=0)
        assert duplicate.status_code == 409
        assert duplicate.json()["status"] == "duplicate"
        await rig.tick(0)
        assert rig.store.batch_for("ext", Interval.H1, at(0)) is not None
        # candle 1 passes with no batch: the daemon records it missed, then a batch is late
        await rig.tick(1)
        late = await post(rig, k=1)
        assert late.status_code == 202
        assert late.json()["status"] == "late"
        assert late.json()["lateness_seconds"] > 0
        # refusals: a candle that has not closed, an unknown strategy, an in-process one
        future = await post(rig, k=30)
        assert future.status_code == 422
        assert "not closed" in future.json()["reason"]
        unknown = await post(rig, k=2, name="nobody")
        assert unknown.status_code == 422
        assert "unknown strategy" in unknown.json()["reason"]
        inproc = await post(rig, k=2, name="inproc")
        assert inproc.status_code == 422
        assert "in process" in inproc.json()["reason"]


async def test_invalid_batches_are_422_with_a_reason(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        good = batch("ext", 0).model_dump(mode="json")
        cases = {
            "contract": {**good, "contract": 2},
            "candle_time": {**good, "candle_time": at(0).replace(minute=7).isoformat()},
            "signals": {**good, "signals": [{"symbol": "AAAUSDT", "action": "buy"}]},
            "extra": {**good, "surprise": 1},
        }
        for where, body in cases.items():
            response = await rig.http.post("/signals", json=body)
            assert response.status_code == 422, where
            document = response.json()
            assert document["status"] == "invalid"
            assert where in document["reason"] or where == "extra" or where == "candle_time"
            assert document["errors"]
        response = await rig.http.post("/signals", content=b"{not json")
        assert response.status_code == 422


# -- candles ---------------------------------------------------------------------------------


async def test_candles_as_arrow_and_json(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        for k in range(5):
            await rig.tick(k)
        wanted = f"{SYMBOLS[0]},{CONTEXT}"
        response = await rig.http.get(
            "/candles", params={"interval": "1h", "symbols": wanted, "limit": 3}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.apache.arrow.stream"
        table = pl.read_ipc_stream(io.BytesIO(response.content))
        assert table.columns == ["symbol", *KLINE_SCHEMA]
        assert table["symbol"].unique(maintain_order=True).to_list() == [SYMBOLS[0], CONTEXT]
        assert table.height == 6
        assert table.filter(pl.col("symbol") == CONTEXT)["open_time"].to_list() == [
            at(2), at(3), at(4)
        ]  # fmt: skip
        assert table.schema["open_time"] == KLINE_SCHEMA["open_time"]
        as_json = await rig.http.get(
            "/candles",
            params={"interval": "1h", "symbols": SYMBOLS[0], "limit": 2, "format": "json"},
        )
        document = as_json.json()
        assert document["columns"] == list(KLINE_SCHEMA)
        rows = document["windows"][SYMBOLS[0]]
        assert [r[0] for r in rows] == [at(3).isoformat(), at(4).isoformat()]
        arrow = pl.read_ipc_stream(
            io.BytesIO(
                (
                    await rig.http.get(
                        "/candles", params={"interval": "1h", "symbols": SYMBOLS[0], "limit": 2}
                    )
                ).content
            )
        )
        assert arrow["close"].to_list() == [r[4] for r in rows]
        # everything subscribed when no symbols are named
        every = await rig.http.get("/candles", params={"interval": "1h", "format": "json"})
        assert set(every.json()["windows"]) == {*SYMBOLS, CONTEXT, "DDDUSDT"}


async def test_candles_rejects_what_is_not_subscribed(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        other = await rig.http.get("/candles", params={"interval": "5m"})
        assert other.status_code == 422
        assert "no strategy trades 5m" in other.json()["reason"]
        unknown = await rig.http.get("/candles", params={"interval": "1h", "symbols": "ZZZUSDT"})
        assert unknown.status_code == 422
        assert "ZZZUSDT" in unknown.json()["reason"]
        bad = await rig.http.get("/candles", params={"interval": "7x"})
        assert bad.status_code == 422
        over = await rig.http.get("/candles", params={"interval": "1h", "limit": 10**6})
        assert over.status_code == 422


# -- the tick stream -------------------------------------------------------------------------


async def test_ticks_reach_the_subscribers_of_that_interval(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        uri = f"ws://127.0.0.1:{rig.port}{API_PREFIX}/ticks?interval=1h"
        async with (
            connect(uri, additional_headers=AUTH) as one,
            connect(uri, additional_headers=AUTH) as two,
        ):
            assert json.loads(await one.recv())["collecting"] is None
            await two.recv()
            await until(lambda: rig.app.state.hub.client_count == 2)
            await rig.tick(0, [*SYMBOLS, CONTEXT])
            for ws in (one, two):
                message = json.loads(await asyncio.wait_for(ws.recv(), 5))
                assert message["type"] == "tick"
                assert message["interval"] == "1h"
                assert message["candle_time"] == at(0).isoformat()
                assert message["columns"] == list(KLINE_SCHEMA)
                assert set(message["candles"]) == {*SYMBOLS, CONTEXT}
                assert message["missing"] == ["DDDUSDT"]
                row = dict(zip(message["columns"], message["candles"]["AAAUSDT"], strict=True))
                assert row["open_time"] == at(0).isoformat()
                assert row["close"] == 100.0
        await until(lambda: rig.app.state.hub.client_count == 0)
        # a client of another interval is refused: nothing trades it
        async with connect(uri.replace("1h", "5m"), additional_headers=AUTH) as ws:
            closed = await closed_by_server(ws)
            assert closed.rcvd is not None
            assert closed.rcvd.code == 1008


async def test_a_client_that_connects_mid_tick_is_told_what_is_pending(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        rig.clock.now = at(1) + timedelta(seconds=1)
        await rig.feed.emit(0)
        await until(lambda: rig.engine.collecting(Interval.H1) == at(0))
        uri = f"ws://127.0.0.1:{rig.port}{API_PREFIX}/ticks?interval=1h"
        async with connect(uri, additional_headers=AUTH) as ws:
            hello = json.loads(await ws.recv())
        assert hello["collecting"] == at(0).isoformat()
        await until(lambda: at(0) in rig.done)


def _tick(k: int, interval: Interval = Interval.H1) -> Tick:
    candle = Candle(at(k), 1.0, 2.0, 0.5, 1.5, 3.0, 4.0, 5, 6.0, 7.0)
    return Tick(interval, at(k), {"AAAUSDT": candle}, (), at(k + 1))


async def test_a_slow_client_is_dropped_and_never_stalls_the_engine(tmp_path: Path) -> None:
    async with ApiRig(tmp_path, queue_size=2) as rig:
        hub: TickHub = rig.app.state.hub
        lazy = hub.add(Interval.H1)  # never reads: its queue fills up
        fine = hub.add(Interval.H1)
        started = asyncio.get_running_loop().time()
        for k in range(6):
            hub.publish(_tick(k))
            fine.queue.get_nowait()
        assert asyncio.get_running_loop().time() - started < 0.5
        assert lazy.dropped.is_set()
        assert not fine.dropped.is_set()
        assert hub.client_count == 1
        events = [e for e in rig.store.events_since() if e.kind == "ws_client_dropped"]
        assert len(events) == 1
        assert events[0].level is EventLevel.WARNING
        # the engine keeps processing ticks while a stuck client is registered
        stuck = hub.add(Interval.H1)
        for k in range(4):
            await rig.tick(k)
        assert stuck.dropped.is_set()
        assert [t for t in rig.done] == [at(k) for k in range(4)]


async def test_a_dropped_websocket_client_is_closed_with_1013(tmp_path: Path) -> None:
    async with ApiRig(tmp_path, queue_size=1) as rig:
        uri = f"ws://127.0.0.1:{rig.port}{API_PREFIX}/ticks?interval=1h"
        async with connect(uri, additional_headers=AUTH) as ws:
            await ws.recv()
            hub: TickHub = rig.app.state.hub
            await until(lambda: hub.client_count == 1)
            # no await between the publishes: the sender cannot drain, so the queue overflows
            hub.publish(_tick(0))
            hub.publish(_tick(1))
            closed = await closed_by_server(ws)
            assert closed.rcvd is not None
            assert closed.rcvd.code == 1013


async def test_tick_message_is_compact_json() -> None:
    text = tick_message(_tick(3))
    message = json.loads(text)
    assert message["candles"]["AAAUSDT"][0] == at(3).isoformat()
    assert " " not in text


# -- controls and events ---------------------------------------------------------------------


async def test_controls_and_events(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        paused = await rig.http.post("/control/pause")
        assert paused.json() == {
            "status": "ok",
            "action": "pause",
            "controls": {"paused": True, "flatten_requested": False},
        }
        assert rig.store.controls().paused
        flat = (await rig.http.post("/control/flatten")).json()
        assert flat["controls"] == {"paused": True, "flatten_requested": True}
        resumed = (await rig.http.post("/control/resume")).json()
        assert resumed["controls"]["paused"] is False
        assert (await rig.http.post("/control/explode")).status_code == 422
        events = (await rig.http.get("/events")).json()
        kinds = [e["kind"] for e in events["events"]]
        assert kinds == ["paused", "flatten_requested", "resumed"]
        assert events["last_id"] == events["events"][-1]["id"]
        first = events["events"][0]["id"]
        after = (await rig.http.get("/events", params={"since": first, "limit": 1})).json()
        assert [e["kind"] for e in after["events"]] == ["flatten_requested"]
        none = (await rig.http.get("/events", params={"since": events["last_id"]})).json()
        assert none == {"events": [], "last_id": events["last_id"]}
        latest = (await rig.http.get("/events", params={"limit": 2})).json()
        assert [e["kind"] for e in latest["events"]] == ["flatten_requested", "resumed"]
        assert T0.isoformat()[:4] in events["events"][0]["time"]
        assert json.loads(json.dumps(events))["events"][0]["level"] == "info"
