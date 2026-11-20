"""The daemon wiring: `run_daemon` against a fake Binance (WebSocket server and REST)."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import threading
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from kline_fakes import FakeBinance
from pydantic import SecretStr
from stream_fakes import FakeStreamServer, kline_message, until

from crypto_moonshot.config import Settings
from crypto_moonshot.data.binance import BinanceClient
from crypto_moonshot.data.stream import StreamConfig
from crypto_moonshot.live.config import ConfigError, RunConfig, config_from_dict
from crypto_moonshot.live.daemon import DaemonError, run_daemon
from crypto_moonshot.live.state import PositionStatus, StateStore
from crypto_moonshot.models import Interval

pytestmark = pytest.mark.anyio

M5 = Interval.M5
T0 = datetime(2024, 3, 1, tzinfo=UTC)
TOKEN = "tkn-daemon-5Xn2Qe8uLp"
SYMBOLS = ["AAAUSDT", "BBBUSDT"]
PLUGIN = """
from datetime import timedelta

import polars as pl

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.models import Interval


class Enter:
    interval = Interval.M5
    warmup = 3
    context_symbols = ()
    exit = ExitPlan(max_hold=timedelta(hours=1))

    def on_candles(self, windows, time, context):
        return pl.DataFrame({"symbol": ["AAAUSDT"], "priority": [0.0]})
"""


def at(k: int) -> datetime:
    return T0 + k * M5.length


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def make_config(tmp_path: Path, **top: Any) -> RunConfig:
    plugin = tmp_path / "plugin.py"
    plugin.write_text(PLUGIN)
    return config_from_dict(
        {
            "state": str(tmp_path / "state" / "paper.sqlite"),
            "account": {"initial_cash": 1000.0},
            "strategies": [
                {
                    "name": "enter",
                    "source": f"{plugin}:Enter",
                    "symbols": SYMBOLS,
                    "capital": 1000.0,
                    "sizing": {"stake": 100.0},
                }
            ],
            **top,
        }
    )


def make_settings(server: FakeStreamServer, tmp_path: Path, token: str | None = TOKEN) -> Settings:
    return Settings(
        api_token=None if token is None else SecretStr(token),
        stream_url=server.url,
        data_dir=tmp_path / "no-data",
    )


class Daemon:
    """`run_daemon` as a task with a fake REST, a controllable clock and a stop event."""

    def __init__(
        self, config: RunConfig, settings: Settings, *, install_signals: bool = False
    ) -> None:
        self.rest = FakeBinance({s: [at(k) for k in range(-3, 1)] for s in SYMBOLS})
        self.clock = Clock(at(1) + timedelta(seconds=1))
        self.stop = asyncio.Event()
        self.port: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self.config = config
        self.task = asyncio.create_task(
            run_daemon(
                config,
                settings,
                port=0,
                client=self.rest,
                stream_config=StreamConfig(
                    base_url=settings.stream_url,
                    subscribe_gap=0.0,
                    backoff_base=0.01,
                    backoff_max=0.05,
                ),
                clock=self.clock,
                stop=self.stop,
                install_signals=install_signals,
                on_ready=self.port.set_result,
            )
        )

    async def api(self) -> httpx.AsyncClient:
        port = await asyncio.wait_for(self.port, 20)
        return httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}/api/v1", headers={"Authorization": f"Bearer {TOKEN}"}
        )


@pytest.fixture
async def server() -> AsyncIterator[FakeStreamServer]:
    async with FakeStreamServer() as fake:
        yield fake


async def close_candle(server: FakeStreamServer, k: int) -> None:
    for symbol in SYMBOLS:
        await server.send(kline_message(symbol, M5, at(k)))


async def test_the_daemon_processes_ticks_serves_the_api_and_stops_cleanly(
    tmp_path: Path, server: FakeStreamServer, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")
    config = make_config(tmp_path)
    daemon = Daemon(config, make_settings(server, tmp_path))
    async with await daemon.api() as api:
        assert (await api.get("/health", headers={"Authorization": ""})).status_code == 200
        assert (await api.get("/status", headers={"Authorization": ""})).status_code == 401
        await until(lambda: server.connections != [] and server.subscribed(0) != [])
        assert sorted(server.subscribed(0)) == ["aaausdt@kline_5m", "bbbusdt@kline_5m"]
        await close_candle(server, 1)

        async def processed() -> bool:
            status = (await api.get("/status")).json()
            return "5m" in status["last_ticks"]

        async with asyncio.timeout(10):
            while not await processed():
                await asyncio.sleep(0.02)
        status = (await api.get("/status")).json()
        assert status["last_ticks"]["5m"] == at(1).isoformat()
        assert status["mode"] == "paper"
        assert status["running"] is True
        assert [p["symbol"] for p in status["positions"]] == ["AAAUSDT"]  # the plugin entered
        health = (await api.get("/health")).json()
        assert health["last_ticks"] == {"5m": at(1).isoformat()}
        assert health["feed"]["connected"] is True
        port = daemon.port.result()

    daemon.stop.set()
    await asyncio.wait_for(daemon.task, 20)  # a clean return
    assert daemon.task.exception() is None
    with pytest.raises(OSError, match=r"refused|Connect"):
        socket.create_connection(("127.0.0.1", port), timeout=2).close()
    # the state file is closed and consistent
    with StateStore.open(config.state) as store:
        assert store.last_ticks() == {M5: at(1)}
        (position,) = store.live_positions()
        assert position.symbol == "AAAUSDT"
        assert position.status is PositionStatus.PENDING
        kinds = [e.kind for e in store.events_since()]
        assert "daemon_started" in kinds
        assert kinds[-1] == "daemon_stopped"
        assert store._db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        derived = store.derive_book("enter", 1000.0, at(2))
        assert derived.cash == pytest.approx(900.0)
    assert TOKEN not in caplog.text


async def test_a_stop_signal_shuts_it_down(tmp_path: Path, server: FakeStreamServer) -> None:
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers need the main thread")
    daemon = Daemon(make_config(tmp_path), make_settings(server, tmp_path), install_signals=True)
    await asyncio.wait_for(daemon.port, 20)
    await until(lambda: server.connections != [] and server.subscribed(0) != [])
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(daemon.task, 20)
    assert daemon.task.exception() is None
    # the daemon's handlers are gone again: there is nothing left to remove
    assert asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM) is False


async def test_it_refuses_to_start_without_a_token(
    tmp_path: Path, server: FakeStreamServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MOONSHOT_API_TOKEN", raising=False)
    for token in (None, "", "  "):
        settings = make_settings(server, tmp_path, token)
        with pytest.raises(DaemonError, match="MOONSHOT_API_TOKEN is not set") as caught:
            await run_daemon(make_config(tmp_path), settings, port=0, install_signals=False)
        assert caught.value.usage
    assert not (tmp_path / "state").exists()  # nothing was opened


@pytest.mark.parametrize("mode", ["testnet", "live"])
async def test_it_refuses_modes_that_would_place_orders(
    tmp_path: Path, server: FakeStreamServer, mode: str
) -> None:
    config = make_config(tmp_path, mode=mode)
    with pytest.raises(ConfigError, match="not available in this build"):
        await run_daemon(config, make_settings(server, tmp_path), port=0, install_signals=False)
    assert not (tmp_path / "state").exists()


async def test_a_taken_port_is_a_clear_error_and_the_state_is_closed(
    tmp_path: Path, server: FakeStreamServer
) -> None:
    config = make_config(tmp_path)
    with socket.create_server(("127.0.0.1", 0)) as taken:
        port = taken.getsockname()[1]
        with pytest.raises(DaemonError, match=f"cannot listen on 127.0.0.1:{port}"):
            await run_daemon(
                config,
                make_settings(server, tmp_path),
                port=port,
                client=FakeBinance(),
                install_signals=False,
            )
    with StateStore.open(config.state) as store:  # not locked
        store.set_meta("x", "1")


async def test_the_rest_client_is_built_for_a_long_running_process(
    tmp_path: Path, server: FakeStreamServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[dict[str, Any]] = []
    rest = FakeBinance({s: [at(k) for k in range(-3, 1)] for s in SYMBOLS})

    def from_settings(settings: Settings, **kwargs: Any) -> FakeBinance:
        built.append(kwargs)
        return rest

    monkeypatch.setattr(BinanceClient, "from_settings", from_settings)
    stop = asyncio.Event()
    ready: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(
        run_daemon(
            make_config(tmp_path),
            make_settings(server, tmp_path),
            port=0,
            stream_config=StreamConfig(base_url=server.url, subscribe_gap=0.0),
            clock=Clock(at(1) + timedelta(seconds=1)),
            stop=stop,
            install_signals=False,
            on_ready=ready.set_result,
        )
    )
    await asyncio.wait_for(ready, 20)
    stop.set()
    await asyncio.wait_for(task, 20)
    assert built[0]["max_rate_limit_hits"] >= 1_000_000  # never aborts for good after 3 hits
    assert rest.closed  # the daemon closes a client it built
