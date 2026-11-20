"""A running engine on a hand-driven feed with its API served by uvicorn on a random port."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
from engine_fakes import FakeClock, FakeFeed, Scheduled, flat_bars, until
from pydantic import SecretStr

from crypto_moonshot.live.api import create_app
from crypto_moonshot.live.config import RunConfig, config_from_dict
from crypto_moonshot.live.contract import CONTRACT_VERSION, SignalBatch, SignalIn
from crypto_moonshot.live.daemon import serve_app
from crypto_moonshot.live.engine import Engine, Step, build_specs
from crypto_moonshot.live.state import StateStore
from crypto_moonshot.models import Interval

TOKEN = "tkn-7Qz93xLmVb1e0RfA"
"""A recognisable fake token: tests grep logs and outputs for it."""
H1 = Interval.H1
HOUR = timedelta(hours=1)
T0 = datetime(2024, 3, 1, tzinfo=UTC)
SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
CONTEXT = "BTCUSDT"


def at(k: int) -> datetime:
    return T0 + k * HOUR


def batch(name: str, k: int, *signals: SignalIn, interval: Interval = H1) -> SignalBatch:
    return SignalBatch(
        contract=CONTRACT_VERSION,
        strategy=name,
        interval=interval,
        candle_time=at(k),
        signals=list(signals),
    )


def run_config(
    path: Path, *, extra: Sequence[dict[str, Any]] = (), deadline: str = "1.05s"
) -> RunConfig:
    return config_from_dict(
        {
            "state": str(path),
            "account": {"initial_cash": 10_000.0},
            "engine": {"signal_deadline": deadline, "entry_grace": "30m"},
            "strategies": [
                {
                    "name": "ext",
                    "source": "external",
                    "interval": "1h",
                    "warmup": 3,
                    "symbols": list(SYMBOLS),
                    "context_symbols": [CONTEXT],
                    "capital": 1000.0,
                    "sizing": {"stake": 100.0},
                    "exit": {"max_hold": "10h"},
                },
                {
                    "name": "inproc",
                    "source": "tests:not_loaded",
                    "symbols": ["DDDUSDT"],
                    "capital": 1000.0,
                    "sizing": {"stake": 100.0},
                },
                *extra,
            ],
        }
    )


class ApiRig:
    """Engine + `FakeFeed` + API server + an authorised httpx client (`http`)."""

    def __init__(
        self, tmp_path: Path, *, queue_size: int = 64, token: str = TOKEN, deadline: str = "1.05s"
    ) -> None:
        self.config = run_config(tmp_path / "state.sqlite", deadline=deadline)
        self.clock = FakeClock(at(0))
        self.feed = FakeFeed(H1, T0, {s: flat_bars(40) for s in (*SYMBOLS, CONTEXT, "DDDUSDT")})
        self.store = StateStore.open(self.config.state)
        plugin = Scheduled({}, name="inproc")
        self.specs = build_specs(self.config, None, plugins={"inproc": plugin})
        self.done: list[datetime] = []
        self.engine = Engine(
            self.config, self.specs, self.store, clock=self.clock, step_hook=self._hook
        )
        self.app = create_app(self.engine, self.feed, SecretStr(token), queue_size=queue_size)
        self.token = token
        self.task: asyncio.Task[None] | None = None
        self.port = 0
        self.http: httpx.AsyncClient
        self._serving: Any = None

    def _hook(self, step: str, tick: Any) -> None:
        if step == Step.TICK_DONE:
            self.done.append(tick.open_time)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def __aenter__(self) -> Self:
        self._serving = serve_app(self.app, "127.0.0.1", 0)
        self.port = await self._serving.__aenter__()
        self.http = httpx.AsyncClient(
            base_url=self.url + "/api/v1", headers={"Authorization": f"Bearer {self.token}"}
        )
        self.task = asyncio.create_task(self.engine.run(self.feed))
        await until(lambda: self.engine._started)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.http.aclose()
        await self.engine.stop()
        assert self.task is not None
        await self.task
        await self._serving.__aexit__(None, None, None)
        self.store.close()

    async def tick(self, k: int, symbols: Sequence[str] | None = None) -> None:
        """Candle k closes (one second ago by the clock) and is fully processed. External
        batches for it must arrive within the 0.05 s left of the deadline."""
        self.clock.now = at(k + 1) + timedelta(seconds=1)
        await self.feed.emit(k, symbols)
        await until(lambda: at(k) in self.done)
