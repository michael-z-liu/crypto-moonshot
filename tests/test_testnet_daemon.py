"""Testnet mode end to end: the daemon trading through `BinanceBroker` on the fake exchange.

The same candle path drives a paper daemon and a testnet daemon (market data through the fake
stream server; for the testnet also the fake exchange's prices, walked open -> low/high -> close
before each candle's close message). They must enter the same symbols on the same ticks and
leave for the same reasons. Expected differences, asserted below: the exchange fills at its
price (not the candle's open plus modelled slippage), sells whole lot steps (the remainder of a
buy's base-asset commission stays behind as dust) and pays commissions in the base asset.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from binance_fakes import REST_URL, Clock, fast_sleep
from fake_exchange import API_KEY, API_SECRET, FakeExchange
from kline_fakes import FakeBinance
from pydantic import SecretStr
from stream_fakes import FakeStreamServer, until

from crypto_moonshot.config import Settings
from crypto_moonshot.data.stream import StreamConfig
from crypto_moonshot.live.config import RunConfig, config_from_dict
from crypto_moonshot.live.daemon import DaemonError, run_daemon
from crypto_moonshot.live.state import EventLevel, PositionRecord, StateStore
from crypto_moonshot.live.testnet import ExchangeOptions
from crypto_moonshot.models import EPOCH, Interval

pytestmark = pytest.mark.anyio

M5 = Interval.M5
T0 = datetime(2024, 3, 1, tzinfo=UTC)
TOKEN = "tkn-testnet-9Zq4Lw7RbXe"
ENTER_TICK = 2
LAST_TICK = 9

# strategy -> (symbol, exit plan source)
STRATEGIES = {
    "tp": ("AAAUSDT", "ExitPlan(take_profit=0.05, stop=0.10, max_hold=timedelta(hours=1))"),
    "stop": ("BBBUSDT", "ExitPlan(take_profit=0.20, stop=0.05, max_hold=timedelta(hours=1))"),
    "time": ("CCCUSDT", "ExitPlan(take_profit=0.50, stop=0.50, max_hold=timedelta(minutes=15))"),
}
PLUGIN = """
from datetime import datetime, timedelta, timezone

import polars as pl

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.models import EPOCH, Interval

ENTER_AT = datetime(2024, 3, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * {tick})


class Enter:
    interval = Interval.M5
    warmup = 3
    context_symbols = ()
    exit = {plan}

    def on_candles(self, windows, time, context):
        rows = [{symbol!r}] if time == ENTER_AT else []
        return pl.DataFrame(
            {{"symbol": pl.Series(rows, dtype=pl.String),
              "priority": pl.Series([0.0] * len(rows), dtype=pl.Float64)}}
        )
"""


def at(k: int) -> datetime:
    return T0 + k * M5.length


def candle(k: int, symbol: str) -> tuple[float, float, float, float]:
    """(open, high, low, close) of candle `k`: continuous, flat except where the symbol exits."""
    if symbol == "AAAUSDT":  # take profit: +5% is 105
        if k == ENTER_TICK + 1:
            return 100.0, 106.0, 99.5, 105.5
        return (100.0, 100.5, 99.5, 100.0) if k <= ENTER_TICK else (105.5, 105.5, 105.5, 105.5)
    if symbol == "BBBUSDT":  # stop: -5% is 95
        if k == ENTER_TICK + 1:
            return 100.0, 100.5, 94.0, 95.0
        return (100.0, 100.5, 99.5, 100.0) if k <= ENTER_TICK else (95.0, 95.0, 95.0, 95.0)
    return 100.0, 100.5, 99.5, 100.0  # time exit: flat


def kline(symbol: str, k: int) -> dict[str, Any]:
    o, h, lo, c = candle(k, symbol)
    ms = (at(k) - EPOCH) // timedelta(milliseconds=1)
    close_ms = (at(k + 1) - EPOCH) // timedelta(milliseconds=1) - 1
    return {
        "stream": f"{symbol.lower()}@kline_5m",
        "data": {
            "e": "kline",
            "E": close_ms,
            "s": symbol,
            "k": {
                "t": ms, "T": close_ms, "s": symbol, "i": "5m",
                "o": str(o), "h": str(h), "l": str(lo), "c": str(c),
                "v": "10.5", "n": 42, "x": True, "q": "1050.25", "V": "5.25", "Q": "525.125",
                "B": "0",
            },
        },
    }  # fmt: skip


def make_config(
    tmp_path: Path,
    mode: str,
    *,
    strategies: dict[str, tuple[Any, ...]] | None = None,
    capital: float = 1000.0,
    symbols: dict[str, list[str]] | None = None,
) -> RunConfig:
    """A config of the three strategies (`strategies` replaces them: name -> (symbol, plan))."""
    wired = strategies or STRATEGIES
    account: dict[str, Any] = {"costs": {"fee": 0.001, "slippage": 0.0, "latency": 0.0}}
    if mode == "paper":
        account["initial_cash"] = 10_000.0
    entries = []
    for name, (symbol, plan) in wired.items():
        path = tmp_path / f"{name}.py"
        path.write_text(PLUGIN.format(tick=ENTER_TICK, plan=plan, symbol=symbol))
        entries.append(
            {
                "name": name,
                "source": f"{path}:Enter",
                "symbols": (symbols or {}).get(name, [symbol]),
                "capital": capital,
                "sizing": {"stake": 100.0},
            }
        )
    return config_from_dict(
        {
            "mode": mode,
            "state": str(tmp_path / "state" / f"{mode}.sqlite"),
            "account": account,
            "engine": {"drain_interval": "1s", "entry_grace": "30m"},
            "strategies": entries,
        }
    )


@pytest.fixture
async def server() -> AsyncIterator[FakeStreamServer]:
    async with FakeStreamServer() as fake:
        yield fake


@dataclass
class Rig:
    clock: Clock
    fake: FakeExchange
    settings: Settings
    options: ExchangeOptions
    ws_url: str

    async def close(self) -> None:
        if self.options.http is not None:
            await self.options.http.aclose()
        await self.fake.stop_ws()


async def make_rig(
    server: FakeStreamServer, tmp_path: Path, *, usdt: int = 10_000, **settings: Any
) -> Rig:
    clock = Clock(at(1) + timedelta(seconds=1))
    fake = FakeExchange(clock, usdt=Decimal(usdt))
    ws_url = await fake.start_ws()
    values: dict[str, Any] = {
        "api_token": SecretStr(TOKEN),
        "stream_url": server.url,
        "data_dir": tmp_path / "no-data",
        "binance_testnet_api_key": SecretStr(API_KEY),
        "binance_testnet_api_secret": SecretStr(API_SECRET),
        "binance_testnet_rest_url": REST_URL,
        "binance_testnet_ws_api_url": ws_url,
    }
    values.update(settings)
    options = ExchangeOptions(
        http=httpx.AsyncClient(transport=httpx.MockTransport(fake)),
        client={"sleep": fast_sleep, "backoff_base": 0.0},
        stream={"sleep": fast_sleep, "backoff_base": 0.005, "backoff_max": 0.02},
        broker={"poll_interval_down": timedelta(0)},
    )
    return Rig(clock, fake, Settings(**values), options, ws_url)


@dataclass
class Daemon:
    task: asyncio.Task[None]
    stop: asyncio.Event
    port: asyncio.Future[int]

    async def api(self) -> httpx.AsyncClient:
        waiting: set[asyncio.Future[Any]] = {self.port, self.task}
        await asyncio.wait(waiting, timeout=20, return_when=asyncio.FIRST_COMPLETED)
        if self.task.done():
            self.task.result()  # the daemon failed to start: show why
        port = self.port.result()
        return httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}/api/v1",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )


def start(config: RunConfig, server: FakeStreamServer, rig: Rig) -> Daemon:
    symbols = sorted({sym for sc in config.strategies for sym in sc.symbols})
    rest = FakeBinance({s: [at(k) for k in range(-3, 1)] for s in symbols})
    stop = asyncio.Event()
    port: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(
        run_daemon(
            config,
            rig.settings,
            port=0,
            client=rest,
            stream_config=StreamConfig(
                base_url=server.url, subscribe_gap=0.0, backoff_base=0.01, backoff_max=0.05
            ),
            clock=rig.clock,
            stop=stop,
            install_signals=False,
            on_ready=port.set_result,
            exchange_options=rig.options if config.mode == "testnet" else None,
        )
    )
    return Daemon(task, stop, port)


async def settle(api: httpx.AsyncClient, rig: Rig, *, exchange: bool) -> dict[str, Any]:
    """Wait until the engine's view stops changing: nothing pending, every open position
    protected, and the same for a few polls in a row."""
    if exchange:
        await rig.fake.flushed()
    last: str | None = None
    stable = 0
    async with asyncio.timeout(15):
        while True:
            status: dict[str, Any] = (await api.get("/status")).json()
            positions = status["positions"]
            ready = not exchange or all(
                p["status"] == "open" and p["bracket_id"] for p in positions
            )
            signature = json.dumps([positions, status["open_orders"]], sort_keys=True)
            stable = stable + 1 if ready and signature == last else 0
            last = signature
            if stable >= 4:
                return status
            await asyncio.sleep(0.05)


async def drive(
    config: RunConfig, server: FakeStreamServer, rig: Rig
) -> tuple[Daemon, dict[str, Any]]:
    """Start a daemon, replay candles 1..LAST_TICK and return it with its last status."""
    daemon = start(config, server, rig)
    testnet = config.mode == "testnet"
    symbols = sorted({s for sc in config.strategies for s in sc.symbols})
    async with await daemon.api() as api:
        await until(lambda: server.connections != [] and server.subscribed(0) != [])
        status: dict[str, Any] = {}
        for k in range(1, LAST_TICK + 1):
            if testnet:
                for symbol in symbols:
                    o, h, lo, c = candle(k, symbol)
                    rig.fake.walk(symbol, o, h, lo, c)
            rig.clock.now = at(k + 1) + timedelta(seconds=1)
            for symbol in symbols:
                await server.send(kline(symbol, k))
            async with asyncio.timeout(15):
                while (await api.get("/status")).json()["last_ticks"].get("5m") != at(
                    k
                ).isoformat():
                    await asyncio.sleep(0.02)
            status = await settle(api, rig, exchange=testnet)
    return daemon, status


async def shut_down(daemon: Daemon) -> None:
    daemon.stop.set()
    await asyncio.wait_for(daemon.task, 20)
    assert daemon.task.exception() is None


def trades(config: RunConfig) -> list[PositionRecord]:
    with StateStore.open(config.state) as store:
        return sorted(store.closed_positions(), key=lambda p: (p.strategy, p.entry_time or T0))


# -- end to end ---------------------------------------------------------------------------------


async def test_a_testnet_run_trades_like_a_paper_run_on_the_same_candles(
    tmp_path: Path, server: FakeStreamServer, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")
    paper_dir, testnet_dir = tmp_path / "paper", tmp_path / "testnet"
    paper_dir.mkdir()
    testnet_dir.mkdir()

    paper_config = make_config(paper_dir, "paper")
    paper_rig = await make_rig(server, paper_dir)
    paper, _ = await drive(paper_config, server, paper_rig)
    await shut_down(paper)
    await paper_rig.close()

    async with FakeStreamServer() as server2:
        config = make_config(testnet_dir, "testnet")
        rig = await make_rig(server2, testnet_dir)
        daemon, status = await drive(config, server2, rig)
        assert status["mode"] == "testnet"
        exchange = status["exchange"]
        assert exchange["stream_connected"] is True
        assert exchange["halted"] is None
        assert exchange["last_reconcile"]["clean"] is True
        events = [e.kind for e in StateStore.open(config.state).events_since()]
        assert "daemon_started" in events
        health = (await (await daemon.api()).get("/health")).json()
        assert health["mode"] == "testnet"
        await shut_down(daemon)

    expected = {"tp": "take_profit", "stop": "stop", "time": "time"}
    paper_trades, testnet_trades = trades(paper_config), trades(config)
    assert [(t.strategy, t.symbol, t.entry_time) for t in testnet_trades] == [
        (t.strategy, t.symbol, t.entry_time) for t in paper_trades
    ]
    assert {t.strategy: t.exit_reason for t in testnet_trades} == expected
    assert {t.strategy: t.exit_reason for t in paper_trades} == expected
    assert all(t.entry_time == at(ENTER_TICK + 1) for t in testnet_trades)

    # the expected differences (see the module docstring)
    for paper_trade, trade in zip(paper_trades, testnet_trades, strict=True):
        assert trade.entry_price == pytest.approx(100.0)  # the exchange fills at its price
        assert paper_trade.entry_price == pytest.approx(100.0)  # (no slippage configured)
        assert trade.quantity is not None
        assert (Decimal(repr(trade.quantity)) % Decimal("0.001")) == 0  # whole lot steps
        assert trade.pnl is not None
        assert paper_trade.pnl is not None
        assert trade.pnl != pytest.approx(paper_trade.pnl, abs=1e-9)  # commissions differ
    by_strategy = {t.strategy: t for t in testnet_trades}
    assert by_strategy["tp"].exit_price == pytest.approx(105.0, abs=0.01)
    # a stop triggers a market sell: it fills at the market, not at the stop level (paper: 95)
    assert {t.strategy: t for t in paper_trades}["stop"].exit_price == pytest.approx(95.0)
    assert by_strategy["stop"].exit_price == pytest.approx(94.0, abs=0.01)
    # the exchange never saw a market order from the engine other than entries and the time exit
    assert rig.fake.balance("USDT")[1] == 0
    await rig.close()

    # no key, secret or token anywhere the daemon wrote
    for secret in (API_KEY, API_SECRET, TOKEN):
        assert secret not in caplog.text
        with StateStore.open(config.state) as store:
            assert all(secret not in repr(e) for e in store.events_since())


async def test_the_exchange_state_is_in_status_and_the_secrets_are_not(
    tmp_path: Path, server: FakeStreamServer
) -> None:
    config = make_config(tmp_path, "testnet")
    rig = await make_rig(server, tmp_path)
    daemon = start(config, server, rig)
    async with await daemon.api() as api:
        await until(lambda: rig.fake.subscribed > 0)
        body = (await api.get("/status")).text
        status = json.loads(body)
        assert status["mode"] == "testnet"
        assert set(status["exchange"]) >= {
            "stream_connected",
            "halted",
            "dust",
            "other_commissions",
            "last_reconcile",
        }
        assert status["exchange"]["last_reconcile"] is not None
        for secret in (API_KEY, API_SECRET, TOKEN):
            assert secret not in body
    await shut_down(daemon)
    await rig.close()


# -- start-up checks and refusals ---------------------------------------------------------------


async def refused(config: RunConfig, server: FakeStreamServer, rig: Rig) -> DaemonError:
    with pytest.raises(DaemonError) as caught:
        await run_daemon(
            config,
            rig.settings,
            port=0,
            client=FakeBinance(),
            stream_config=StreamConfig(base_url=server.url, subscribe_gap=0.0),
            clock=rig.clock,
            install_signals=False,
            exchange_options=rig.options,
        )
    return caught.value


async def test_it_refuses_to_start_without_keys(tmp_path: Path, server: FakeStreamServer) -> None:
    rig = await make_rig(
        server, tmp_path, binance_testnet_api_key=None, binance_testnet_api_secret=None
    )
    error = await refused(make_config(tmp_path, "testnet"), server, rig)
    assert error.usage
    assert "MOONSHOT_BINANCE_TESTNET_API_KEY" in str(error)
    assert "MOONSHOT_BINANCE_TESTNET_API_SECRET" in str(error)
    assert not (tmp_path / "state").exists()  # nothing was opened
    assert rig.fake.requests == []
    await rig.close()


async def test_keys_under_the_wrong_variable_names_do_not_count(
    tmp_path: Path, server: FakeStreamServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_BINANCE_API_KEY", API_KEY)
    monkeypatch.setenv("MOONSHOT_BINANCE_API_SECRET", API_SECRET)
    monkeypatch.setenv("MOONSHOT_BINANCE_TESTNET_KEY", API_KEY)
    monkeypatch.setenv("MOONSHOT_API_TOKEN", TOKEN)
    rig = await make_rig(server, tmp_path)
    rig.settings = Settings(stream_url=server.url, data_dir=tmp_path / "no-data")
    error = await refused(make_config(tmp_path, "testnet"), server, rig)
    assert error.usage
    assert "MOONSHOT_BINANCE_TESTNET_API_KEY" in str(error)
    assert API_KEY not in str(error)
    await rig.close()


async def test_an_unsupported_exit_plan_is_a_usage_error(
    tmp_path: Path, server: FakeStreamServer
) -> None:
    rig = await make_rig(server, tmp_path)
    for plan, text in (
        ("ExitPlan(stop=0.02, trailing=0.05)", "tighter than trailing"),
        ("ExitPlan(take_profit=0.05, tp_fill='close')", "tp_fill"),
    ):
        config = make_config(tmp_path, "testnet", strategies={"bad": ("AAAUSDT", (plan))})
        error = await refused(config, server, rig)
        assert error.usage
        assert "strategy bad" in str(error)
        assert text in str(error)
    assert not (tmp_path / "state").exists()
    await rig.close()


async def test_insufficient_usdt_is_refused_and_everything_is_closed(
    tmp_path: Path, server: FakeStreamServer
) -> None:
    rig = await make_rig(server, tmp_path, usdt=2_000)  # the books need 3 x 1000
    config = make_config(tmp_path, "testnet")
    error = await refused(config, server, rig)
    assert error.usage
    assert "not enough free USDT" in str(error)
    assert "2000" in str(error)
    assert {r.method for r in rig.fake.requests} == {"GET"}  # nothing was ordered
    with StateStore.open(config.state) as store:  # the state file is not locked any more
        store.set_meta("x", "1")
    await rig.close()


@pytest.mark.parametrize(
    "update",
    [
        {"binance_testnet_rest_url": "https://api.binance.com"},
        {"binance_testnet_ws_api_url": "wss://ws-api.binance.com/ws-api/v3"},
        {"binance_testnet_rest_url": "https://data-api.binance.vision"},
        {"binance_testnet_rest_url": "https://api.binance.us"},
        {"binance_testnet_rest_url": "https://testnet.binance.vision@api.binance.com"},
        {"binance_testnet_rest_url": "https://testnet.binance.vision.evil.example"},
    ],
)
async def test_a_mainnet_url_is_refused_before_anything_happens(
    tmp_path: Path, server: FakeStreamServer, update: dict[str, str]
) -> None:
    rig = await make_rig(server, tmp_path)
    rig.settings = rig.settings.model_copy(update=update)
    error = await refused(make_config(tmp_path, "testnet"), server, rig)
    assert error.usage
    assert "testnet.binance.vision" in str(error)
    assert rig.fake.requests == []
    assert not (tmp_path / "state").exists()
    await rig.close()


async def test_the_testnet_hosts_pass_the_guard() -> None:
    from crypto_moonshot.exchange.guard import require_testnet_url

    for url in (
        "https://testnet.binance.vision",
        "wss://ws-api.testnet.binance.vision/ws-api/v3",
        "ws://127.0.0.1:9999",
    ):
        require_testnet_url(url)


async def test_symbols_not_on_the_exchange_are_dropped_with_a_warning(
    tmp_path: Path, server: FakeStreamServer
) -> None:
    config = make_config(
        tmp_path, "testnet", symbols={"tp": ["AAAUSDT", "ZZZUSDT"], "stop": ["BBBUSDT"]}
    )
    rig = await make_rig(server, tmp_path)
    daemon = start(config, server, rig)
    async with await daemon.api() as api:
        await until(lambda: rig.fake.subscribed > 0)
        await until(lambda: server.connections != [] and server.subscribed(0) != [])
        assert "zzzusdt@kline_5m" not in server.subscribed(0)
        assert "aaausdt@kline_5m" in server.subscribed(0)
        events = (await api.get("/events")).json()
    warnings = [e for e in events["events"] if e["kind"] == "symbols_dropped"]
    assert len(warnings) == 1
    assert warnings[0]["level"] == EventLevel.WARNING.value
    assert "ZZZUSDT" in warnings[0]["message"]
    await shut_down(daemon)
    await rig.close()


async def test_a_strategy_left_without_symbols_refuses_the_start(
    tmp_path: Path, server: FakeStreamServer
) -> None:
    config = make_config(tmp_path, "testnet", symbols={"tp": ["ZZZUSDT", "YYYUSDT"]})
    rig = await make_rig(server, tmp_path)
    error = await refused(config, server, rig)
    assert error.usage
    assert "strategy tp" in str(error)
    assert "none of its symbols" in str(error)
    await rig.close()


async def test_a_delisted_symbol_is_dropped_too(tmp_path: Path, server: FakeStreamServer) -> None:
    config = make_config(
        tmp_path,
        "testnet",
        strategies={"tp": STRATEGIES["tp"]},
        symbols={"tp": ["AAAUSDT", "BBBUSDT"]},
    )
    rig = await make_rig(server, tmp_path)
    for s in rig.fake.symbols:
        if s["symbol"] == "BBBUSDT":
            s["status"] = "BREAK"
    daemon = start(config, server, rig)
    await until(lambda: server.connections != [] and server.subscribed(0) != [])
    assert "bbbusdt@kline_5m" not in server.subscribed(0)
    await shut_down(daemon)
    await rig.close()
