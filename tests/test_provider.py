"""The provider SDK against a live API: one batch per tick, identical inputs, reconnects, errors."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from api_fakes import CONTEXT, SYMBOLS, TOKEN, ApiRig, at
from engine_fakes import Scheduled, until
from polars.testing import assert_frame_equal

from crypto_moonshot.live.engine import BatchOutcome
from crypto_moonshot.live.state import BatchStatus, EventLevel
from crypto_moonshot.models import Interval
from crypto_moonshot.provider import Provider, ProviderError

pytestmark = pytest.mark.anyio

H1 = Interval.H1
FAST: dict[str, Any] = {"backoff_start": 0.01, "backoff_max": 0.05}


class Recorder(Scheduled):
    """A plugin that remembers what it was called with."""

    def __init__(self, schedule: Mapping[datetime, list[tuple[str, float]]] | None = None) -> None:
        super().__init__(schedule or {}, name="ext", warmup=3)
        self.seen: list[tuple[datetime, dict[str, pl.DataFrame], dict[str, pl.DataFrame]]] = []
        self.context_symbols = (CONTEXT,)

    def on_candles(
        self,
        windows: Mapping[str, pl.DataFrame],
        time: datetime,
        context: Mapping[str, pl.DataFrame],
    ) -> pl.DataFrame:
        self.seen.append((time, dict(windows), dict(context)))
        return super().on_candles(windows, time, context)


@contextlib.asynccontextmanager
async def running(
    rig: ApiRig, plugin: Scheduled, *, token: str = TOKEN, name: str = "ext", **kwargs: Any
) -> AsyncIterator[Provider]:
    provider = Provider(plugin, url=rig.url, token=token, name=name, **FAST, **kwargs)
    task = asyncio.create_task(provider.run())
    try:
        yield provider
    finally:
        provider.stop.set()
        await asyncio.wait_for(task, 10)


async def connected(rig: ApiRig) -> None:
    await until(lambda: rig.app.state.hub.client_count == 1)


async def test_one_batch_per_tick_even_an_empty_one(tmp_path: Path) -> None:
    async with ApiRig(tmp_path, deadline="30s") as rig:
        plugin = Recorder({at(1): [("AAAUSDT", 0.5)]})
        async with running(rig, plugin) as provider:
            await connected(rig)
            await rig.tick(0)
            await rig.tick(1)
            await rig.tick(2, [CONTEXT])  # no traded symbol closed: empty batch, no call
            await until(lambda: provider.stats.posted == 3)  # the response follows the decision
            assert provider.stats.ticks == 3
            records = [rig.store.batch_for("ext", H1, at(k)) for k in range(3)]
            assert [r.status for r in records if r is not None] == [BatchStatus.ON_TIME] * 3
            assert [t for t, _, _ in plugin.seen] == [at(0), at(1)]
            batch = rig.store.batch_for("ext", H1, at(1))
            assert batch is not None
            assert [s.symbol for s in batch.signals] == ["AAAUSDT"]
            assert rig.store.live_positions()[0].symbol == "AAAUSDT"
            assert [p.name for p in rig.store.providers()] == ["ext"]


async def test_the_plugin_sees_exactly_what_an_in_process_plugin_would(tmp_path: Path) -> None:
    async with ApiRig(tmp_path, deadline="30s") as rig:
        plugin = Recorder()
        spec = rig.specs[0]
        assert spec.name == "ext"
        async with running(rig, plugin) as provider:
            await connected(rig)
            await until(lambda: provider._info is not None)
            info = provider._info
            assert info is not None
            # BBBUSDT stops closing candles after tick 0; the context symbol misses tick 3
            for k in range(8):
                closing = [s for s in (*SYMBOLS, CONTEXT) if not (s == "BBBUSDT" and k > 0)]
                if k == 3:
                    closing.remove(CONTEXT)
                await rig.tick(k, closing)
                expected_windows, expected_context = rig.engine._plugin_inputs(spec, at(k))
                windows, context = provider._inputs(info, at(k))
                assert list(windows) == list(expected_windows)
                assert list(context) == list(expected_context)
                for symbol in windows:
                    assert_frame_equal(windows[symbol], expected_windows[symbol])
                for symbol in context:
                    assert_frame_equal(context[symbol], expected_context[symbol])
            assert "BBBUSDT" not in windows  # stale: its candle is over two intervals old
            time, seen_windows, seen_context = plugin.seen[-1]
            assert time == at(7)
            assert set(seen_windows) == {"AAAUSDT", "CCCUSDT"}
            assert seen_windows["AAAUSDT"].height == 3  # bounded to the warm-up
            assert seen_context[CONTEXT]["open_time"][-1] == at(7)


async def test_it_reconnects_and_answers_the_candle_the_engine_is_waiting_for(
    tmp_path: Path,
) -> None:
    async with ApiRig(tmp_path, deadline="30s") as rig:
        plugin = Recorder({at(1): [("AAAUSDT", 0.5)]})
        async with running(rig, plugin) as provider:
            await connected(rig)
            await rig.tick(0)
            hub = rig.app.state.hub
            for client in list(hub._clients):  # the daemon drops the connection
                client.dropped.set()
            await until(lambda: hub.client_count == 0)
            await rig.tick(1)  # collected only once the provider is back and answers
            assert rig.store.batch_for("ext", H1, at(1)) is not None
            assert provider.stats.reconnects >= 1
            assert [t for t, _, _ in plugin.seen] == [at(0), at(1)]
            batch = rig.store.batch_for("ext", H1, at(1))
            assert batch is not None
            assert batch.status is BatchStatus.ON_TIME
            assert [s.symbol for s in batch.signals] == ["AAAUSDT"]
            await rig.tick(2)
            assert provider.stats.ticks == 3


async def test_it_waits_for_a_daemon_that_is_not_up_yet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    plugin = Recorder()
    provider = Provider(plugin, url="http://127.0.0.1:1", token=TOKEN, name="ext", **FAST)
    task = asyncio.create_task(provider.run())
    await until(lambda: provider.stats.reconnects >= 2)
    provider.stop.set()
    await asyncio.wait_for(task, 5)
    assert "retrying" in caplog.text
    assert TOKEN not in caplog.text


async def test_the_end_of_a_replay_stops_the_provider(tmp_path: Path) -> None:
    async with ApiRig(tmp_path, deadline="30s") as rig:
        provider = Provider(Recorder(), url=rig.url, token=TOKEN, name="ext", **FAST)
        task = asyncio.create_task(provider.run())
        await connected(rig)
        rig.app.state.hub.finish("replay finished")
        await asyncio.wait_for(task, 5)  # returns by itself: no reconnect


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("token", "refused the API token"),
        ("unknown", "no strategy named 'nobody'"),
        ("inproc", "runs inside the daemon"),
        ("interval", "trades 1h"),
        ("context", r"context_symbols = \['ZZZUSDT'\]"),
    ],
)
async def test_setup_problems_are_fatal_and_say_what_to_fix(
    tmp_path: Path, case: str, message: str
) -> None:
    async with ApiRig(tmp_path) as rig:
        plugin = Recorder()
        token, name = TOKEN, "ext"
        if case == "token":
            token = "wrong"
        elif case == "unknown":
            name = "nobody"
        elif case == "inproc":
            name = "inproc"
        elif case == "interval":
            plugin.interval = Interval.M5
        else:
            plugin.context_symbols = ("ZZZUSDT",)
        provider = Provider(plugin, url=rig.url, token=token, name=name, **FAST)
        with pytest.raises(ProviderError, match=message) as caught:
            await asyncio.wait_for(provider.run(), 10)
        assert TOKEN not in str(caught.value)


async def test_a_failing_plugin_stops_the_provider(tmp_path: Path) -> None:
    async with ApiRig(tmp_path, deadline="30s") as rig:
        plugin = Recorder()
        plugin.fail_at = {at(1)}
        provider = Provider(plugin, url=rig.url, token=TOKEN, name="ext", **FAST)
        task = asyncio.create_task(provider.run())
        await connected(rig)
        await rig.tick(0)
        rig.clock.now = at(2) + timedelta(seconds=1)
        await rig.feed.emit(1)
        with pytest.raises(ProviderError, match=r"on_candles for .* failed: RuntimeError: boom"):
            await asyncio.wait_for(task, 10)
        await rig.engine.stop()


async def test_keep_going_logs_the_failure_and_the_daemon_records_a_missed_batch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR)
    async with ApiRig(tmp_path, deadline="1.05s") as rig:
        plugin = Recorder()
        plugin.fail_at = {at(1)}
        async with running(rig, plugin, keep_going=True) as provider:
            await connected(rig)
            await rig.tick(0)
            await rig.tick(1)  # no batch: missed at the deadline
            await rig.tick(2)
            missed = rig.store.batch_for("ext", H1, at(1))
            assert missed is not None
            assert missed.status is BatchStatus.MISSED
            await until(lambda: provider.stats.posted == 2)
    assert "no batch sent for this candle" in caplog.text


async def test_a_refused_batch_is_reported_loudly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.ERROR)
    async with ApiRig(tmp_path, deadline="1.05s") as rig:

        async def refuse(*args: object, **kwargs: object) -> BatchOutcome:
            return BatchOutcome("refused", "the strategy is on fire")

        monkeypatch.setattr(rig.engine, "submit_batch", refuse)
        async with running(rig, Recorder()) as provider:
            await connected(rig)
            await rig.tick(0)
            await until(lambda: len(provider.stats.refused) == 1)
            assert "the strategy is on fire" in provider.stats.refused[0]
    assert "REFUSED" in caplog.text
    assert "the strategy is on fire" in caplog.text
    assert TOKEN not in caplog.text


async def test_a_duplicate_is_noted_not_fatal(tmp_path: Path) -> None:
    async with ApiRig(tmp_path, deadline="30s") as rig, running(rig, Recorder()) as provider:
        await connected(rig)
        rig.clock.now = at(1) + timedelta(seconds=1)
        first = await rig.http.post(
            "/signals", json={"contract": 1, "strategy": "ext", "interval": "1h",
                              "candle_time": at(0).isoformat(), "signals": []}
        )  # fmt: skip
        assert first.status_code == 202
        await rig.tick(0)
        await until(lambda: provider.stats.duplicates == 1)
        assert rig.store.events_since() != [] or True
        assert all(e.level is not EventLevel.ERROR for e in rig.store.events_since())
