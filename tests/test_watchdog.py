"""The watchdogs: a stalled engine and a blocked event loop."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest
from test_health import Rig, at

from crypto_moonshot.live.watchdog import EXIT_WATCHDOG, LoopWatchdog, StallWatchdog


def stall(rig: Rig) -> StallWatchdog:
    return StallWatchdog(rig.engine, rig.feed, rig.config.health)


def test_the_default_threshold_is_ten_intervals_but_at_least_fifteen_minutes(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    assert stall(rig).threshold == timedelta(hours=10)  # 10 x 1h
    (tmp_path / "x").mkdir()
    custom = Rig(tmp_path / "x", {"watchdog_after": "20m"})
    assert stall(custom).threshold == timedelta(minutes=20)
    health = rig.config.health
    assert health.watchdog_threshold(timedelta(minutes=1)) == timedelta(minutes=15)
    assert health.watchdog_threshold(timedelta(minutes=5)) == timedelta(minutes=50)


def test_a_stall_with_a_live_feed_is_reported(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"watchdog_after": "20m"})
    dog = stall(rig)
    rig.clock.now = at(0) + timedelta(minutes=20)
    rig.feed.last_message_at = rig.clock.now
    assert dog.check() is None  # exactly the threshold
    rig.clock.now += timedelta(seconds=1)
    reason = dog.check()
    assert reason is not None
    assert "no tick was processed for 1201s" in reason
    assert "limit 1200s" in reason


def test_a_processed_tick_resets_the_clock(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"watchdog_after": "20m"})
    dog = stall(rig)
    rig.clock.now = at(1)
    rig.engine._last_tick_at = rig.clock.now  # a tick was decided now
    rig.clock.now = at(1) + timedelta(minutes=19)
    rig.feed.last_message_at = rig.clock.now
    assert dog.check() is None
    rig.clock.now = at(1) + timedelta(minutes=21)
    rig.feed.last_message_at = rig.clock.now
    assert dog.check() is not None


def test_a_silent_or_down_feed_is_not_the_engines_fault(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"watchdog_after": "20m", "feed_grace": "60s"})
    dog = stall(rig)
    rig.clock.now = at(5)
    rig.feed.last_message_at = None  # never heard anything
    assert dog.check() is None
    rig.feed.last_message_at = rig.clock.now - timedelta(seconds=61)  # silent for a while
    assert dog.check() is None
    rig.feed.last_message_at = rig.clock.now - timedelta(seconds=59)
    assert dog.check() is not None  # messages are flowing, ticks are not


def test_it_stays_quiet_in_normal_operation(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"watchdog_after": "20m"})
    dog = stall(rig)
    for minute in range(0, 120, 10):
        rig.clock.now = at(0) + timedelta(minutes=minute)
        rig.feed.last_message_at = rig.clock.now
        rig.engine._last_tick_at = rig.clock.now  # a tick every ten minutes
        assert dog.check() is None


# -- the event loop watchdog ------------------------------------------------------------------


@pytest.mark.anyio
async def test_a_blocked_loop_calls_the_exit_function() -> None:
    exits: list[int] = []
    fired = threading.Event()

    def fake_exit(code: int) -> None:
        exits.append(code)
        fired.set()

    dog = LoopWatchdog(0.3, exit_fn=fake_exit, poll=0.02)
    task = asyncio.create_task(dog.run())
    await asyncio.sleep(0.1)  # running: it beats
    assert exits == []
    time.sleep(0.8)  # block the loop (this is the failure being detected)
    assert fired.wait(5)
    assert exits == [EXIT_WATCHDOG] == [3]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_a_loop_that_keeps_running_is_left_alone() -> None:
    exits: list[int] = []
    dog = LoopWatchdog(0.5, exit_fn=exits.append, poll=0.02)
    task = asyncio.create_task(dog.run())
    for _ in range(30):
        await asyncio.sleep(0.02)
        time.sleep(0.002)  # short blocking steps are fine
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert exits == []
    assert not dog.fired.is_set()


@pytest.mark.anyio
async def test_the_thread_stops_with_the_task() -> None:
    exits: list[int] = []
    dog = LoopWatchdog(0.2, exit_fn=exits.append, poll=0.02)
    task = asyncio.create_task(dog.run())
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0.4)  # beyond the threshold, but the watcher is gone
    assert exits == []
    assert all(t.name != "loop-watchdog" for t in threading.enumerate())


@pytest.mark.anyio
async def test_the_exit_is_logged_before_exiting(caplog: pytest.LogCaptureFixture) -> None:
    fired = threading.Event()
    dog = LoopWatchdog(0.2, exit_fn=lambda code: fired.set(), poll=0.02)
    task = asyncio.create_task(dog.run())
    await asyncio.sleep(0.05)
    time.sleep(0.5)
    assert fired.wait(5)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert "event loop has been blocked" in caplog.text
    assert "exiting with code 3" in caplog.text


def test_a_non_positive_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        LoopWatchdog(0)
