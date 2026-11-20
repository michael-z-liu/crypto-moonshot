"""The daemon's operations: lock, health rules, watchdogs, backups, over a fake Binance."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from engine_fakes import until as wait_for
from replay_fakes import END, START, rule_strategy, run_config
from replay_fakes import at as hour
from stream_fakes import FakeStreamServer, kline_message
from test_daemon import M5, Daemon, at, close_candle, make_config, make_settings

from crypto_moonshot.config import Settings
from crypto_moonshot.live.backup import check_backup, list_backups
from crypto_moonshot.live.config import HealthConfig
from crypto_moonshot.live.daemon import DaemonError, run_daemon, serve_replay
from crypto_moonshot.live.health import GB
from crypto_moonshot.live.lock import StateLock, StateLockedError
from crypto_moonshot.live.replay import prepare_replay
from crypto_moonshot.live.state import SCHEMA_VERSION, StateStore

pytestmark = pytest.mark.anyio

FAST = {"health_every": "0.05s"}


@pytest.fixture
async def server() -> AsyncIterator[FakeStreamServer]:
    async with FakeStreamServer() as fake:
        yield fake


def ops_config(tmp_path: Path, **health: Any) -> Any:
    return make_config(tmp_path, health={**FAST, "feed_grace": "60s", **health})


def events(path: Path) -> list[tuple[str, str]]:
    with StateStore.open_readonly(path) as store:
        return [(e.kind, e.message) for e in store.events_since()]


async def active_alerts(api: Any) -> dict[str, dict[str, Any]]:
    body = (await api.get("/alerts")).json()
    return {a["key"]: a for a in body["active"]}


async def alerts_until(api: Any, check: Any, timeout: float = 10.0) -> dict[str, dict[str, Any]]:
    async with asyncio.timeout(timeout):
        while True:
            found = await active_alerts(api)
            if check(found):
                return found
            await asyncio.sleep(0.02)


async def started(daemon: Daemon, server: FakeStreamServer) -> Any:
    api = await daemon.api()
    await wait_for(lambda: server.connections != [] and server.subscribed(0) != [])
    return api


async def tick(daemon: Daemon, server: FakeStreamServer, api: Any, k: int) -> None:
    daemon.clock.now = at(k) + M5.length + timedelta(seconds=1)
    await close_candle(server, k)
    async with asyncio.timeout(10):
        while (await api.get("/status")).json()["last_ticks"].get("5m") != at(k).isoformat():
            await asyncio.sleep(0.02)


# -- the single-instance lock ---------------------------------------------------------------------


async def test_a_second_daemon_on_the_same_state_file_is_refused(
    tmp_path: Path, server: FakeStreamServer
) -> None:
    config = ops_config(tmp_path)
    settings = make_settings(server, tmp_path)
    first = Daemon(config, settings)
    api = await started(first, server)
    with pytest.raises(StateLockedError, match="in use by another moonshot process"):
        await run_daemon(config, settings, port=0, install_signals=False)
    assert (await api.get("/health")).status_code == 200  # the first one is untouched
    await api.aclose()
    first.stop.set()
    await asyncio.wait_for(first.task, 20)
    # released on exit: a new daemon starts
    again = Daemon(config, settings)
    await asyncio.wait_for(again.port, 20)
    again.stop.set()
    await asyncio.wait_for(again.task, 20)


async def test_a_replay_and_a_restore_are_refused_while_the_daemon_runs(
    tmp_path: Path, server: FakeStreamServer, replay_root: Path
) -> None:
    from crypto_moonshot.live.backup import make_backup, restore_backup

    config = ops_config(tmp_path)
    daemon = Daemon(config, make_settings(server, tmp_path))
    api = await started(daemon, server)
    backup = make_backup(config.state)  # a backup works with the daemon running
    assert check_backup(backup.path) == SCHEMA_VERSION
    with pytest.raises(StateLockedError):
        restore_backup(backup.path, config.state)
    with pytest.raises(StateLockedError):
        prepare_replay(
            run_config(config.state), hour(START), hour(END), replay_root,
            plugins={"cap": rule_strategy()},
        )  # fmt: skip
    await api.aclose()
    daemon.stop.set()
    await asyncio.wait_for(daemon.task, 20)
    restore_backup(backup.path, config.state)  # the daemon is gone: allowed


async def test_a_served_replay_holds_the_lock_and_runs_the_rules_that_make_sense(
    tmp_path: Path, replay_root: Path
) -> None:
    path = tmp_path / "state.sqlite"
    config = run_config(path).model_copy(
        update={"health": HealthConfig(health_every=timedelta(seconds=0.05))}
    )
    prepared = prepare_replay(
        config, hour(START), hour(END), replay_root, plugins={"cap": rule_strategy()}
    )
    ready: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    settings = Settings(api_token="tkn-replay-ops")  # type: ignore[arg-type]
    try:
        with pytest.raises(StateLockedError):  # a second replay (or daemon) cannot start
            prepare_replay(config, hour(START), hour(END), replay_root)
        await serve_replay(
            prepared,
            settings,
            port=0,
            install_signals=False,
            on_ready=ready.set_result,
            disk_free=lambda _path: int(0.1 * GB),
        )
        with pytest.raises(StateLockedError):  # still locked until closed
            prepare_replay(config, hour(START), hour(END), replay_root)
    finally:
        prepared.close()
    with StateStore.open(path) as store:
        keys = [a.key for a in store.alerts()]
        assert keys == ["disk_low"]  # the monitor ran in the replay...
        # ...and the wall-clock rules stayed off in simulated time
        fired = {e.data["key"] for e in store.events_since() if e.kind == "alert" and e.data}
        assert fired <= {"disk_low", "stale_symbols:1h"}
    StateLock(path).acquire()


# -- the watchdog ---------------------------------------------------------------------------------


async def test_a_stalled_engine_with_a_live_feed_stops_the_daemon_with_code_3(
    tmp_path: Path, server: FakeStreamServer
) -> None:
    config = ops_config(tmp_path, watchdog_after="30m")
    daemon = Daemon(config, make_settings(server, tmp_path))
    api = await started(daemon, server)
    await tick(daemon, server, api, 1)
    daemon.clock.now += timedelta(minutes=31)  # no tick for 31 minutes...
    await server.send(kline_message("AAAUSDT", M5, at(5), closed=False))  # ...but messages flow
    with pytest.raises(DaemonError, match="the watchdog stopped the daemon") as caught:
        await asyncio.wait_for(daemon.task, 20)
    assert caught.value.code == 3
    assert not caught.value.usage
    await api.aclose()
    kinds = [kind for kind, _ in events(config.state)]
    assert "watchdog" in kinds
    assert kinds.index("watchdog") > kinds.index("daemon_started")
    assert kinds[-1] == "daemon_stopped"  # the graceful path
    with StateStore.open_readonly(config.state) as store:
        (watchdog,) = [e for e in store.events_since() if e.kind == "watchdog"]
        assert watchdog.level.value == "error"
        assert "no tick was processed for" in watchdog.message
        assert store._db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    StateLock(config.state).acquire()  # the lock was released


async def test_the_watchdog_stays_quiet_in_normal_operation(
    tmp_path: Path, server: FakeStreamServer
) -> None:
    config = ops_config(tmp_path, watchdog_after="30m")
    exits: list[int] = []
    daemon = Daemon(config, make_settings(server, tmp_path), loop_exit=exits.append)
    api = await started(daemon, server)
    for k in range(1, 6):
        await tick(daemon, server, api, k)  # a tick every five minutes
        daemon.clock.now += timedelta(minutes=8)  # most of a candle passes before the next tick
        await server.send(kline_message("AAAUSDT", M5, at(k + 1), closed=False))
        await asyncio.sleep(0.2)  # several health checks
    assert not daemon.task.done()
    assert exits == []
    assert (await active_alerts(api)) == {}
    await api.aclose()
    daemon.stop.set()
    await asyncio.wait_for(daemon.task, 20)
    assert "watchdog" not in [kind for kind, _ in events(config.state)]


async def test_a_down_feed_raises_feed_disconnected_but_never_the_watchdog(
    tmp_path: Path,
) -> None:
    config = ops_config(tmp_path, watchdog_after="30m", feed_grace="60s")
    server = await FakeStreamServer().__aenter__()
    daemon = Daemon(config, make_settings(server, tmp_path))
    api = await started(daemon, server)
    await tick(daemon, server, api, 1)
    await server.__aexit__(None, None, None)  # the exchange goes away
    async with asyncio.timeout(10):
        while (await api.get("/health")).json()["feed"]["connected"]:
            await asyncio.sleep(0.02)
    await asyncio.sleep(0.3)  # the monitor has seen it down
    daemon.clock.now += timedelta(minutes=31)  # long beyond the watchdog threshold
    found = await alerts_until(api, lambda a: "feed_disconnected" in a)
    assert found["feed_disconnected"]["level"] == "error"
    await asyncio.sleep(0.3)  # more checks: the watchdog does not fire for a feed that is down
    assert not daemon.task.done()
    assert "watchdog" not in [kind for kind, _ in events(config.state)]
    await api.aclose()
    daemon.stop.set()
    await asyncio.wait_for(daemon.task, 20)


async def test_a_blocked_event_loop_calls_the_exit_function_with_3(
    tmp_path: Path, server: FakeStreamServer
) -> None:
    config = ops_config(tmp_path, loop_blocked_after="0.4s")
    exits: list[int] = []
    daemon = Daemon(config, make_settings(server, tmp_path), loop_exit=exits.append)
    api = await started(daemon, server)
    assert exits == []
    time.sleep(1.0)  # block the loop, as a hung native call would
    await wait_for(lambda: exits != [], 5)
    assert exits[0] == 3
    await api.aclose()
    daemon.stop.set()
    await asyncio.wait_for(daemon.task, 20)


# -- health rules in the daemon -----------------------------------------------------------


async def test_disk_low_and_the_acknowledgement_over_the_running_daemon(
    tmp_path: Path, server: FakeStreamServer
) -> None:
    free = [50 * GB]
    config = ops_config(tmp_path, min_free_gb=1)
    daemon = Daemon(config, make_settings(server, tmp_path), disk_free=lambda _p: free[0])
    api = await started(daemon, server)
    assert await active_alerts(api) == {}
    free[0] = int(0.2 * GB)
    found = await alerts_until(api, lambda a: "disk_low" in a)
    assert found["disk_low"]["acknowledged"] is False
    ack = await api.post("/alerts/disk_low/ack")
    assert ack.status_code == 200
    await asyncio.sleep(0.3)  # still firing: stays acknowledged, no second event
    body = (await api.get("/alerts")).json()
    assert body["unacknowledged"] == 0
    assert [h["state"] for h in body["history"]] == ["firing"]
    free[0] = 50 * GB
    await alerts_until(api, lambda a: a == {})
    assert [h["state"] for h in (await api.get("/alerts")).json()["history"]] == [
        "resolved",
        "firing",
    ]
    free[0] = 0
    found = await alerts_until(api, lambda a: "disk_low" in a)
    assert found["disk_low"]["acknowledged"] is False  # cleared by the resolution
    # an acknowledged alert survives a restart of the daemon
    assert (await api.post("/alerts/disk_low/ack")).status_code == 200
    await api.aclose()
    daemon.stop.set()
    await asyncio.wait_for(daemon.task, 20)
    again = Daemon(config, make_settings(server, tmp_path), disk_free=lambda _p: 0)
    api = await again.api()
    await asyncio.sleep(0.3)
    found = await active_alerts(api)
    assert found["disk_low"]["acknowledged"] is True
    assert [h["state"] for h in (await api.get("/alerts")).json()["history"]].count("firing") == 2
    await api.aclose()
    again.stop.set()
    await asyncio.wait_for(again.task, 20)


# -- backups in the daemon ----------------------------------------------------------------


async def test_the_daemon_backs_up_at_start_and_reports_a_failing_backup(
    tmp_path: Path, server: FakeStreamServer
) -> None:
    config = ops_config(tmp_path, backup_every="0.3s", backup_keep=2)
    blocker = config.state.parent / "backups"
    config.state.parent.mkdir(parents=True)
    blocker.write_text("a file where the backup folder should be")
    daemon = Daemon(config, make_settings(server, tmp_path))
    api = await started(daemon, server)
    found = await alerts_until(api, lambda a: "backup_failed" in a)
    assert found["backup_failed"]["level"] == "error"
    blocker.unlink()  # fixed: the next scheduled backup works and the alert resolves
    await alerts_until(api, lambda a: "backup_failed" not in a)
    (backup,) = list_backups(config.state)[-1:]
    assert check_backup(backup) == SCHEMA_VERSION
    with StateStore.open_readonly(backup) as copy:
        assert copy.schema_version == SCHEMA_VERSION
    assert "backup_done" in [kind for kind, _ in events(config.state)]
    await api.aclose()
    daemon.stop.set()
    await asyncio.wait_for(daemon.task, 20)
    assert len(list_backups(config.state)) <= 2  # never more than backup_keep
