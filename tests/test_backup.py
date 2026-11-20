"""Backups, restore and the single-instance lock of the state file."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from engine_fakes import dump_state
from ops_fakes import add_position
from replay_fakes import END, H1, START, at, rule_strategy, run_config, strategy_dict

from crypto_moonshot.live.backup import (
    BackupError,
    BackupScheduler,
    backup_name,
    check_backup,
    list_backups,
    make_backup,
    prune_backups,
    restore_backup,
)
from crypto_moonshot.live.lock import StateLock, StateLockedError, lock_path
from crypto_moonshot.live.replay import prepare_replay, run_replay
from crypto_moonshot.live.state import (
    MIGRATIONS,
    SCHEMA_VERSION,
    EventLevel,
    StateStore,
)
from crypto_moonshot.strategy.base import Signals

NOW = datetime(2024, 5, 6, 7, 8, 9, tzinfo=UTC)


@pytest.fixture
def state(tmp_path: Path) -> Path:
    path = tmp_path / "state" / "paper.sqlite"
    path.parent.mkdir()
    with StateStore.open(path) as store:
        store.upsert_strategy("alpha", {}, 1000.0)
        for n in range(20):
            store.add_event(NOW, EventLevel.INFO, "k", f"event {n}")
    return path


def events(path: Path) -> list[str]:
    with StateStore.open_readonly(path) as store:
        return [e.message for e in store.events_since()]


# -- the lock -----------------------------------------------------------------------------------


def test_a_second_holder_is_refused_at_once_and_the_lock_is_released(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    first = StateLock(path)
    first.acquire()
    assert lock_path(path) == tmp_path / "s.sqlite.lock"
    with pytest.raises(
        StateLockedError, match=r"in use by another moonshot process \(process \d+\)"
    ):
        StateLock(path).acquire()
    first.release()
    with StateLock(path):  # free again
        assert StateLock(path).held is False
    with StateLock(path):
        pass


def test_the_lock_is_released_when_its_holder_is_collected(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    holder = StateLock(path)
    holder.acquire()
    del holder
    StateLock(path).acquire()


def test_a_replay_holds_the_lock_until_it_is_closed(replay_root: Path, tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    config = run_config(path)
    first = prepare_replay(
        config, at(START), at(END), replay_root, plugins={"cap": rule_strategy()}
    )
    with pytest.raises(StateLockedError):
        prepare_replay(config, at(START), at(END), replay_root, plugins={"cap": rule_strategy()})
    first.close()
    second = prepare_replay(
        config, at(START), at(END), replay_root, plugins={"cap": rule_strategy()}
    )
    second.close()


def test_a_failed_prepare_does_not_keep_the_lock(replay_root: Path, tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    config = run_config(path, [strategy_dict(source="tests:not_loaded", symbols=["NOPEUSDT"])])
    with pytest.raises(Exception):  # noqa: B017, PT011 - any failure while building
        prepare_replay(config, at(START), at(END), replay_root)
    StateLock(path).acquire()


# -- making a backup ----------------------------------------------------------------------------


def test_a_backup_is_a_verified_self_contained_copy(state: Path) -> None:
    result = make_backup(state, now=NOW)
    assert result.path == state.parent / "backups" / "paper-20240506T070809Z.sqlite"
    assert result.path.name == backup_name(state, NOW)
    assert result.schema_version == SCHEMA_VERSION
    assert result.size == result.path.stat().st_size > 0
    assert sorted(p.name for p in result.path.parent.iterdir()) == [result.path.name]  # no -wal
    assert check_backup(result.path) == SCHEMA_VERSION
    assert events(result.path) == events(state)
    with StateStore.open_readonly(result.path) as store:
        assert [s.name for s in store.strategies()] == ["alpha"]
        assert store.schema_version == SCHEMA_VERSION


def test_a_backup_works_while_the_daemon_holds_the_file_open_and_the_lock(state: Path) -> None:
    with StateStore.open(state) as live, StateLock(state):
        live.add_event(NOW, EventLevel.INFO, "k", "written just before")
        result = make_backup(state, now=NOW)
        live.add_event(NOW, EventLevel.INFO, "k", "written after")  # the writer is not blocked
    assert events(result.path)[-1] == "written just before"
    assert len(events(result.path)) == 21


def test_an_out_directory_is_created_and_used(state: Path, tmp_path: Path) -> None:
    result = make_backup(state, tmp_path / "elsewhere" / "deep", now=NOW)
    assert result.path.parent == tmp_path / "elsewhere" / "deep"
    assert list_backups(state, result.path.parent) == [result.path]
    assert list_backups(state) == []


def test_a_missing_state_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(BackupError, match="no state file at"):
        make_backup(tmp_path / "nope.sqlite")
    assert not (tmp_path / "backups").exists() or list((tmp_path / "backups").iterdir()) == []


def test_a_state_file_that_is_not_a_database_makes_no_backup(tmp_path: Path) -> None:
    junk = tmp_path / "junk.sqlite"
    junk.write_text("this is not a database at all, " * 50)
    with pytest.raises(BackupError):
        make_backup(junk, now=NOW)
    assert list((tmp_path / "backups").iterdir()) == []  # not even a partial file


def test_a_failed_integrity_check_leaves_no_backup(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import crypto_moonshot.live.backup as module

    def corrupt(path: Path) -> int:
        raise BackupError(f"{path} failed the integrity check: boom")

    monkeypatch.setattr(module, "check_backup", corrupt)
    with pytest.raises(BackupError, match="integrity check"):
        make_backup(state, now=NOW)
    assert list((state.parent / "backups").iterdir()) == []


def test_retention_keeps_exactly_the_newest(state: Path) -> None:
    made = [make_backup(state, now=NOW + timedelta(hours=h)).path for h in range(6)]
    (state.parent / "backups" / "unrelated.txt").write_text("keep me")
    other = state.parent / "backups" / "other-20240506T070809Z.sqlite"
    other.write_bytes(b"another state file's backup")
    deleted = prune_backups(state, 4)
    assert deleted == made[:2]
    assert list_backups(state) == made[2:]
    assert other.exists()
    assert (state.parent / "backups" / "unrelated.txt").exists()
    assert prune_backups(state, 4) == []  # nothing more to delete
    assert prune_backups(state, 10) == []


# -- a backup while the engine writes ---------------------------------------------------


def test_backups_taken_while_a_replay_writes_are_consistent(
    replay_root: Path, replay_signals: Signals, tmp_path: Path
) -> None:
    path = tmp_path / "state.sqlite"
    taken: list[Path] = []
    config = run_config(path, [strategy_dict(capital=1000.0, sizing={"slots": 3})])
    counter = [0]

    def hook(step: str, tick: Any) -> None:
        # Mid-tick, inside the engine's open transaction at the tick marker, and between steps.
        counter[0] += 1
        if step in ("before_tick_marker", "after_broker") and counter[0] % 7 == 0:
            taken.append(
                make_backup(path, tmp_path / "b", now=NOW + timedelta(seconds=len(taken))).path
            )

    result = run_replay(
        config,
        at(START),
        at(END),
        replay_root,
        plugins={"cap": rule_strategy()},
        precomputed={"cap": replay_signals},
        step_hook=hook,
    )
    assert result.trades.height > 30
    assert len(taken) >= 5
    for backup in taken:
        assert check_backup(backup) == SCHEMA_VERSION
        with StateStore.open_readonly(backup) as store:
            # a committed snapshot: nothing half written, the books derive, no dangling rows
            closed = store.closed_positions()
            assert all(p.pnl is not None and p.exit_time is not None for p in closed)
            store.derive_book("cap", 1000.0, store.last_ticks().get(H1, at(START)))
            assert store._db.execute("PRAGMA foreign_key_check").fetchall() == []
            fills = len(store.fills())
            opened = len(closed) + len([p for p in store.live_positions() if p.entry_time])
            assert fills >= opened  # every entered position has its entry fill
    # the replay itself was not disturbed: same trades as without backups
    plain = tmp_path / "plain.sqlite"
    again = run_replay(
        run_config(plain, [strategy_dict(capital=1000.0, sizing={"slots": 3})]),
        at(START),
        at(END),
        replay_root,
        plugins={"cap": rule_strategy()},
        precomputed={"cap": replay_signals},
    )
    assert again.trades.equals(result.trades)
    with StateStore.open(path) as a, StateStore.open(plain) as b:
        assert dump_state(a) == dump_state(b)


def test_backups_from_another_thread_while_a_replay_runs(
    replay_root: Path, replay_signals: Signals, tmp_path: Path
) -> None:
    path = tmp_path / "state.sqlite"
    config = run_config(path, [strategy_dict(capital=1000.0, sizing={"slots": 3})])
    started = threading.Event()
    finished = threading.Event()
    failures: list[BaseException] = []

    def replay() -> None:
        try:
            run_replay(
                config,
                at(START),
                at(END),
                replay_root,
                plugins={"cap": rule_strategy()},
                precomputed={"cap": replay_signals},
                step_hook=lambda step, tick: started.set(),
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=replay)
    thread.start()
    started.wait(30)
    made = 0
    while not finished.is_set():
        backup = make_backup(path, tmp_path / "b", now=NOW + timedelta(milliseconds=made * 1500))
        assert check_backup(backup.path) == SCHEMA_VERSION
        StateStore.open_readonly(backup.path).close()
        made += 1
    thread.join(30)
    assert not failures
    assert made >= 1


# -- the scheduler ------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_the_scheduler_backs_up_at_start_then_every_period_and_prunes(state: Path) -> None:
    sink: list[tuple[EventLevel, str, str]] = []
    stamps = iter(NOW + timedelta(minutes=m) for m in range(100))
    scheduler = BackupScheduler(
        state,
        every=0.01,
        keep=3,
        on_event=lambda level, kind, message, data: sink.append((level, kind, message)),
        clock=lambda: next(stamps),
    )
    task = asyncio.create_task(scheduler.run())
    async with asyncio.timeout(10):
        while len([s for s in sink if s[1] == "backup_done"]) < 5:
            await asyncio.sleep(0.005)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert scheduler.last_error is None
    assert len(list_backups(state)) == 3  # exactly `keep`
    assert all(check_backup(p) == SCHEMA_VERSION for p in list_backups(state))
    assert sink[0][1] == "backup_done"


@pytest.mark.anyio
async def test_a_failing_backup_is_reported_and_clears_when_it_works_again(state: Path) -> None:
    sink: list[tuple[EventLevel, str]] = []
    scheduler = BackupScheduler(
        state,
        every=1,
        keep=3,
        on_event=lambda level, kind, message, data: sink.append((level, kind)),
    )
    blocker = state.parent / "backups"
    blocker.write_text("a file where the folder should be")
    assert await scheduler.run_once() is False
    assert scheduler.last_error is not None
    assert sink == [(EventLevel.ERROR, "backup_failed")]
    blocker.unlink()
    assert await scheduler.run_once() is True
    assert scheduler.last_error is None
    assert sink[-1] == (EventLevel.INFO, "backup_done")


# -- restore ------------------------------------------------------------------------------------


def test_restore_replaces_the_state_and_keeps_the_old_file(state: Path) -> None:
    backup = make_backup(state, now=NOW)
    with StateStore.open(state) as store:
        store.add_event(NOW, EventLevel.INFO, "k", "after the backup")
    assert "after the backup" in events(state)
    result = restore_backup(backup.path, state, now=NOW + timedelta(hours=1))
    assert result.replaced == state.with_name("paper.sqlite.before-restore-20240506T080809Z")
    assert result.schema_version == SCHEMA_VERSION
    assert "after the backup" not in events(state)
    assert len(events(state)) == 20
    assert "after the backup" in events(result.replaced)
    assert backup.path.exists()  # the backup itself is untouched
    with StateStore.open(state) as store:  # opens (WAL again) and works
        store.add_event(NOW, EventLevel.INFO, "k", "new")
    assert not state.with_name("paper.sqlite.restoring").exists()


def test_restore_keeps_what_was_still_in_the_wal(state: Path) -> None:
    backup = make_backup(state, now=NOW)
    live = StateStore.open(state)  # an unclean stop: the WAL is not checkpointed
    live.add_event(NOW, EventLevel.INFO, "k", "only in the wal")
    live._db.execute("PRAGMA wal_checkpoint(PASSIVE)")
    live.add_event(NOW, EventLevel.INFO, "k", "later in the wal")
    live._db.close()
    result = restore_backup(backup.path, state, now=NOW)
    assert result.replaced is not None
    assert "later in the wal" in events(result.replaced)
    assert not state.with_name("paper.sqlite-wal").exists()


def test_restore_into_a_missing_state_file(state: Path, tmp_path: Path) -> None:
    backup = make_backup(state, now=NOW)
    target = tmp_path / "fresh" / "new.sqlite"
    result = restore_backup(backup.path, target)
    assert result.replaced is None
    assert events(target) == events(state)


def test_restore_is_refused_while_the_lock_is_held(state: Path) -> None:
    backup = make_backup(state, now=NOW)
    before = state.read_bytes()
    with StateLock(state), pytest.raises(StateLockedError, match="in use"):
        restore_backup(backup.path, state)
    assert state.read_bytes() == before
    assert not list(state.parent.glob("*.before-restore-*"))


def test_restore_is_refused_while_a_replay_holds_the_state(
    replay_root: Path, tmp_path: Path
) -> None:
    path = tmp_path / "state.sqlite"
    run = prepare_replay(
        run_config(path), at(START), at(END), replay_root, plugins={"cap": rule_strategy()}
    )
    backup = make_backup(path, now=NOW)
    with pytest.raises(StateLockedError):
        restore_backup(backup.path, path)
    run.close()
    restore_backup(backup.path, path)


def test_a_corrupt_backup_is_refused_and_nothing_changes(state: Path, tmp_path: Path) -> None:
    good = make_backup(state, now=NOW).path
    data = bytearray(good.read_bytes())
    for offset in range(4096, len(data), 512):  # trash the data pages, keep the header
        data[offset : offset + 64] = b"\xff" * 64
    broken = tmp_path / "broken.sqlite"
    broken.write_bytes(bytes(data))
    before = state.read_bytes()
    with pytest.raises(BackupError):
        restore_backup(broken, state)
    junk = tmp_path / "junk.sqlite"
    junk.write_text("not a database " * 200)
    with pytest.raises(BackupError, match="not a usable state file"):
        restore_backup(junk, state)
    with pytest.raises(BackupError, match="no such file"):
        restore_backup(tmp_path / "missing.sqlite", state)
    assert state.read_bytes() == before
    assert not list(state.parent.glob("*.before-restore-*"))


def test_a_backup_of_a_newer_schema_is_refused_and_an_older_one_accepted(
    state: Path, tmp_path: Path
) -> None:
    newer = tmp_path / "newer.sqlite"
    with StateStore.open(newer):
        pass
    db = sqlite3.connect(newer)
    db.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION + 1),))
    db.commit()
    db.close()
    with pytest.raises(BackupError, match="newer than this build's"):
        restore_backup(newer, state)
    # a version-1 file restores, and the next open migrates it
    old = tmp_path / "old.sqlite"
    with StateStore.open(old, migrations=MIGRATIONS[:1]) as store:
        store.upsert_strategy("legacy", {}, 5.0)
    assert check_backup(old) == 1
    restore_backup(old, state)
    with StateStore.open(state) as store:
        assert store.schema_version == SCHEMA_VERSION
        assert [s.name for s in store.strategies()] == ["legacy"]


def test_a_backup_made_with_positions_restores_them(state: Path) -> None:
    with StateStore.open(state) as store:
        add_position(store, "alpha", "AAAUSDT", NOW, pnl=5.0, exit_time=NOW + timedelta(hours=1))
        add_position(store, "alpha", "BBBUSDT", NOW, exit_time=NOW)
    backup = make_backup(state, now=NOW)
    state.unlink()
    restore_backup(backup.path, state)
    with StateStore.open(state) as store:
        assert [p.symbol for p in store.closed_positions()] == ["AAAUSDT"]
        assert [p.symbol for p in store.live_positions()] == ["BBBUSDT"]
