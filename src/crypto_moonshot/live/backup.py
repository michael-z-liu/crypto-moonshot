"""Backups of the state file: online copies, retention, restore.

`make_backup` copies the state file with SQLite's online backup API while the daemon keeps
writing (the file is in WAL mode, so a reader never blocks the writer and the copy is one
committed snapshot: a transaction in progress is not in it). The copy goes to
`backups/<stem>-<UTC timestamp>.sqlite` next to the state file (or to `--out`), is switched to
the rollback journal (one self-contained file, no `-wal`) and has to pass `PRAGMA
integrity_check` before it gets its final name, so a name in `backups/` is always a verified
backup. `BackupScheduler` makes one at daemon start and every `backup_every`, keeping the newest
`backup_keep`. `restore_backup` replaces the state file with a backup (never while another
process holds the file's lock) and keeps what it replaced as `<state>.before-restore-<time>`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crypto_moonshot.live.lock import StateLock
from crypto_moonshot.live.state import SCHEMA_VERSION, EventLevel, StateError

logger = logging.getLogger(__name__)

BACKUP_DIR = "backups"
_STAMP = "%Y%m%dT%H%M%SZ"
_SUFFIXES = ("", "-wal", "-shm")


class BackupError(StateError):
    """A backup could not be made, or a backup file cannot be used."""


@dataclass(frozen=True, slots=True)
class BackupResult:
    path: Path
    size: int
    schema_version: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RestoreResult:
    state: Path
    restored_from: Path
    replaced: Path | None
    """Where the state file that was replaced went (`None` if there was none)."""
    schema_version: int


def default_backup_dir(state: Path) -> Path:
    return state.parent / BACKUP_DIR


def backup_name(state: Path, now: datetime) -> str:
    return f"{state.stem}-{now.astimezone(UTC).strftime(_STAMP)}.sqlite"


def _pattern(state: Path) -> re.Pattern[str]:
    return re.compile(re.escape(state.stem) + r"-\d{8}T\d{6}Z\.sqlite")


def list_backups(state: Path, directory: Path | None = None) -> list[Path]:
    """The backups of `state` in `directory` (default `backups/`), oldest first."""
    folder = directory or default_backup_dir(state)
    if not folder.is_dir():
        return []
    pattern = _pattern(state)
    return sorted(p for p in folder.iterdir() if pattern.fullmatch(p.name))


def prune_backups(state: Path, keep: int, directory: Path | None = None) -> list[Path]:
    """Delete all but the newest `keep` backups; the deleted paths."""
    backups = list_backups(state, directory)
    doomed = backups[: max(0, len(backups) - keep)]
    for path in doomed:
        path.unlink(missing_ok=True)
    return doomed


def _open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def check_backup(path: Path) -> int:
    """Verify a state file or backup: it opens, `PRAGMA integrity_check` says `ok` and it is a
    state file (a schema version). Returns that version; raises `BackupError` otherwise."""
    if not path.is_file():
        msg = f"no such file: {path}"
        raise BackupError(msg)
    try:
        db = _open_readonly(path)
        try:
            rows = db.execute("PRAGMA integrity_check").fetchall()
            if [r[0] for r in rows] != ["ok"]:
                problems = "; ".join(str(r[0]) for r in rows[:3])
                msg = f"{path} failed the integrity check: {problems}"
                raise BackupError(msg)
            row = db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        finally:
            db.close()
    except sqlite3.DatabaseError as exc:
        msg = f"{path} is not a usable state file: {exc}"
        raise BackupError(msg) from None
    if row is None:
        msg = f"{path} is not a state file (no schema version)"
        raise BackupError(msg)
    return int(row[0])


def _copy(source: Path, target: Path, *, rollback_journal: bool) -> None:
    src = _open_readonly(source)
    try:
        dst = sqlite3.connect(target)
        try:
            src.backup(dst)
            if rollback_journal:
                dst.execute("PRAGMA journal_mode = DELETE")
        finally:
            dst.close()
    finally:
        src.close()


def make_backup(
    state: Path, directory: Path | None = None, *, now: datetime | None = None
) -> BackupResult:
    """Back `state` up into `directory` (default `backups/` next to it), safe while the daemon
    runs. Raises `BackupError` (the half-made copy is removed) if the copy is not sound."""
    if not state.is_file():
        msg = f"no state file at {state}"
        raise BackupError(msg)
    folder = directory or default_backup_dir(state)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC) if now is None else now
    final = folder / backup_name(state, stamp)
    partial = final.with_name(final.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        try:
            _copy(state, partial, rollback_journal=True)
        except sqlite3.DatabaseError as exc:
            msg = f"could not copy {state}: {exc}"
            raise BackupError(msg) from None
        version = check_backup(partial)
        os.replace(partial, final)
    finally:
        partial.unlink(missing_ok=True)
    return BackupResult(final, final.stat().st_size, version, stamp)


def restore_backup(backup: Path, state: Path, *, now: datetime | None = None) -> RestoreResult:
    """Replace `state` with `backup`.

    Checks the backup first (integrity; a schema version no newer than this build's; an older
    one is migrated when the daemon next opens it), then takes the state file's lock, so it
    refuses (`StateLockedError`) while a daemon or replay uses the file. The replaced file (and
    its `-wal`) is kept as `<state>.before-restore-<UTC time>`.
    """
    version = check_backup(backup)
    if version > SCHEMA_VERSION:
        msg = (
            f"{backup} has schema version {version}, newer than this build's {SCHEMA_VERSION}; "
            "upgrade crypto-moonshot to use it"
        )
        raise BackupError(msg)
    stamp = datetime.now(UTC) if now is None else now
    state.parent.mkdir(parents=True, exist_ok=True)
    with StateLock(state):
        staged = state.with_name(state.name + ".restoring")
        staged.unlink(missing_ok=True)
        try:
            _copy(backup, staged, rollback_journal=True)
            check_backup(staged)
            replaced: Path | None = None
            if state.exists():
                replaced = state.with_name(f"{state.name}.before-restore-{stamp.strftime(_STAMP)}")
                for suffix in _SUFFIXES:
                    old = state.with_name(state.name + suffix)
                    if not old.exists():
                        continue
                    if suffix == "-shm":
                        old.unlink()  # an index of the WAL, rebuilt on demand
                    else:
                        os.replace(old, replaced.with_name(replaced.name + suffix))
            else:
                for suffix in _SUFFIXES[1:]:  # a stale WAL must never meet the restored file
                    state.with_name(state.name + suffix).unlink(missing_ok=True)
            os.replace(staged, state)
        finally:
            staged.unlink(missing_ok=True)
    return RestoreResult(state, backup, replaced, version)


EventSink = Callable[[EventLevel, str, str, dict[str, Any] | None], None]


class BackupScheduler:
    """Backs the state file up at start and then every `every` seconds, keeping the newest
    `keep`; `last_error` is the latest failure (cleared by the next success) for the
    `backup_failed` health rule. Backups run in a worker thread with their own connection."""

    def __init__(
        self,
        state: Path,
        *,
        every: float,
        keep: int,
        on_event: EventSink,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.state = state
        self.every = every
        self.keep = keep
        self.last_error: str | None = None
        self.last_backup: BackupResult | None = None
        self._on_event = on_event
        self._clock = clock

    async def run_once(self) -> bool:
        """One backup and the pruning; whether it succeeded."""
        try:
            result = await asyncio.to_thread(make_backup, self.state, now=self._clock())
            await asyncio.to_thread(prune_backups, self.state, self.keep)
        except (StateError, OSError, sqlite3.Error) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.error("automatic backup failed: %s", self.last_error)
            self._on_event(EventLevel.ERROR, "backup_failed", self.last_error, None)
            return False
        self.last_error = None
        self.last_backup = result
        self._on_event(
            EventLevel.INFO,
            "backup_done",
            f"state backed up to {result.path.name}",
            {"path": str(result.path), "size": result.size},
        )
        return True

    async def run(self) -> None:
        while True:
            await self.run_once()
            await asyncio.sleep(self.every)
