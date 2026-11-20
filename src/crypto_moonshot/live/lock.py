"""The single-instance lock of a state file.

The daemon, a replay and `live restore` take an exclusive, non-blocking `flock` on
`<state>.lock` for as long as they use the state file, so two of them can never write one file.
The lock belongs to the open file, so it goes away with the process, whatever way it ends; the
lock file itself stays (deleting it would race with a process that is about to lock it).
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from crypto_moonshot.live.state import StateError


class StateLockedError(StateError):
    """Another process holds the state file's lock."""


def lock_path(state: Path) -> Path:
    """`<state>.lock`, next to the state file."""
    return state.with_name(state.name + ".lock")


class StateLock:
    """Exclusive lock on a state file; use as a context manager or `acquire` / `release`."""

    def __init__(self, state: Path) -> None:
        self.state = state
        self.path = lock_path(state)
        self._file: IO[str] | None = None

    @property
    def held(self) -> bool:
        return self._file is not None

    def acquire(self) -> None:
        """Take the lock or raise `StateLockedError` at once (it never waits)."""
        if self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+")
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            file.seek(0)
            holder = file.read().strip()
            file.close()
            who = f" (process {holder})" if holder.isdigit() else ""
            msg = (
                f"the state file {self.state} is in use by another moonshot process{who}: a "
                f"daemon, a replay or a restore holds {self.path}; stop it first"
            )
            raise StateLockedError(msg) from None
        file.seek(0)
        file.truncate()
        file.write(f"{os.getpid()}\n")
        file.flush()
        self._file = file

    def release(self) -> None:
        file, self._file = self._file, None
        if file is not None:
            file.close()  # closing drops the flock

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
