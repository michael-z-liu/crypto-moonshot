"""`moonshot live backup`, `live restore`, the log options and the watchdog exit code."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from api_fakes import TOKEN
from test_live_cli import ENV, invoke, rule_config

from crypto_moonshot import live_cli
from crypto_moonshot.live.backup import check_backup, list_backups, make_backup
from crypto_moonshot.live.daemon import DaemonError
from crypto_moonshot.live.lock import StateLock, StateLockedError
from crypto_moonshot.live.state import SCHEMA_VERSION, EventLevel, StateStore


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = rule_config(tmp_path)
    state = tmp_path / "state" / "run.sqlite"
    state.parent.mkdir()
    with StateStore.open(state) as store:
        from datetime import UTC, datetime

        for n in range(5):
            store.add_event(datetime(2024, 1, 1, tzinfo=UTC), EventLevel.INFO, "k", f"event {n}")
    return path


def state_of(tmp_path: Path) -> Path:
    return tmp_path / "state" / "run.sqlite"


def events(path: Path) -> list[str]:
    with StateStore.open_readonly(path) as store:
        return [e.message for e in store.events_since()]


# -- backup -------------------------------------------------------------------------------------


def test_backup_writes_a_verified_copy_next_to_the_state_file(config: Path, tmp_path: Path) -> None:
    result = invoke("live", "backup", "--config", str(config))
    assert result.exit_code == 0, result.output + result.stderr
    (made,) = list_backups(state_of(tmp_path))
    assert made.parent == tmp_path / "state" / "backups"
    assert str(made) in result.stdout
    assert "verified" in result.stdout
    assert check_backup(made) == SCHEMA_VERSION
    assert events(made) == events(state_of(tmp_path))


def test_backup_to_another_folder_as_json(config: Path, tmp_path: Path) -> None:
    out = tmp_path / "elsewhere"
    result = invoke("live", "backup", "--config", str(config), "--out", str(out), "--json")
    assert result.exit_code == 0, result.output + result.stderr
    document = json.loads(result.stdout)
    assert Path(document["path"]).parent == out
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["size"] > 0
    assert document["created_at"].endswith("+00:00")
    assert list_backups(state_of(tmp_path)) == []  # nothing in the default folder


def test_backup_works_while_a_daemon_holds_the_state_file(config: Path, tmp_path: Path) -> None:
    with StateStore.open(state_of(tmp_path)), StateLock(state_of(tmp_path)):
        result = invoke("live", "backup", "--config", str(config))
    assert result.exit_code == 0, result.output + result.stderr


def test_backup_errors(config: Path, tmp_path: Path) -> None:
    state_of(tmp_path).unlink()
    result = invoke("live", "backup", "--config", str(config))
    assert result.exit_code == 1
    assert "no state file at" in result.stderr
    assert invoke("live", "backup", "--config", str(tmp_path / "nope.toml")).exit_code == 2


# -- restore ------------------------------------------------------------------------------------


def test_restore_replaces_the_state_and_keeps_the_old_one(config: Path, tmp_path: Path) -> None:
    state = state_of(tmp_path)
    backup = make_backup(state).path
    with StateStore.open(state) as store:
        from datetime import UTC, datetime

        store.add_event(datetime(2024, 1, 2, tzinfo=UTC), EventLevel.INFO, "k", "after")
    result = invoke("live", "restore", str(backup), "--config", str(config))
    assert result.exit_code == 0, result.output + result.stderr
    assert "restored" in result.stdout
    assert "kept as" in result.stdout
    assert events(state) == [f"event {n}" for n in range(5)]
    (kept,) = state.parent.glob("run.sqlite.before-restore-*")
    assert "after" in events(kept)
    assert str(kept) in result.stdout


def test_restore_is_refused_while_the_state_is_in_use(config: Path, tmp_path: Path) -> None:
    state = state_of(tmp_path)
    backup = make_backup(state).path
    before = state.read_bytes()
    with StateLock(state):
        result = invoke("live", "restore", str(backup), "--config", str(config))
    assert result.exit_code == 1
    assert "in use by another moonshot process" in result.stderr
    assert state.read_bytes() == before


def test_restore_refuses_a_bad_backup(config: Path, tmp_path: Path) -> None:
    state = state_of(tmp_path)
    before = state.read_bytes()
    junk = tmp_path / "junk.sqlite"
    junk.write_text("definitely not sqlite " * 100)
    result = invoke("live", "restore", str(junk), "--config", str(config))
    assert result.exit_code == 1
    assert "not a usable state file" in result.stderr
    missing = invoke("live", "restore", str(tmp_path / "gone.sqlite"), "--config", str(config))
    assert missing.exit_code == 1
    assert "no such file" in missing.stderr
    empty = tmp_path / "empty.sqlite"
    sqlite3.connect(empty).close()  # a valid database that is not a state file
    other = invoke("live", "restore", str(empty), "--config", str(config))
    assert other.exit_code == 1
    assert "not a usable state file" in other.stderr or "not a state file" in other.stderr
    assert state.read_bytes() == before
    assert not list(state.parent.glob("*.before-restore-*"))


# -- run: exit codes and logging ----------------------------------------------------------------


def test_run_exits_3_for_a_watchdog_stop_and_1_when_the_state_is_in_use(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome: list[BaseException] = []

    async def fake(run_config: Any, settings: Any, *, host: str, port: int) -> None:
        raise outcome[0]

    monkeypatch.setattr(live_cli, "run_daemon", fake)
    outcome.append(DaemonError("the watchdog stopped the daemon: no tick", code=3))
    result = invoke("live", "run", "--config", str(config), env=ENV)
    assert result.exit_code == 3
    assert "the watchdog stopped the daemon" in result.stderr
    outcome[0] = StateLockedError("the state file x is in use by another moonshot process")
    result = invoke("live", "run", "--config", str(config), env=ENV)
    assert result.exit_code == 1
    assert "in use by another moonshot process" in result.stderr


def test_run_logs_json_lines_to_stdout_without_the_token(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake(run_config: Any, settings: Any, *, host: str, port: int) -> None:
        logging.getLogger("crypto_moonshot.live.daemon").info("API listening on port %d", port)
        logging.getLogger("crypto_moonshot.live.daemon").error("a slip: header Bearer %s", TOKEN)

    monkeypatch.setattr(live_cli, "run_daemon", fake)
    result = invoke(
        "live", "run", "--config", str(config), "--log-format", "json", "--port", "1234", env=ENV
    )
    assert result.exit_code == 0, result.output + result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert records[0]["message"] == "API listening on port 1234"
    assert records[0]["level"] == "info"
    assert records[0]["logger"] == "crypto_moonshot.live.daemon"
    assert TOKEN not in result.stdout + result.stderr  # masked even when code slips
    assert "***" in records[1]["message"]


def test_run_logs_text_to_stderr_by_default_and_honours_the_level(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake(run_config: Any, settings: Any, *, host: str, port: int) -> None:
        logging.getLogger("x").info("an info line")
        logging.getLogger("x").warning("a warning line")

    monkeypatch.setattr(live_cli, "run_daemon", fake)
    result = invoke("live", "run", "--config", str(config), env=ENV)
    assert "INFO x: an info line" in result.stderr
    assert result.stdout == ""
    quiet = invoke("live", "run", "--config", str(config), "--log-level", "warning", env=ENV)
    assert "an info line" not in quiet.stderr
    assert "WARNING x: a warning line" in quiet.stderr
    bad = invoke("live", "run", "--config", str(config), "--log-level", "loud", env=ENV)
    assert bad.exit_code == 2
    bad_format = invoke("live", "run", "--config", str(config), "--log-format", "xml", env=ENV)
    assert bad_format.exit_code == 2


def test_replay_accepts_the_log_options(replay_root: Path, tmp_path: Path) -> None:
    config = rule_config(tmp_path)
    result = invoke(
        "live", "replay", "--config", str(config), "--data-dir", str(replay_root),
        "--start", "2024-01-09", "--end", "2024-01-12", "--out", str(tmp_path / "out"),
        "--log-format", "json", "--log-level", "info",
    )  # fmt: skip
    assert result.exit_code == 0, result.output + result.stderr


def test_replay_is_refused_while_the_state_file_is_in_use(
    replay_root: Path, tmp_path: Path
) -> None:
    config = rule_config(tmp_path)
    state = tmp_path / "state" / "run.sqlite"
    state.parent.mkdir()
    with StateLock(state):
        result = invoke(
            "live", "replay", "--config", str(config), "--data-dir", str(replay_root),
            "--start", "2024-01-09", "--end", "2024-01-12", "--out", str(tmp_path / "out"),
        )  # fmt: skip
    assert result.exit_code == 1
    assert "in use by another moonshot process" in result.stderr
