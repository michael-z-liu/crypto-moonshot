"""Logging: JSON lines on stdout, text on stderr, and the API token never in either."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from engine_fakes import until
from stream_fakes import FakeStreamServer
from test_daemon import TOKEN, Daemon, make_config, make_settings

from crypto_moonshot.logs import LogFormat, LogLevel, configure_logging


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


def lines(stream: io.StringIO) -> list[str]:
    return [line for line in stream.getvalue().splitlines() if line]


def test_json_lines_are_valid_and_carry_the_documented_fields() -> None:
    out = io.StringIO()
    configure_logging(LogFormat.JSON, LogLevel.INFO, stream=out)
    log = logging.getLogger("moonshot.test")
    log.info("hello %s", "world", extra={"strategy": "cnn", "count": 3})
    log.warning('with a quote " and a newline\nin it')
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("it failed")
    log.debug("not shown at info")
    records = [json.loads(line) for line in lines(out)]
    assert len(records) == 3
    first = records[0]
    assert first["level"] == "info"
    assert first["logger"] == "moonshot.test"
    assert first["message"] == "hello world"
    assert first["strategy"] == "cnn"
    assert first["count"] == 3
    assert first["time"].endswith("+00:00")  # aware UTC
    assert records[1]["level"] == "warning"
    assert records[1]["message"] == 'with a quote " and a newline\nin it'
    assert records[2]["level"] == "error"
    assert "ValueError: boom" in records[2]["exception"]
    assert set(records[0]) == {"time", "level", "logger", "message", "strategy", "count"}


def test_the_level_filters() -> None:
    out = io.StringIO()
    configure_logging(LogFormat.JSON, LogLevel.WARNING, stream=out)
    logging.getLogger("x").info("quiet")
    logging.getLogger("x").error("loud")
    assert [json.loads(line)["message"] for line in lines(out)] == ["loud"]


def test_text_format_is_a_plain_line() -> None:
    out = io.StringIO()
    configure_logging(LogFormat.TEXT, LogLevel.INFO, stream=out)
    logging.getLogger("x.y").info("hello")
    (line,) = lines(out)
    assert line.endswith("INFO x.y: hello")


def test_configuring_twice_does_not_duplicate_lines() -> None:
    out = io.StringIO()
    configure_logging(LogFormat.JSON, stream=io.StringIO())
    configure_logging(LogFormat.JSON, stream=out)
    logging.getLogger("x").info("once")
    assert len(lines(out)) == 1


@pytest.mark.parametrize("fmt", [LogFormat.JSON, LogFormat.TEXT])
def test_secrets_are_masked_if_they_reach_a_log_line(fmt: LogFormat) -> None:
    out = io.StringIO()
    configure_logging(fmt, stream=out, secrets=[TOKEN, ""])
    log = logging.getLogger("x")
    log.info("the header was Bearer %s", TOKEN)
    log.info("extra", extra={"headers": {"authorization": f"Bearer {TOKEN}"}})
    try:
        raise RuntimeError(f"rejected {TOKEN}")
    except RuntimeError:
        log.exception("failure")
    assert TOKEN not in out.getvalue()
    assert "***" in out.getvalue()


@pytest.mark.anyio
async def test_a_running_daemon_logs_json_without_the_token(tmp_path: Path) -> None:
    out = io.StringIO()
    # No masking here: the code itself must never write the token.
    configure_logging(LogFormat.JSON, LogLevel.DEBUG, stream=out, secrets=[])
    async with FakeStreamServer() as server:
        daemon = Daemon(make_config(tmp_path), make_settings(server, tmp_path))
        api = await daemon.api()
        await until(lambda: server.connections != [] and server.subscribed(0) != [])
        # requests with right and wrong tokens, which must not be logged either
        await api.get("/status")
        await api.get("/status", headers={"Authorization": "Bearer wrong-token-value"})
        await api.get("/status", headers={"Authorization": f"Bearer {TOKEN}x"})
        await api.aclose()
        daemon.stop.set()
        await asyncio.wait_for(daemon.task, 20)
    text = out.getvalue()
    records = [json.loads(line) for line in lines(out)]  # every line is JSON
    assert any("API listening" in r["message"] for r in records)
    assert any(r["message"] == "stopping" for r in records)
    assert TOKEN not in text
    assert "wrong-token-value" not in text
