"""Logging setup for the long-running commands (`live run`, `live replay`, `provider run`).

`text` (default): `time level logger: message` lines on stderr. `json`: one JSON object per line
on stdout (what Docker's log driver wants) with `time` (ISO-8601 UTC), `level`, `logger`,
`message`, any `extra=` fields, and `exception` for a traceback.

Secrets never get into a log line: the code does not log them, and as a second line of defence
every value passed to `configure_logging(secrets=...)` is replaced by `***` in whatever gets
written, in both formats.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import IO, Any

_STANDARD = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}
REDACTED = "***"


class LogFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


class LogLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def _redact(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        text = text.replace(secret, REDACTED)
    return text


class JsonFormatter(logging.Formatter):
    """One JSON object per record."""

    def __init__(self, secrets: tuple[str, ...] = ()) -> None:
        super().__init__()
        self._secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            entry["exception"] = record.exc_text
        if record.stack_info:
            entry["stack"] = self.formatStack(record.stack_info)
        return _redact(json.dumps(entry, default=str, ensure_ascii=False), self._secrets)


class TextFormatter(logging.Formatter):
    def __init__(self, secrets: tuple[str, ...] = ()) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        return _redact(super().format(record), self._secrets)


def configure_logging(
    fmt: LogFormat = LogFormat.TEXT,
    level: LogLevel | int = LogLevel.INFO,
    *,
    secrets: Iterable[str] = (),
    stream: IO[str] | None = None,
) -> logging.Handler:
    """Send all logging to one handler (replacing the root logger's handlers); returns it.

    `stream` defaults to stderr for `text` and stdout for `json`.
    """
    hidden = tuple(s for s in secrets if s)
    if stream is None:
        stream = sys.stdout if fmt is LogFormat.JSON else sys.stderr
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(hidden) if fmt is LogFormat.JSON else TextFormatter(hidden))
    root = logging.getLogger()
    for old in list(root.handlers):
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(level if isinstance(level, int) else getattr(logging, level.value.upper()))
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its INFO lines repeat every request
    return handler
