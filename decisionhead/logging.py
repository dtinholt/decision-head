"""Structured JSON logging to stderr.

Every log line is one JSON object with at least ``ts`` (ISO-8601 UTC),
``level``, and ``event``; a request-scoped log call additionally carries
``request_id`` so every line belonging to one HTTP request can be grepped
out with ``jq 'select(.request_id == "...")'``. This module never writes
to stdout (stdout is reserved for the human-readable startup banner
printed by :mod:`decisionhead.__main__`) and never logs a secret: callers
must not pass an API key or bearer token as a field value; nothing here
scans for that, so it is a hard rule for every call site instead.
"""

from __future__ import annotations

import datetime
import json
import sys
import threading
from typing import Any, TextIO

JSONDict = dict[str, Any]

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


class JSONLogger:
    """A minimal structured logger. Not a replacement for the stdlib
    ``logging`` module's ecosystem (handlers, formatters); this package
    has exactly one destination (stderr) and one format (one JSON object
    per line), so a small dedicated writer is clearer than configuring
    that module to do the same thing."""

    def __init__(self, level: str = "INFO", stream: TextIO | None = None) -> None:
        """``stream`` defaults to ``sys.stderr``; tests may inject an
        in-memory stream instead."""
        self.level = level.upper()
        if self.level not in _LEVELS:
            raise ValueError(f"unknown log level {level!r}")
        self._stream: TextIO = stream if stream is not None else sys.stderr
        self._lock = threading.Lock()

    def _enabled(self, level: str) -> bool:
        return _LEVELS[level] >= _LEVELS[self.level]

    def log(self, level: str, event: str, **fields: Any) -> None:
        """Writes one JSON line if ``level`` meets this logger's
        configured threshold; a no-op otherwise. ``fields`` are merged
        into the record alongside the timestamp, level, and event name."""
        if not self._enabled(level):
            return
        record: JSONDict = {
            "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds"),
            "level": level,
            "event": event,
        }
        record.update(fields)
        line = json.dumps(record, default=str)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def debug(self, event: str, **fields: Any) -> None:
        """Shorthand for ``self.log("DEBUG", event, **fields)``."""
        self.log("DEBUG", event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        """Shorthand for ``self.log("INFO", event, **fields)``."""
        self.log("INFO", event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        """Shorthand for ``self.log("WARNING", event, **fields)``."""
        self.log("WARNING", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        """Shorthand for ``self.log("ERROR", event, **fields)``."""
        self.log("ERROR", event, **fields)


_default_logger: JSONLogger | None = None
_default_lock = threading.Lock()


def configure(level: str = "INFO", stream: TextIO | None = None) -> JSONLogger:
    """Sets and returns the process-wide default logger. Safe to call more
    than once (a later call replaces the default); this exists so
    ``decisionhead serve`` can apply ``--log-level`` once at startup and
    every module import a shared, already-configured logger."""
    global _default_logger
    with _default_lock:
        _default_logger = JSONLogger(level=level, stream=stream)
        return _default_logger


def get_logger() -> JSONLogger:
    """Returns the process-wide default logger, configuring one at the
    ``INFO`` level on first use if nothing called :func:`configure` yet."""
    global _default_logger
    with _default_lock:
        if _default_logger is None:
            _default_logger = JSONLogger()
        return _default_logger
