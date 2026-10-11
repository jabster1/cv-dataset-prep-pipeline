"""Structured logging.

Console output is human-readable (``12:00:01 INFO  ingest  frames_extracted video=a.mp4 n=30``);
the optional file handler writes one JSON object per line so runs can be inspected or
aggregated with standard tooling.

Use :func:`log_event` to attach machine-readable fields to a log line.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_LOGGER_NAME = "cv_pipeline"
_FIELDS_ATTR = "event_fields"


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``cv_pipeline`` namespace."""
    if name.startswith(ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log ``event`` with structured key/value ``fields``."""
    logger.log(level, event, extra={_FIELDS_ATTR: fields})


def _short_name(record: logging.LogRecord) -> str:
    name = record.name
    prefix = ROOT_LOGGER_NAME + "."
    return name[len(prefix) :] if name.startswith(prefix) else name


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        fields: dict[str, Any] = getattr(record, _FIELDS_ATTR, None) or {}
        kv = " ".join(f"{k}={_fmt_value(v)}" for k, v in fields.items())
        line = f"{ts} {record.levelname:<7} {_short_name(record):<20} {record.getMessage()}"
        if kv:
            line = f"{line}  {kv}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, _FIELDS_ATTR, None)
        if fields:
            payload["fields"] = fields
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return f'"{text}"' if " " in text else text


def setup_logging(level: str | int = "INFO", log_file: Path | None = None) -> None:
    """Configure the ``cv_pipeline`` logger tree. Safe to call more than once."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level if isinstance(level, int) else level.upper())
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)
