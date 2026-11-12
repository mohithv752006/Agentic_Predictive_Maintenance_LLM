"""Structured logging setup (plain text or one JSON object per line)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Format records as JSON, including any ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure the root logger once; safe to call repeatedly."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        if getattr(handler, "_pdm_handler", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler._pdm_handler = True  # type: ignore[attr-defined]
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
    root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "sentence_transformers", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Module logger helper."""
    return logging.getLogger(name)
