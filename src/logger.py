"""Structured logging setup for the trading bot."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Merge any extra fields attached via `extra={}`
        for key in ("symbol", "signal", "trade_id", "pnl", "action"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry)


def setup_logger(
    name: str = "crypto_bot",
    level: int = logging.INFO,
    json_output: bool = True,
) -> logging.Logger:
    """Create and return a configured logger.

    Args:
        name: Logger name.
        level: Logging level.
        json_output: If True, emit JSON lines; otherwise use human-readable format.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(module)s: %(message)s")
        )

    logger.addHandler(handler)
    logger.propagate = False
    return logger
