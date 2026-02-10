"""Configuration loader — merges settings.yaml with environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file and overlay environment variables.

    Environment variables referenced in the YAML (via `*_env` keys) are
    resolved at load time so the rest of the application never touches
    ``os.environ`` directly.
    """
    load_dotenv(_PROJECT_ROOT / ".env")

    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path, "r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    # Resolve exchange secrets from env vars
    exchange = cfg.get("exchange", {})
    exchange["api_key"] = os.getenv(exchange.pop("api_key_env", ""), "")
    exchange["api_secret"] = os.getenv(exchange.pop("api_secret_env", ""), "")

    return cfg


def get_nested(cfg: dict[str, Any], dotpath: str, default: Any = None) -> Any:
    """Retrieve a nested config value using dot notation.

    >>> get_nested(cfg, "risk.max_position_pct", 0.05)
    """
    keys = dotpath.split(".")
    current: Any = cfg
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default
        if current is None:
            return default
    return current
