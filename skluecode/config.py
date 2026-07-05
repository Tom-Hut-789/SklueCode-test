from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import AppConfig, ProtocolType, SessionMode


class ConfigError(ValueError):
    """Raised when the YAML config is missing required fields or values."""


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        raise ConfigError("Config file must contain a YAML mapping at the top level.")

    return AppConfig(
        protocol=_parse_protocol(raw.get("protocol")),
        model=_require_string(raw, "model"),
        base_url=_require_string(raw, "base_url"),
        api_key=_require_string(raw, "api_key"),
        session_mode=_parse_session_mode(raw.get("session_mode", SessionMode.EPHEMERAL.value)),
        enable_extended_thinking=_parse_bool(raw.get("enable_extended_thinking", False), "enable_extended_thinking"),
        thinking_budget_tokens=_parse_optional_int(raw.get("thinking_budget_tokens")),
        storage_path=_parse_optional_string(raw.get("storage_path", "data/sessions"), "storage_path"),
    )


def _require_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Config field '{key}' is required and must be a non-empty string.")
    return value.strip()


def _parse_optional_string(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"Config field '{key}' must be a string when provided.")
    if not value.strip():
        raise ConfigError(f"Config field '{key}' must not be empty when provided.")
    return value.strip()


def _parse_protocol(value: Any) -> ProtocolType:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("Config field 'protocol' is required and must be a non-empty string.")
    try:
        return ProtocolType(value.strip().lower())
    except ValueError as error:
        allowed = ", ".join(item.value for item in ProtocolType)
        raise ConfigError(f"Unsupported protocol '{value}'. Expected one of: {allowed}.") from error


def _parse_session_mode(value: Any) -> SessionMode:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("Config field 'session_mode' must be a non-empty string when provided.")
    try:
        return SessionMode(value.strip().lower())
    except ValueError as error:
        allowed = ", ".join(item.value for item in SessionMode)
        raise ConfigError(f"Unsupported session_mode '{value}'. Expected one of: {allowed}.") from error


def _parse_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"Config field '{key}' must be true or false.")


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("Config field 'thinking_budget_tokens' must be an integer when provided.")
    if value <= 0:
        raise ConfigError("Config field 'thinking_budget_tokens' must be greater than 0.")
    return value
