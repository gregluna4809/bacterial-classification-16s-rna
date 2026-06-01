from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bacterial_classifier.paths import resolve_path


DATA_PATH_ENV_VAR = "BACTERIAL_FASTA_PATH"


@dataclass(frozen=True)
class AppConfig:
    fasta_path: Path


def load_config(
    data_path: str | None = None,
    config_path: str | None = None,
    env_file: str | None = ".env",
) -> AppConfig:
    """Load configuration from CLI value, config file, or environment.

    Precedence is:
    1. explicit data_path argument
    2. YAML config value named fasta_path
    3. BACTERIAL_FASTA_PATH from .env or the process environment
    """
    if env_file:
        _load_env_file(env_file)

    config_values = _load_yaml_config(config_path) if config_path else {}
    configured_path = (
        data_path
        or config_values.get("fasta_path")
        or os.getenv(DATA_PATH_ENV_VAR)
    )

    if not configured_path:
        raise ValueError(
            "FASTA path was not provided. Use --data-path, --config with "
            "fasta_path, or set BACTERIAL_FASTA_PATH in .env."
        )

    return AppConfig(fasta_path=resolve_path(configured_path))


def _load_yaml_config(config_path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load config files. Install dependencies with "
            "'pip install -r requirements.txt'."
        ) from exc

    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")

    return loaded


def _load_env_file(env_file: str) -> None:
    path = resolve_path(env_file)
    if not path.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError(
            "python-dotenv is required to load .env files. Install dependencies "
            "with 'pip install -r requirements.txt', or pass --env-file ''."
        ) from exc

    load_dotenv(path)
