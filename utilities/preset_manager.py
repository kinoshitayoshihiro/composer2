from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

PRESET_DIR = Path.home() / ".otokotoba" / "presets"


def _ensure_dir() -> None:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)


def _preset_path(name: str) -> Path:
    return PRESET_DIR / f"{name}.yml"


def list_presets() -> list[str]:
    """Return available preset names."""
    _ensure_dir()
    return sorted(p.stem for p in PRESET_DIR.glob("*.yml"))


def load_preset(name: str) -> dict[str, Any]:
    """Load preset ``name`` and return the configuration."""
    path = _preset_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Preset '{name}' not found")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_preset(name: str, config: dict[str, Any]) -> None:
    """Save ``config`` under ``name``."""
    _ensure_dir()
    path = _preset_path(name)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
