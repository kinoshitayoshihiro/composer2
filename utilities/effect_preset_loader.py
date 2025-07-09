from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class EffectPresetLoader:
    _LIB: dict[str, dict] = {}
    _PATH: Path | None = None

    @classmethod
    def load(cls, path: str) -> None:
        p = Path(path)
        cls._PATH = p
        try:
            with open(p, "r", encoding="utf-8") as fh:
                if p.suffix.lower() == ".json":
                    data = json.load(fh)
                else:
                    data = yaml.safe_load(fh)
        except Exception as exc:  # pragma: no cover - optional
            logger.error("Failed to load effect presets %s: %s", path, exc)
            return
        if isinstance(data, dict):
            cls._LIB = {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}
        else:
            cls._LIB = {}

    @classmethod
    def get(cls, name: str) -> dict | None:
        preset = cls._LIB.get(name)
        if preset is None:
            logger.warning("Effect preset '%s' not found", name)
        return preset

    @classmethod
    def reload(cls) -> None:
        """Reload the preset file if previously loaded."""
        if cls._PATH is not None:
            cls.load(str(cls._PATH))
