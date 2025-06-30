from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_emotion_profile(path: str | Path) -> Dict[str, Any]:
    """Load emotion profile YAML and return as dictionary."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Emotion profile must be a mapping")
    return data
