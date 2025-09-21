# utilities/pattern_loader.py
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml


@lru_cache(maxsize=16)
def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
