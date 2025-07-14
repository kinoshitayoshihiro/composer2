from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).with_name("progression_templates.yaml")


@lru_cache()
def _load(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Progression template file must contain a mapping")
    return data


def get_progressions(bucket: str, *, mode: str = "major", path: str | Path = DEFAULT_PATH) -> list[str]:
    data = _load(path=path)
    try:
        modes = data[bucket]
        return list(modes[mode])
    except KeyError as exc:
        raise KeyError(bucket, mode) from exc
    except Exception as exc:
        raise
