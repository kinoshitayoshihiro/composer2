import yaml
import functools
from pathlib import Path
from typing import Dict, List

DEFAULT_YAML = Path(__file__).with_suffix("").parent / "progression_templates.yaml"

@functools.lru_cache()
def _load(path: str | Path = DEFAULT_YAML) -> Dict[str, Dict[str, List[str]]]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("progression_templates.yaml root must be a mapping")
    return data

def get_progressions(bucket: str, *, mode: str = "major", path: str | Path = DEFAULT_YAML) -> List[str]:
    """Return list[str] of chord progressions for the given emotion `bucket` and tonal `mode` ('major'|'minor' etc.). Raises KeyError if missing."""
    data = _load(path)
    return data[bucket][mode]

if __name__ == "__main__":
    import sys

    bucket = sys.argv[1] if len(sys.argv) > 1 else "soft_reflective"
    mode = sys.argv[2] if len(sys.argv) > 2 else "major"
    try:
        lst = get_progressions(bucket, mode=mode)
        print(yaml.dump(lst, allow_unicode=True))
    except KeyError as e:
        raise SystemExit(f"Missing key: {e}")
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
