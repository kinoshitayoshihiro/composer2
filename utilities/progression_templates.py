from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_YAML = Path(__file__).with_name("progression_templates.yaml")
DEFAULT_PATH = DEFAULT_YAML  # backward compatibility


@lru_cache()
def _load(path: str | Path = DEFAULT_YAML) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Progression template file must contain a mapping")
    if not isinstance(data.get("_default", {}), dict):
        raise ValueError("Missing _default bucket")
    return data


def get_progressions(
    bucket: str, mode: str = "major", path: str | Path = DEFAULT_YAML
) -> list[str]:
    """Return list of progressions.

    Fallback order:
    1. exact bucket / mode
    2. exact bucket / "major"
    3. "_default"  / mode
    4. "_default"  / "major"
    Raises KeyError if nothing found.
    """
    data = _load(path)
    try:
        return list(data[bucket][mode])
    except KeyError:
        try:
            return list(data[bucket]["major"])
        except KeyError:
            try:
                return list(data["_default"][mode])
            except KeyError:
                return list(data["_default"]["major"])


if __name__ == "__main__":
    import argparse
    import json
    import pprint
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("bucket")
    ap.add_argument("mode", nargs="?", default="major")
    ns = ap.parse_args()
    try:
        progs = get_progressions(ns.bucket, ns.mode)
        data = _load()
        if ns.bucket not in data:
            raise KeyError(ns.bucket)
        pprint.pp(progs)
    except KeyError as e:
        sys.exit(f"Not found: {e}")
