import json
import logging
import os
from pathlib import Path
from typing import Dict

import yaml

logger = logging.getLogger(__name__)

cc_map: Dict[str, int] = {}


def _default_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "cc_map.yml"


def load_cc_map(path: str | None = None) -> Dict[str, int]:
    """Load CC number mapping from YAML/JSON file."""

    if path is None:
        path = os.environ.get("CC_MAP_PATH")
    if not path:
        path = str(_default_path())
    try:
        with open(path, "r", encoding="utf-8") as fh:
            if path.endswith(".json"):
                data = json.load(fh)
            else:
                data = yaml.safe_load(fh)
    except Exception as exc:  # pragma: no cover - optional
        logger.warning("Failed to load CC map %s: %s", path, exc)
        data = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                cc_map[str(k)] = int(v)
            except Exception:
                logger.debug("Invalid CC number for %s: %r", k, v)
    return cc_map


load_cc_map()

