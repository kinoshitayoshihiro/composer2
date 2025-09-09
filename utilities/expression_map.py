import os
import json
import copy
import logging
from pathlib import Path
from typing import Dict, Any

import yaml

logger = logging.getLogger(__name__)


_DEF_PATH = Path(__file__).resolve().parents[1] / "data" / "expression_map.yml"


def load_expression_map(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Load expression map definitions from YAML or JSON.

    Parameters
    ----------
    path:
        Optional path to YAML/JSON file. Environment variable ``EXPRESSION_MAP_PATH``
        takes precedence when *path* is ``None``.
    """
    if path is None:
        path = os.environ.get("EXPRESSION_MAP_PATH")
    p = Path(path) if path else _DEF_PATH
    try:
        with open(p, "r", encoding="utf-8") as fh:
            if p.suffix.lower() == ".json":
                data = json.load(fh)
            else:
                data = yaml.safe_load(fh)
    except FileNotFoundError:  # pragma: no cover - optional
        logger.warning("Expression map file not found: %s", p)
        return {}
    except Exception as exc:  # pragma: no cover - optional
        logger.warning("Failed to load expression map %s: %s", p, exc)
        return {}
    mapping: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                mapping[str(k)] = dict(v)
    return mapping


def resolve_expression(
    section_name: str | None,
    intensity: str | None,
    style: str | None,
    mapping: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Return resolved expression mapping for given parameters."""
    keys = []
    if section_name and style and intensity:
        keys.append(f"{section_name}_{style}_{intensity}")
    if style and intensity:
        keys.append(f"{style}_{intensity}")
    if section_name and intensity:
        keys.append(f"{section_name}_{intensity}")
    if section_name and style:
        keys.append(f"{section_name}_{style}")
    if section_name:
        keys.append(section_name)
    if style:
        keys.append(style)
    if intensity:
        keys.append(intensity)
    for k in keys:
        if k in mapping:
            return copy.deepcopy(mapping[k])
    return copy.deepcopy(mapping.get("default", {}))
