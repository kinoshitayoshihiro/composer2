from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_sections(path: str | Path) -> list[dict[str, Any]]:
    """Load section list from YAML file.

    Each section must contain ``vocal_midi_path`` and optionally
    ``consonant_json``. All paths are resolved relative to the YAML file.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        raise ValueError("sections YAML must be a list")

    sections: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict section entry: %r", item)
            continue
        sec = dict(item)
        if "vocal_midi_path" not in sec:
            raise ValueError("section missing 'vocal_midi_path'")
        base = p.parent
        sec["vocal_midi_path"] = str((base / sec["vocal_midi_path"]).resolve())
        cjson = sec.get("consonant_json")
        if cjson is not None:
            sec["consonant_json"] = str((base / cjson).resolve())
        else:
            sec["consonant_json"] = None
        sections.append(sec)
    return sections
