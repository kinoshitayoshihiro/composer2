from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_chordmap(path: str | Path) -> dict[str, Any]:
    """Load chordmap YAML and resolve new fields."""
    p = Path(path).expanduser().resolve()
    data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    sections = data.get("sections")
    if not isinstance(sections, dict):
        sections = data.get("global_settings", {}).get("sections", {})
    data["sections"] = sections if isinstance(sections, dict) else {}
    if isinstance(sections, dict):
        for name, sec in sections.items():
            if not isinstance(sec, dict):
                continue
            base = p.parent
            vocal = sec.get("vocal_midi_path", f"{name}_vocal.mid")
            sec["vocal_midi_path"] = str((base / vocal).resolve())
            cjson = sec.get("consonant_json")
            if cjson is not None:
                sec["consonant_json"] = str((base / cjson).resolve())
            else:
                sec["consonant_json"] = None
    return data

__all__ = ["load_chordmap"]
