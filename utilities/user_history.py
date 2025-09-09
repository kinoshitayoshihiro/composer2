from __future__ import annotations

import json
from pathlib import Path

_HISTORY_FILE = Path.home() / ".otokotoba" / "history.jsonl"


def record_generate(config: dict, events: list[dict]) -> None:
    """Append a generation entry to the history file."""
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"config": config, "events": events}) + "\n")


def load_history(max_items: int = 1000) -> list[dict]:
    """Return up to ``max_items`` history entries."""
    if not _HISTORY_FILE.exists():
        return []
    try:
        with _HISTORY_FILE.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception:
        return []
    data: list[dict] = []
    for line in lines[-max_items:]:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return data

__all__ = ["record_generate", "load_history"]
