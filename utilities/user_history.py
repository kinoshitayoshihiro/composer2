from __future__ import annotations

import json
from pathlib import Path

_HISTORY_FILE = Path.home() / ".modcompose_history.json"


def record_generate(config: dict, events: list[dict]) -> None:
    history = load_history()
    history.append({"config": config, "events": events})
    with _HISTORY_FILE.open("w", encoding="utf-8") as fh:
        json.dump(history, fh)


def load_history() -> list[dict]:
    if _HISTORY_FILE.exists():
        try:
            with _HISTORY_FILE.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return []
    return []

__all__ = ["record_generate", "load_history"]
