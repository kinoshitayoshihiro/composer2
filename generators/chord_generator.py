from __future__ import annotations

import random
from typing import Iterable

from utilities.progression_templates import get_progressions


def _pick_progression(bucket: str, mode: str = "major") -> str:
    """Return one progression randomly from the YAML bucket."""
    try:
        candidates: Iterable[str] = get_progressions(bucket, mode)
    except KeyError:
        return "I V vi IV"
    return random.choice(list(candidates))
