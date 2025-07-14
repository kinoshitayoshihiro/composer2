"""Simple chord progression selector."""

from __future__ import annotations

import random
from typing import Any

from utilities.progression_templates import get_progressions


DEFAULT_PROGRESSION = "I V vi IV"


def choose_progression(section: Any) -> str:
    """Return a chord progression string for the given section data."""
    try:
        progs = get_progressions(section.emotion_bucket, mode=section.mode)
        return random.choice(progs)
    except KeyError:
        return DEFAULT_PROGRESSION
