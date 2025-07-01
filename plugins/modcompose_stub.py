"""Fallback stub used when the JUCE plugin is not built."""

from __future__ import annotations

from typing import Any


def generateBar(preset: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return a static bar of events."""
    return [{"instrument": "kick", "offset": 0.0, "velocity": 100}]
