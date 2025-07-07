"""Fallback stub used when the JUCE plugin is not built."""

from __future__ import annotations

from typing import Any


def generate_notes(opts: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return a static note with growl/altissimo flags."""
    opts = opts or {}
    return [
        {
            "note": 60 if not opts.get("altissimo") else 72,
            "growl": bool(opts.get("growl")),
            "velocity": 100,
            "offset": 0.0,
        }
    ]
