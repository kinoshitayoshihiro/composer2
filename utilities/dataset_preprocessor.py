from __future__ import annotations

from typing import Any, Iterable


def add_gliss_tokens(events: list[dict[str, Any]], slide_events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ``events`` with ``<gliss>`` tokens inserted for slides.

    Parameters
    ----------
    events : list[dict[str, Any]]
        Base event sequence sorted by offset.
    slide_events : Iterable[dict[str, Any]]
        Slide definitions with ``offset`` keys.
    """
    tokens = list(events)
    for ev in slide_events:
        tok = {"type": "special", "value": "<gliss>", "offset": float(ev.get("offset", 0.0))}
        tokens.append(tok)
    tokens.sort(key=lambda e: float(e.get("offset", 0.0)))
    return tokens


__all__ = ["add_gliss_tokens"]
