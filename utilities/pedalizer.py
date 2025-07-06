"""Auto sustain pedal CC64 generation based on chord changes."""

from __future__ import annotations

from collections.abc import Iterable
from music21 import harmony

from .cc_tools import CCEvent


def _get_root(symbol: str) -> str | None:
    try:
        cs = harmony.ChordSymbol(symbol)
        r = cs.root()
        return r.name if r is not None else None
    except Exception:
        return None


def _get_value(obj: object, *keys: str):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
    for k in keys:
        if hasattr(obj, k):
            return getattr(obj, k)
    return None


DEFAULT_BPM = 120.0


def generate_pedal_cc(
    chord_stream: Iterable[object], *, lift_offset_ms: int = 10
) -> set[CCEvent]:
    """Generate CC64 lift/down events when harmony changes."""
    events: set[CCEvent] = set()
    prev_root: str | None = None
    prev_symbol: str | None = None
    scale = DEFAULT_BPM / 60000.0
    for ch in chord_stream:
        symbol = _get_value(ch, "chord_symbol_for_voicing", "chord", "symbol")
        if symbol is None:
            continue
        root = _get_root(str(symbol))
        offset_val = _get_value(ch, "absolute_offset_beats", "offset")
        if offset_val is None:
            continue
        off = float(offset_val)
        changed = prev_root is None or root != prev_root or symbol != prev_symbol
        if changed:
            lift_time = max(0.0, off - float(lift_offset_ms) * scale)
            events.add((lift_time, 64, 0))
            events.add((off, 64, 127))
        prev_root = root
        prev_symbol = symbol
    return events

__all__ = ["generate_pedal_cc"]
