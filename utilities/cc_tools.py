"""Helpers for CC event handling."""

from __future__ import annotations

from typing import Iterable, Tuple, Dict, List


CCEvent = Tuple[float, int, int]


def merge_cc_events(base: Iterable[CCEvent], more: Iterable[CCEvent]) -> List[CCEvent]:
    """Merge CC events where later events override earlier ones."""
    result: Dict[tuple[float, int], int] = {(t, c): v for t, c, v in base}
    for t, c, v in more:
        result[(float(t), int(c))] = int(v)
    return [(t, c, v) for (t, c), v in result.items()]


def to_sorted_dicts(events: Iterable[CCEvent]) -> List[dict]:
    """Return sorted list of dicts from CC event tuples."""
    return [
        {"time": t, "cc": c, "val": v}
        for (t, c, v) in sorted(events, key=lambda x: x[0])
    ]

__all__ = ["merge_cc_events", "to_sorted_dicts", "CCEvent"]

