"""Helpers for CC event handling."""

from __future__ import annotations

from collections.abc import Iterable

from music21 import stream

CCEvent = tuple[float, int, int]


def merge_cc_events(
    base: Iterable[CCEvent], more: Iterable[CCEvent]
) -> list[CCEvent]:
    """Merge CC events where later events override earlier ones."""
    base_set = set(base)
    more_set = set(more)
    result: dict[tuple[float, int], int] = {
        (float(t), int(c)): int(v)
        for t, c, v in base_set
    }
    for t, c, v in more_set:
        result[(float(t), int(c))] = int(v)

    return [(t, c, v) for (t, c), v in sorted(result.items())]


def to_sorted_dicts(events: Iterable[CCEvent]) -> list[dict]:
    """Return sorted list of dicts from CC event tuples."""
    return [
        {"time": t, "cc": c, "val": v}
        for (t, c, v) in sorted(events, key=lambda x: x[0])
    ]


def finalize_cc_events(part: stream.Part) -> list[dict]:
    """Merge ``_extra_cc`` into ``extra_cc`` and sort by time."""
    if not hasattr(part, "_extra_cc") and not hasattr(part, "extra_cc"):
        part.extra_cc = []
        return part.extra_cc

    events: set[CCEvent] = set(getattr(part, "_extra_cc", set()))
    if hasattr(part, "extra_cc"):
        existing = [
            (e["time"], e["cc"], e["val"]) if isinstance(e, dict) else e
            for e in getattr(part, "extra_cc", [])
        ]
        events = set(merge_cc_events(events, set(existing)))
    part.extra_cc = to_sorted_dicts(events)
    if hasattr(part, "metadata") and part.metadata is not None:
        part.metadata.extra_cc = part.extra_cc
    if hasattr(part, "_extra_cc"):
        delattr(part, "_extra_cc")
    return part.extra_cc

__all__ = [
    "merge_cc_events",
    "to_sorted_dicts",
    "finalize_cc_events",
    "CCEvent",
]

