"""Helpers for CC event handling."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Iterable as TypingIterable


from music21 import stream

CCEvent = tuple[float, int, int]


def _to_tuple_set(events: Iterable[CCEvent] | Iterable[dict]) -> set[CCEvent]:
    result: set[CCEvent] = set()
    for e in events:
        if isinstance(e, dict):
            result.add((float(e.get("time", 0.0)), int(e.get("cc", 0)), int(e.get("val", 0))))
        else:
            t, c, v = e
            result.add((float(t), int(c), int(v)))
    return result


def _to_tuple_list(events: Iterable[CCEvent] | Iterable[dict]) -> list[CCEvent]:
    """Normalize iterable of events to a list of ``(time, cc, value)`` tuples."""
    result: list[CCEvent] = []
    for e in events:
        if isinstance(e, dict):
            result.append(
                (
                    float(e.get("time", 0.0)),
                    int(e.get("cc", 0)),
                    int(e.get("val", 0)),
                )
            )
        else:
            t, c, v = e
            result.append((float(t), int(c), int(v)))
    return result


def merge_cc_events(
    base: Iterable[CCEvent] | Iterable[dict],
    more: Iterable[CCEvent] | Iterable[dict],
    *,
    as_dict: bool = False,
) -> list[CCEvent] | list[dict]:
    """Merge CC events where later events override earlier ones."""
    base_set = _to_tuple_set(base)
    more_set = _to_tuple_set(more)
    result: dict[tuple[float, int], int] = {
        (float(t), int(c)): int(v)
        for t, c, v in base_set
    }
    for t, c, v in more_set:
        result[(float(t), int(c))] = int(v)

    merged = [(t, c, v) for (t, c), v in sorted(result.items())]
    if as_dict:
        return [{"time": t, "cc": c, "val": v} for (t, c, v) in merged]
    return merged


def to_sorted_dicts(events: Iterable[CCEvent] | Iterable[dict]) -> list[dict]:
    """Return sorted list of dicts from CC events.

    Duplicate events with the same time and controller keep the last value.
    """
    tuples = _to_tuple_list(events)
    latest: dict[tuple[float, int], int] = {}
    for t, c, v in tuples:
        latest[(t, c)] = v
    return [
        {"time": t, "cc": c, "val": v}
        for (t, c), v in sorted(latest.items(), key=lambda x: (x[0][0], x[0][1]))
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


def add_cc_events(part: stream.Part, events: TypingIterable[dict] | TypingIterable[CCEvent]) -> None:
    """Merge CC events into ``part.extra_cc`` preserving order."""

    base = getattr(part, "extra_cc", [])
    merged = merge_cc_events(base, events)
    part.extra_cc = to_sorted_dicts(merged)

__all__ = [
    "merge_cc_events",
    "to_sorted_dicts",
    "finalize_cc_events",
    "add_cc_events",
    "CCEvent",
]


def _install_part_cc_property() -> None:
    """Install extra_cc property on :class:`music21.stream.Part`.

    The function is idempotent; calling it multiple times will have no effect.
    """

    if getattr(stream.Part, "_extra_cc_installed", False):
        return
    if isinstance(getattr(stream.Part, "extra_cc", None), property):
        stream.Part._extra_cc_installed = True
        return

    def _get(self: stream.Part) -> list[dict]:
        events = getattr(self, "_extra_cc", set())
        data = self.__dict__.get("extra_cc")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            extra_set = _to_tuple_set(data)
            events = set(merge_cc_events(events, extra_set))
        return to_sorted_dicts(events)

    def _set(
        self: stream.Part, value: TypingIterable[dict] | TypingIterable[CCEvent]
    ) -> None:
        tuples = _to_tuple_set(value)
        current = getattr(self, "_extra_cc", set())
        merged = merge_cc_events(current, tuples)
        self._extra_cc = set(merged)
        self.__dict__["extra_cc"] = to_sorted_dicts(self._extra_cc)

    stream.Part.extra_cc = property(_get, _set)
    stream.Part._extra_cc_installed = True


_install_part_cc_property()

