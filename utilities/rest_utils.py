"""Helpers for dealing with vocal rest windows."""

from __future__ import annotations


def get_rest_windows(
    vocal_metrics: dict | None, min_dur: float = 0.5
) -> list[tuple[float, float]]:
    """Return ``(start, end)`` tuples for rests lasting at least ``min_dur`` beats."""
    if not vocal_metrics:
        return []

    return [
        (start, start + dur)
        for start, dur in vocal_metrics.get("rests", [])
        if dur >= min_dur
    ]
