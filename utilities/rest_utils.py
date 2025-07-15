"""Helpers for dealing with vocal rest windows."""

from __future__ import annotations

from typing import Dict, List, Tuple


def get_rest_windows(
    vocal_metrics: Dict | None, min_dur: float = 0.5
) -> List[Tuple[float, float]]:
    """Return (start, end) tuples for vocal rests longer than ``min_dur`` quarter notes."""
    windows: List[Tuple[float, float]] = []
    if vocal_metrics:
        for start, dur in vocal_metrics.get("rests", []):
            if dur >= min_dur:
                windows.append((start, start + dur))
    return windows
