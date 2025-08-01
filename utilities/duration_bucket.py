from __future__ import annotations

"""Utilities for mapping note durations to discrete buckets."""

# Duration boundaries in quarter lengths.  Durations fall into bucket ``i`` if
# ``_BOUNDS[i-1] < dur <= _BOUNDS[i]``.  Values beyond the last bound map to the
# final bucket.
_BOUNDS: list[float] = [0.25, 0.5, 1.0, 3.0, 5.0, 8.0]

__all__ = ["_BOUNDS", "to_bucket"]


def to_bucket(dur: float) -> int:
    """Convert a duration (in quarter notes) into a bucket index."""
    boundaries = _BOUNDS
    for i, bound in enumerate(boundaries):
        if dur <= bound:
            return i
    return len(boundaries)
