from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from eval import blec

Event = Mapping[str, float | int]


def swing_score(
    events: Sequence[Event], *, ideal_offbeat: float = 2 / 3, max_offset: float = 0.1
) -> float:
    """Return deviation from ``ideal_offbeat`` normalised to ``0-1``.

    Events must contain ``offset`` in beats.
    ``ideal_offbeat`` represents the expected off-beat position in beats
    (e.g. ``2/3`` for a shuffle feel). ``max_offset`` controls the
    normalisation range.
    """

    offbeats = []
    for ev in events:
        pos = float(ev.get("offset", 0)) % 1
        if 0.4 <= pos <= 0.8:
            offbeats.append(pos)
    if not offbeats:
        return 0.0
    actual = float(np.mean(offbeats))
    diff = abs(actual - ideal_offbeat)
    score = diff / max_offset
    return float(min(max(score, 0.0), 1.0))


def note_density(events: Sequence[Event], *, resolution: int = 16) -> float:
    """Return note count normalised by ``resolution``."""
    if not events:
        return 0.0
    return float(min(len(events) / resolution, 1.0))


def velocity_var(events: Sequence[Event]) -> float:
    """Return normalised velocity variance (0-1)."""
    if not events:
        return 0.0
    vels = [float(ev.get("velocity", 0)) for ev in events]
    var = float(np.var(vels))
    return min(var / (127.0**2 / 4), 1.0)


def blec_score(ref: Sequence[Event], pred: Sequence[Event]) -> float:
    """Return BLEC between reference and prediction."""
    return blec.blec(ref, pred)
