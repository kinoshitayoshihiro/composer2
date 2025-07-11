from __future__ import annotations


def scale_velocity(v: int | float, factor: float) -> int:
    """Scale *v* by *factor* and clamp to MIDI velocity range."""
    val = float(v) * factor
    return max(1, min(127, int(round(val))))
