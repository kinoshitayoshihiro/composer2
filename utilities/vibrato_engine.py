"""Utility functions to generate basic MIDI expression curves."""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Literal


Event = Tuple[Literal["pitch_wheel", "aftertouch"], float, int]


def generate_vibrato(
    duration_qL: float, depth: float, rate: float, step_qL: float = 0.0625
) -> List[Event]:
    """Return vibrato pitch bend and aftertouch events.

    Pitch bend (``"pitch_wheel"``) and CC74 aftertouch events are produced
    alternately starting at ``time=0`` with ``step_qL`` spacing.  ``depth`` is
    interpreted in semitones assuming a ±2 range for pitch bends and is also
    mapped to the CC74 value.

    Parameters
    ----------
    duration_qL:
        Duration of the note in quarterLength.
    depth:
        Depth in semitones. Converted to pitch-wheel value assuming +/-2 range.
    rate:
        Oscillation rate in cycles per quarter note.
    step_qL:
        Spacing for the vibrato waveform in quarterLength. Defaults to one 64th
        note (0.0625).
    """
    events: List[Event] = []
    if duration_qL <= 0 or step_qL <= 0:
        return events
    semitone_range = 2.0
    amplitude = depth / semitone_range * 8192
    cc_val = max(0, min(127, int(round(64 + depth * 63))))
    t = 0.0
    index = 0
    while t <= duration_qL + 1e-9:
        if index % 2 == 0:
            bend = int(round(amplitude * math.sin(2 * math.pi * rate * t)))
            events.append(("pitch_wheel", round(t, 6), bend))
        else:
            events.append(("aftertouch", round(t, 6), cc_val))
        t += step_qL
        index += 1
    return events


def generate_gliss(
    start_pitch: int, end_pitch: int, duration_qL: float
) -> List[Tuple[int, float]]:
    """Return pitch values for a glissando.

    Values are linearly interpolated from ``start_pitch`` to ``end_pitch`` at
    32nd-note spacing.
    """

    step = 0.125
    if duration_qL <= 0 or start_pitch == end_pitch:
        return [(start_pitch, 0.0)]

    count = max(1, int(round(duration_qL / step)))
    step_pitch = (end_pitch - start_pitch) / float(count)
    events: List[Tuple[int, float]] = []
    for i in range(count + 1):
        t = min(duration_qL, i * step)
        p = int(round(start_pitch + step_pitch * i))
        events.append((p, t))
    return events


def generate_trill(
    pitch: int, duration_qL: float, rate: float = 10.0
) -> List[Tuple[int, float, int]]:
    """Return a sequence of trilled notes.

    Notes alternate one semitone above and below ``pitch`` at the given ``rate``
    in cycles per quarter note. Velocities are randomized by ±5 around 64 for a
    slightly human feel.  Each tuple in the result is ``(pitch, time, velocity)``.
    """

    if duration_qL <= 0 or rate <= 0:
        return []

    step = 1.0 / rate
    rng = random.Random(0)
    events: List[Tuple[int, float, int]] = []
    t = 0.0
    use_upper = True
    while t <= duration_qL + 1e-9:
        p = pitch + (1 if use_upper else -1)
        vel = max(1, min(127, 64 + rng.randint(-5, 5)))
        events.append((p, round(t, 6), vel))
        use_upper = not use_upper
        t += step
    return events


__all__ = ["generate_vibrato", "generate_gliss", "generate_trill"]
