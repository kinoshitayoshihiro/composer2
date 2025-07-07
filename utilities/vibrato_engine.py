import math
from typing import List, Dict


def generate_vibrato(duration_qL: float, depth: float, rate: float, *, step: float = 0.05) -> List[Dict[str, int | float]]:
    """Return vibrato pitchwheel/aftertouch events.

    Parameters
    ----------
    duration_qL:
        Duration of the note in quarterLength.
    depth:
        Depth in semitones. Converted to pitch-wheel value assuming +/-2 range.
    rate:
        Oscillation rate in cycles per quarter note.
    step:
        Step size for waveform in quarterLength. Defaults to 0.05.
    """
    events: List[Dict[str, int | float]] = []
    if duration_qL <= 0 or step <= 0:
        return events
    semitone_range = 2.0
    amplitude = depth / semitone_range * 8192
    cc_val = max(0, min(127, int(round(64 + depth * 63))))
    t = 0.0
    while t <= duration_qL + 1e-9:
        bend = int(round(amplitude * math.sin(2 * math.pi * rate * t)))
        events.append({"time": round(t, 6), "type": "pitchwheel", "value": bend})
        events.append({"time": round(t, 6), "type": "aftertouch", "value": cc_val})
        t += step
    return events


__all__ = ["generate_vibrato"]
