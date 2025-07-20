"""Helpers for time conversions respecting tempo changes."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pretty_midi


@lru_cache(maxsize=8)
def _tempo_segments(pm: pretty_midi.PrettyMIDI) -> tuple[np.ndarray, np.ndarray]:
    tempos, times = pm.get_tempo_changes()
    return np.asarray(tempos, dtype=float), np.asarray(times, dtype=float)


def seconds_to_qlen(pm: pretty_midi.PrettyMIDI, t: float) -> float:
    """Convert absolute time in seconds to quarterLength units."""

    tempos, times = _tempo_segments(pm)
    beats = 0.0
    end_time = pm.get_end_time()
    last_tempo = tempos[-1]
    last_end = end_time
    for tempo, start, end in zip(tempos, times, np.append(times[1:], end_time)):
        if t <= start:
            break
        seg_end = min(t, end)
        beats += (seg_end - start) * (tempo / 60.0)
        last_tempo = tempo
        last_end = end
        if t <= end:
            return beats
    if t > last_end:
        beats += (t - last_end) * (last_tempo / 60.0)
    return beats


__all__ = ["seconds_to_qlen"]
