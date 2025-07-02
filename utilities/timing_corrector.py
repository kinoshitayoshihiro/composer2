from __future__ import annotations

import copy
from music21 import stream
from typing import Optional

try:
    from cyext.humanize import timing_correct_part as cy_timing_correct_part
except Exception:  # pragma: no cover - optional
    cy_timing_correct_part = None


class TimingCorrector:
    """Exponential moving average timing smoother."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)

    def correct_part(self, part: stream.Part) -> stream.Part:
        """Return a timing-smoothed copy of ``part``."""
        if cy_timing_correct_part is not None:
            return cy_timing_correct_part(part, self.alpha)
        new_part = copy.deepcopy(part)
        notes = list(new_part.recurse().notes)
        if not notes:
            return new_part
        ema = notes[0].offset - round(notes[0].offset)
        for n in notes:
            target = round(n.offset)
            delta = n.offset - target
            ema += self.alpha * (delta - ema)
            n.offset = target + ema
        return new_part
