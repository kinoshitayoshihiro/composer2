from __future__ import annotations

import copy
from music21 import stream


class TimingCorrector:
    """Exponential moving average timing smoother."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)

    def correct_part(self, part: stream.Part) -> stream.Part:
        """Return a timing-smoothed copy of ``part``."""
        new_part = copy.deepcopy(part)
        notes = list(new_part.recurse().notes)
        if not notes:
            return new_part
        ema = 0.0
        for n in notes:
            target = round(n.offset)
            delta = n.offset - target
            ema += self.alpha * (delta - ema)
            n.offset = target + ema
        return new_part
