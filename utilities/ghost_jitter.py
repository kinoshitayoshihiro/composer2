from __future__ import annotations

import random
from music21 import note

MU = -0.004
SIGMA = 0.008
MAX_SHIFT = 0.015


def apply_ghost_jitter(n: note.Note, rng: random.Random, *, tempo_bpm: float) -> note.Note:
    """Shift note offset by sampled ghost-note jitter."""
    shift = rng.normalvariate(MU, SIGMA)
    shift = max(-MAX_SHIFT, min(MAX_SHIFT, shift))
    beat_shift = shift * tempo_bpm / 60.0
    n.offset += beat_shift
    return n
