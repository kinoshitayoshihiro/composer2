import random
import pytest

pytest.importorskip("music21")
from music21 import note

from utilities.live_buffer import apply_late_humanization


def test_apply_late_humanization() -> None:
    notes = [note.Note("C4", quarterLength=1.0)]
    apply_late_humanization(notes, jitter_ms=(5, 5), bpm=120.0, rng=random.Random(0))
    assert abs(notes[0].offset) > 0
