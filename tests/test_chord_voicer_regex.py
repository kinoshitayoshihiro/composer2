from music21 import harmony
from generator.chord_voicer import sanitize_chord_label

import pytest

@pytest.mark.parametrize("label", ["C(add9)", "Gsus2", "Dsus2/G"])
def test_chord_voicer_handles_add9_sus2(label):
    sanitized = sanitize_chord_label(label)
    cs = harmony.ChordSymbol(sanitized)
    assert cs.pitches
