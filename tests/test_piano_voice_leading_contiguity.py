import pytest
from music21 import harmony, instrument

from generator.piano_generator import PianoGenerator

class SimplePiano(PianoGenerator):
    def _get_pattern_keys(self, musical_intent, overrides):
        return "rh_test", "lh_test"


def make_gen():
    patterns = {
        "rh_test": {"pattern": [{"offset": 0, "duration": 1, "type": "chord"}], "length_beats": 1.0},
        "lh_test": {"pattern": [{"offset": 0, "duration": 1, "type": "root"}], "length_beats": 1.0},
    }
    return SimplePiano(
        part_name="piano",
        part_parameters=patterns,
        default_instrument=instrument.Piano(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg={},
    )


def test_voice_leading_contiguity():
    gen = make_gen()
    chords = [harmony.ChordSymbol(l) for l in ["Dm7", "G7", "Cmaj7", "Dm7", "G7", "Cmaj7"]]
    seq = gen._voice_leading(chords, mode="shell")
    leaps = []
    for prev, cur in zip(seq, seq[1:]):
        for a, b in zip(prev, cur):
            leaps.append(abs(a.ps - b.ps))
    large = [d for d in leaps if d > 14]
    assert len(large) / max(1, len(leaps)) < 0.05
