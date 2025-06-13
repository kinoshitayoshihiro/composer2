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

def test_minimal_pitch_leap():
    gen = make_gen()
    cs1 = harmony.ChordSymbol("C")
    gen._render_hand_part("RH", cs1, 1.0, "rh_test", {})
    first = list(gen._prev_voicings["RH"])

    cs2 = harmony.ChordSymbol("G")
    gen._render_hand_part("RH", cs2, 1.0, "rh_test", {})
    second = list(gen._prev_voicings["RH"])

    first_sorted = sorted(first, key=lambda p: p.ps)
    second_sorted = sorted(second, key=lambda p: p.ps)
    dist = sum(abs(a.ps - b.ps) for a, b in zip(first_sorted, second_sorted))
    assert dist <= 6
