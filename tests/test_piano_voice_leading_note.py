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


def test_voice_leading_single_note():
    gen = make_gen()
    cs1 = harmony.ChordSymbol("C")
    part1 = gen._render_hand_part("LH", cs1, 1.0, "lh_test", {})
    first = part1.flatten().notes[0].pitch

    cs2 = harmony.ChordSymbol("G")
    part2 = gen._render_hand_part("LH", cs2, 1.0, "lh_test", {})
    second = part2.flatten().notes[0].pitch

    diff = abs(second.ps - first.ps)
    assert diff <= 2
