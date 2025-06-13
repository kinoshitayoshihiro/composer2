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

def _pitches(part):
    elem = part.flatten().notes[0]
    return [p.nameWithOctave for p in (elem.pitches if hasattr(elem, "pitches") else [elem.pitch])]

def test_voicing_styles_output():
    cs = harmony.ChordSymbol("C")
    spread = _pitches(make_gen()._render_hand_part("RH", cs, 1.0, "rh_test", {}, voicing_style="spread"))
    closed = _pitches(make_gen()._render_hand_part("RH", cs, 1.0, "rh_test", {}, voicing_style="closed"))
    inverted = _pitches(make_gen()._render_hand_part("RH", cs, 1.0, "rh_test", {}, voicing_style="inverted"))
    assert spread == ["C4", "E5", "G5"]
    assert closed == ["C4", "E4", "G4"]
    assert inverted == ["E4", "G4", "C5"]
