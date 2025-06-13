import pytest
from music21 import instrument

from generator.piano_generator import PianoGenerator


class SimplePiano(PianoGenerator):
    def _get_pattern_keys(self, musical_intent, overrides):
        return "rh_test", "lh_test"


def make_gen():
    patterns = {
        "rh_test": {
            "pattern": [
                {"offset": 0, "duration": 1, "type": "chord", "velocity_factor": 0.5},
                {"offset": 1, "duration": 1, "type": "chord", "velocity_factor": 1.0},
            ],
            "length_beats": 2.0,
        },
        "lh_test": {
            "pattern": [{"offset": 0, "duration": 2, "type": "root", "velocity_factor": 1.0}],
            "length_beats": 2.0,
        },
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


def test_pedal_value_by_intensity():
    gen = make_gen()
    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 2.0,
        "chord_symbol_for_voicing": "C",
        "musical_intent": {"intensity": "high"},
        "part_params": {},
    }
    part = gen.compose(section_data=section)
    cc64 = [c for c in getattr(part, "extra_cc", []) if c["number"] == 64]
    assert cc64 and cc64[0]["value"] == 127

    section["musical_intent"] = {"intensity": "low"}
    part = gen.compose(section_data=section)
    cc64 = [c for c in getattr(part, "extra_cc", []) if c["number"] == 64]
    assert cc64 and cc64[0]["value"] == 64


def test_cc11_curve_inserted():
    gen = make_gen()
    section = {
        "section_name": "B",
        "absolute_offset": 0.0,
        "q_length": 2.0,
        "chord_symbol_for_voicing": "C",
        "musical_intent": {"intensity": "medium"},
        "part_params": {},
    }
    part = gen.compose(section_data=section)
    c11 = [c for c in getattr(part, "extra_cc", []) if c["number"] == 11]
    assert any(0 < c["time"] < 1.0 for c in c11)

