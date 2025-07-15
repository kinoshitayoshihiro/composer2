import random

from music21 import instrument

from generator.piano_generator import PianoGenerator
from utilities.rest_utils import get_rest_windows


class SimplePiano(PianoGenerator):
    def _get_pattern_keys(self, musical_intent, overrides):
        return "rh", "lh"


def make_gen(anticipatory=False):
    patterns = {
        "rh": {
            "pattern": [{"offset": 0.0, "duration": 1.0, "type": "chord"}],
            "length_beats": 1.0,
        },
        "lh": {
            "pattern": [{"offset": 0.0, "duration": 1.0, "type": "root"}],
            "length_beats": 1.0,
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
        main_cfg={"piano": {"anticipatory_chord": anticipatory}},
        rng=random.Random(0),
    )


def test_anticipatory_chord_notes():
    gen = make_gen(anticipatory=True)
    section = {
        "chord_symbol_for_voicing": "C",
        "q_length": 4.0,
        "part_params": {"piano": {}},
    }
    vm = {"rests": [(0.0, 1.0), (2.0, 1.0)], "onsets": []}
    parts = gen.compose(section_data=section, vocal_metrics=vm)
    rh = parts["piano_rh"].flatten().notes
    for start, end in get_rest_windows(vm):
        within = [n for n in rh if end - 0.125 <= n.offset < end]
        assert within


def test_no_anticipatory_chord():
    gen = make_gen(anticipatory=False)
    section = {
        "chord_symbol_for_voicing": "C",
        "q_length": 4.0,
        "part_params": {"piano": {}},
    }
    vm = {"rests": [(0.0, 1.0)], "onsets": []}
    parts = gen.compose(section_data=section, vocal_metrics=vm)
    rh = parts["piano_rh"].flatten().notes
    assert not [n for n in rh if 0.875 <= n.offset < 1.0]
