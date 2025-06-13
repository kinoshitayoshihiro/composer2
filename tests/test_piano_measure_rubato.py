from music21 import instrument, stream
from generator.piano_generator import PianoGenerator


class SimplePiano(PianoGenerator):
    def _get_pattern_keys(self, musical_intent, overrides):
        return "rh_test", "lh_test"


def make_gen(rng):
    patterns = {
        "rh_test": {
            "pattern": [
                {"offset": 0, "duration": 1, "type": "chord"},
                {"offset": 4, "duration": 1, "type": "chord"},
            ],
            "length_beats": 8.0,
        },
        "lh_test": {"pattern": [{"offset": 0, "duration": 8, "type": "root"}], "length_beats": 8.0},
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
        rng=rng,
    )


def test_measure_rubato_shift():
    rng = __import__("random").Random(0)
    gen = make_gen(rng)
    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 8.0,
        "chord_symbol_for_voicing": "C",
        "musical_intent": {},
        "part_params": {},
    }
    parts = gen.compose(section_data=section)
    combined = stream.Stream()
    for p in parts.values():
        for n in p.recurse().notes:
            combined.insert(n.offset, n)
    offs = [float(n.offset) for n in combined.recurse().notes]
    first = min(offs, key=lambda x: x)
    second = min((o for o in offs if o > 3.5), default=None)
    assert -0.04 <= first <= 0.04
    assert second is not None and 3.96 <= second <= 4.04
