from music21 import instrument

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


def test_pedal_marks_alignment():
    gen = make_gen()
    events = gen._pedal_marks("medium", 8.0)
    ons = events[::2]
    offs = events[1::2]
    for on, off in zip(ons, offs):
        assert abs(on["time"] % 4.0) < 1e-6
        assert abs(off["time"] % 4.0) < 1e-6
        assert abs(off["time"] - on["time"] - 4.0) < 1e-6
