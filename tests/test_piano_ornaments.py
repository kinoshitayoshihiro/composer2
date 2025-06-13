import json
from pathlib import Path
from music21 import instrument, expressions, stream
from generator.piano_generator import PianoGenerator
from utilities.override_loader import load_overrides


class SimplePiano(PianoGenerator):
    def _get_pattern_keys(self, musical_intent, overrides):
        return "rh_test", "lh_test"


def make_gen(rng=None):
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
        rng=rng,
    )


def test_mordent_and_grace(tmp_path: Path):
    overrides = {"SecA": {"piano": {"mordent": True, "grace_note": True}}}
    ov_path = tmp_path / "ov.json"
    ov_path.write_text(json.dumps(overrides))
    ov_model = load_overrides(ov_path)
    gen = make_gen()
    section = {
        "section_name": "SecA",
        "absolute_offset": 0.0,
        "q_length": 1.0,
        "chord_symbol_for_voicing": "C",
        "musical_intent": {},
        "part_params": {},
    }
    parts = gen.compose(section_data=section, overrides_root=ov_model)
    combined = stream.Stream()
    for p in parts.values():
        for n in p.recurse().notes:
            combined.insert(n.offset, n)
    notes = list(combined.recurse().notes)
    assert any(n.duration.isGrace for n in notes)
    assert any(any(isinstance(a, expressions.Mordent) for a in n.expressions) for n in notes)
