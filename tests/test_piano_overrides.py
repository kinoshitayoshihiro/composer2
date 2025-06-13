import json
from pathlib import Path

from music21 import instrument, stream

from generator.piano_generator import PianoGenerator
from utilities.override_loader import load_overrides


class SimplePiano(PianoGenerator):
    def _get_pattern_keys(self, musical_intent, overrides):
        return "rh_test", "lh_test"


def test_piano_override_merge(tmp_path: Path):
    overrides = {
        "SecA": {
            "piano": {
                "weak_beat_style_rh": "rest",
                "fill_on_4th": True,
                "fill_length_beats": 0.5,
            }
        }
    }
    ov_path = tmp_path / "ov.json"
    ov_path.write_text(json.dumps(overrides))
    ov_model = load_overrides(ov_path)

    patterns = {
        "rh_test": {
            "pattern": [
                {"offset": 0, "duration": 1, "type": "chord"},
                {"offset": 1, "duration": 1, "type": "chord"},
                {"offset": 2, "duration": 1, "type": "chord"},
                {"offset": 3, "duration": 1, "type": "chord"},
            ],
            "length_beats": 4.0,
        },
        "lh_test": {
            "pattern": [{"offset": 0, "duration": 4, "type": "root"}],
            "length_beats": 4.0,
        },
    }

    gen = SimplePiano(
        part_name="piano",
        part_parameters=patterns,
        default_instrument=instrument.Piano(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg={},
        rng=__import__("random").Random(0),
    )

    section = {
        "section_name": "SecA",
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "chord_symbol_for_voicing": "C",
        "musical_intent": {},
        "part_params": {},
    }

    parts = gen.compose(section_data=section, overrides_root=ov_model)
    rh_part = parts["piano_rh"]
    notes = list(rh_part.flatten().notes)
    shift = float(round(notes[0].offset, 2))
    expected = sorted({float(round(0 + shift, 2)), float(round(2 + shift, 2)), float(round(3.5 + shift, 2))})
    offsets = sorted({float(round(float(n.offset), 2)) for n in notes})
    assert offsets == expected
