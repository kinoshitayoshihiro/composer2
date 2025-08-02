"""Ensure generators accept contextual parameters.

GitHub CI workflow snippet (Linux):
    - run: sudo apt-get update && sudo apt-get install -y ffmpeg
"""

from music21 import instrument, stream

from generator.piano_generator import PianoGenerator


class SimplePiano(PianoGenerator):
    def _get_pattern_keys(self, musical_intent, overrides):
        return "rh", "lh"


def make_gen() -> SimplePiano:
    patterns = {
        "rh": {
            "pattern": [{"offset": 0, "duration": 1.0, "type": "chord"}],
            "length_beats": 1.0,
        },
        "lh": {
            "pattern": [{"offset": 0, "duration": 1.0, "type": "root"}],
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
        key="C major",
        tempo=120,
        emotion="hope_dawn",
        main_cfg={},
    )


def test_compose_with_params() -> None:
    gen = make_gen()
    section = {
        "section_name": "Verse",
        "chord_symbol_for_voicing": "C",
        "musical_intent": {"emotion": "hope_dawn", "intensity": "medium"},
        "q_length": 4.0,
    }
    part_or_dict = gen.compose(section_data=section)
    if isinstance(part_or_dict, dict):
        part = next(iter(part_or_dict.values()))
    else:
        part = part_or_dict
    assert part.flatten().notes, "Expected generated part to contain notes"
