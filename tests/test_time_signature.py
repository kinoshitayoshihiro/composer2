from unittest.mock import patch
from music21 import instrument

from generator.bass_generator import BassGenerator


def test_time_signature_parsing():
    gen = BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="7/8",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        part_parameters={},
    )

    section = {
        "section_name": "Intro",
        "absolute_offset": 0.0,
        "q_length": gen.measure_duration,
        "chord_symbol_for_voicing": "C",
        "part_params": {"bass": {"rhythm_key": "root_only", "velocity": 70}},
        "musical_intent": {},
    }

    with patch("generator.base_part_generator.apply_swing") as m_swing:
        gen.compose(section_data=section)
        assert m_swing.call_args[1]["subdiv"] == 8

    assert abs(gen.measure_duration - 3.5) < 1e-6

