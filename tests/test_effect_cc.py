import pytest
from music21 import instrument
from generator.guitar_generator import GuitarGenerator


@pytest.fixture
def _gen():
    def factory():
        return GuitarGenerator(
            global_settings={},
            default_instrument=instrument.Guitar(),
            part_name="g",
            global_tempo=120,
            global_time_signature="4/4",
            global_key_signature_tonic="C",
            global_key_signature_mode="major",
        )
    return factory


def test_fx_cc_injection(_gen):
    gen = _gen()
    gen.part_parameters["pat"] = {
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }
    sec = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "pat"}},
        "musical_intent": {"intensity": "medium"},
        "fx_params": {"reverb_send": 80, "chorus_send": 60},
    }
    part = gen.compose(section_data=sec)
    ccs = {(e.get("cc"), e.get("val")) for e in part.extra_cc}
    assert (91, 80) in ccs
    assert (93, 60) in ccs
    assert any(e[0] == 31 for e in ccs)
