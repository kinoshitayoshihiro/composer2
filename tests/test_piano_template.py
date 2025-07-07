import pytest
from music21 import articulations, instrument, spanner

from generator.piano_template_generator import PianoTemplateGenerator


def make_gen():
    return PianoTemplateGenerator(
        part_name="piano",
        default_instrument=instrument.Piano(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )


def test_piano_template_basic():
    gen = make_gen()
    section = {
        "q_length": 4.0,
        "chord_symbol_for_voicing": "C",
        "groove_kicks": [],
        "musical_intent": {},
    }
    parts = gen.compose(section_data=section)
    assert isinstance(parts, dict)
    rh = parts["piano_rh"]
    lh = parts["piano_lh"]
    assert pytest.approx(rh.highestTime, abs=1e-6) == 4.0
    notes = list(rh.flatten().notes) + list(lh.flatten().notes)
    assert len(notes) >= 4


def test_gliss_trill_tags():
    gen = make_gen()
    section = {
        "q_length": 2.0,
        "chord_symbol_for_voicing": "C<gliss><trill>",
        "groove_kicks": [],
        "musical_intent": {},
    }
    parts = gen.compose(section_data=section)
    for p in parts.values():
        assert p.recurse().getElementsByClass(spanner.Glissando)
        assert all(
            any(isinstance(a, articulations.Trill) for a in n.articulations)
            for n in p.flatten().notes
        )
