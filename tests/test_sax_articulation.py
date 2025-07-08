import pytest
from music21 import instrument, articulations, spanner

from generator.sax_generator import SaxGenerator, BREATH_CC, MOD_CC


def test_cc_events_for_articulations():
    gen = SaxGenerator(
        default_instrument=instrument.AltoSaxophone(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "chord_label": "C",
        "part_params": {},
        "musical_intent": {"emotion": "default", "intensity": "medium"},
        "tonic_of_section": "C",
        "mode": "major",
    }

    part = gen.compose(section_data=section)
    notes = list(part.flatten().notes)
    assert len(notes) >= 2
    notes[0].articulations.append(articulations.Staccato())
    sl = spanner.Slur(notes[1])
    part.insert(notes[1].offset, sl)
    gen._apply_articulation_cc(part)
    events = getattr(part, "extra_cc", [])
    assert any(e["cc"] == MOD_CC for e in events)
    assert any(e["cc"] == BREATH_CC for e in events)


def test_staccato_probability_override():
    gen = SaxGenerator(
        staccato_prob=1.0,
        slur_prob=0.0,
        default_instrument=instrument.AltoSaxophone(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "chord_label": "C",
        "part_params": {},
        "musical_intent": {"emotion": "default", "intensity": "medium"},
        "tonic_of_section": "C",
        "mode": "major",
    }

    part = gen.compose(section_data=section)
    notes = list(part.recurse().notes)
    assert notes and all(
        any(isinstance(a, articulations.Staccato) for a in n.articulations)
        for n in notes
    )
    assert not list(part.recurse().getElementsByClass(spanner.Slur))


def test_velocity_curve_intensity():
    gen = SaxGenerator(
        default_instrument=instrument.AltoSaxophone(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "chord_label": "C",
        "part_params": {},
        "musical_intent": {"emotion": "default", "intensity": "high"},
        "tonic_of_section": "C",
        "mode": "major",
    }

    part = gen.compose(section_data=section)
    cc11 = [e["val"] for e in getattr(part, "extra_cc", []) if e["cc"] == 11]
    assert cc11 == sorted(cc11) and len(cc11) >= 2


def test_vibrato_parameter_clamp():
    gen = SaxGenerator(
        vibrato_depth=500.0,
        vibrato_rate=8.0,
        default_instrument=instrument.AltoSaxophone(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "chord_label": "C",
        "part_params": {},
        "musical_intent": {"emotion": "default", "intensity": "medium"},
        "tonic_of_section": "C",
        "mode": "major",
    }

    part = gen.compose(section_data=section)
    pitch_vals = [e["val"] for e in getattr(part, "extra_cc", []) if e["cc"] == -1]
    assert pitch_vals and all(0 <= v <= 16383 for v in pitch_vals)
