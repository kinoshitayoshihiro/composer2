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
