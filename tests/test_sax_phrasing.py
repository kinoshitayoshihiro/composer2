import pytest
from music21 import instrument

from generator.sax_generator import SaxGenerator, DEFAULT_PHRASE_PATTERNS
from utilities.scale_registry import ScaleRegistry


def test_reference_duration_is_two_bars():
    for pat in DEFAULT_PHRASE_PATTERNS.values():
        assert pytest.approx(pat["reference_duration_ql"]) == 8.0


def test_generated_phrase_in_scale_and_length():
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
        "q_length": 8.0,
        "chord_label": "C",
        "part_params": {},
        "musical_intent": {"emotion": "default", "intensity": "medium"},
        "tonic_of_section": "C",
        "mode": "major",
    }

    part = gen.compose(section_data=section)
    notes = list(part.flatten().notes)
    scale_pcs = {p.pitchClass for p in ScaleRegistry.get("C", "major").getPitches("C3", "C5")}
    assert notes
    assert notes[-1].offset + notes[-1].quarterLength <= 8.0
    assert all(n.pitch.pitchClass in scale_pcs for n in notes)


def test_seed_determinism():
    gen = SaxGenerator(
        seed=42,
        default_instrument=instrument.AltoSaxophone(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 8.0,
        "chord_label": "C",
        "part_params": {},
        "musical_intent": {"emotion": "default", "intensity": "medium"},
        "tonic_of_section": "C",
        "mode": "major",
    }

    part = gen.compose(section_data=section)
    notes = [n.pitch.nameWithOctave for n in part.flatten().notes]
    expected = [
        "F4",
        "D3",
        "C3",
        "G4",
        "G3",
        "F3",
        "F3",
        "E3",
        "G4",
        "D3",
        "F4",
        "G4",
        "C5",
        "D4",
        "D3",
        "E4",
    ]
    assert notes == expected
