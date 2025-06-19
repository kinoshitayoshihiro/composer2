import pytest
from music21 import instrument
from generator.bass_generator import BassGenerator


def make_gen():
    return BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg={"global_settings": {"key_tonic": "C", "key_mode": "major"}},
    )


def test_iiv_root_motion():
    gen = make_gen()
    section = {
        "section_name": "Bridge",
        "absolute_offset": 0.0,
        "q_length": 12.0,
        "chord_symbol_for_voicing": "C",
        "musical_intent": {},
        "part_params": {"bass": {"rhythm_key": "basic_chord_tone_quarters", "options": {"approach_style_on_4th": "subdom_dom"}}},
        "tonic_of_section": "C",
        "mode": "major",
    }
    next_section = {
        "section_name": "Outro",
        "chord_symbol_for_voicing": "C",
        "q_length": 4.0,
    }
    part = gen.compose(section_data=section, next_section_data=next_section)
    notes = list(part.flatten().notes)
    starts = []
    for m in range(1,3):
        target = m * 4.0
        for n in notes:
            if abs(n.offset - target) < 0.01:
                starts.append(n.pitch.name)
                break
    last_note = max(notes, key=lambda n: n.offset)
    assert starts == ["D", "G"]
    assert last_note.pitch.name == "D"

