import copy
from music21 import note, instrument
from generator.bass_generator import BassGenerator


def make_gen():
    return BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        mirror_melody=True,
        main_cfg={"global_settings": {"key_tonic": "C", "key_mode": "major"}},
    )


def test_mirror_melody_simple():
    gen = make_gen()
    vocal = [note.Note("C4", quarterLength=1.0), note.Note("D4", quarterLength=1.0), note.Note("E4", quarterLength=1.0)]
    for i, n in enumerate(vocal):
        n.offset = i
    section = {
        "section_name": "Bridge",
        "absolute_offset": 0.0,
        "q_length": 3.0,
        "chord_symbol_for_voicing": "C",
        "vocal_notes": vocal,
        "part_params": {},
        "musical_intent": {},
        "tonic_of_section": "C",
        "mode": "major",
    }
    part = gen.compose(section_data=section)
    notes = part.flatten().notes
    assert [n.pitch.nameWithOctave for n in notes[:3]] == ["C2", "B1", "A1"]
