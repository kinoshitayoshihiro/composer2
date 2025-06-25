from music21 import instrument
from generator.bass_generator import BassGenerator


def make_gen():
    gs = {"bass_range_lo": 30, "bass_range_hi": 72}
    cfg = {"global_settings": {"key_tonic": "C", "key_mode": "major"}}
    return BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg=cfg,
        global_settings=gs,
        part_parameters={},
    )


def test_bass_note_range():
    gen = make_gen()
    section = {
        "section_name": "Verse",
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "chord_symbol_for_voicing": "G#5",
        "part_params": {"bass": {"rhythm_key": "bass_quarter_notes", "velocity": 70}},
        "musical_intent": {},
    }
    part = gen.compose(section_data=section)
    for n in part.flatten().notes:
        assert 30 <= n.pitch.midi <= 72
