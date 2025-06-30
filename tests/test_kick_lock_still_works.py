from music21 import instrument
from generator.bass_generator import BassGenerator


def test_first_note_aligned_to_kick() -> None:
    gen = BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    section = {
        "emotion": "joy",
        "key_signature": "C",
        "tempo_bpm": 120,
        "chord": "C",
        "melody": [],
        "groove_kicks": [0.05],
    }
    part = gen.render_part(section)
    first_off = part.notes[0].offset
    assert abs(first_off - 0.05) <= 1 / 480

