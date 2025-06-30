from music21 import instrument
from generator.bass_generator import BassGenerator


def make_gen() -> BassGenerator:
    return BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )


def test_mirror_interval_inversion() -> None:
    gen = make_gen()
    melody = [(1.0, 62, 0.5), (2.0, 64, 0.5)]
    section = {
        "emotion": "joy",
        "key_signature": "C",
        "tempo_bpm": 120,
        "chord": "C",
        "melody": melody,
        "groove_kicks": [0.0],
    }
    part = gen.render_part(section)
    pcs = [n.pitch.midi % 12 for n in part.notes[1:3]]
    assert pcs == [10, 8]

