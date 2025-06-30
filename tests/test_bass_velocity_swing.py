import textwrap
from music21 import instrument
from generator.bass_generator import BassGenerator


def test_velocity_and_swing():
    gen = BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        emotion_profile_path="data/emotion_profile.yaml",
        global_settings={"swing_ratio": 0.1},
    )
    part = gen.render_part(emotion="funky", key_signature="C", tempo_bpm=120)
    velocities = [n.volume.velocity for n in part.notes]
    assert all(100 <= v <= 110 for v in velocities)
    offsets = [n.offset for n in part.notes]
    assert abs(offsets[1] - 1.1) < 1e-6
    assert abs(offsets[3] - 3.1) < 1e-6
