from music21 import instrument
from generator.bass_generator import BassGenerator


def test_unknown_emotion_fallback():
    gen = BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        emotion_profile_path="data/emotion_profile.yaml",
    )
    part = gen.render_part(emotion="mystery", key_signature="C", tempo_bpm=120)
    pitches = [n.pitch.midi for n in part.notes]
    assert pitches == [48, 55, 48, 55]
    assert len(pitches) == 4
