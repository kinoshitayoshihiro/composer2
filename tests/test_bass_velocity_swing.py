import textwrap
from music21 import instrument
from generator.bass_generator import BassGenerator


def test_velocity_and_swing(tmp_path):
    yaml = textwrap.dedent(
        """
        cool:
          bass_patterns:
            - riff: [1,5,1,5,1,5,1,5]
              velocity: high
              swing: on
          octave_pref: mid
          length_beats: 4
        """
    )
    path = tmp_path / "ep.yaml"
    path.write_text(yaml)
    gen = BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        emotion_profile_path=str(path),
        global_settings={"swing_ratio": 0.1},
    )
    part = gen.render_part(emotion="cool", key_signature="C", tempo_bpm=120)
    velocities = [n.volume.velocity for n in part.notes]
    assert all(100 <= v <= 110 for v in velocities)
    offsets = [n.offset for n in part.notes]
    assert abs(offsets[1] - 0.55) < 1e-3
    assert abs(part.notes[0].duration.quarterLength - 0.45) < 1e-3
