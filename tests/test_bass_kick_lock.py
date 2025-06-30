import copy
from music21 import instrument
from generator.bass_generator import BassGenerator


def make_gen(shift: int):
    patterns = {
        "root_quarters": {
            "pattern_type": "fixed_pattern",
            "pattern": [
                {"offset": i, "duration": 1.0, "type": "root"} for i in range(4)
            ],
            "reference_duration_ql": 4.0,
        },
        "velocity_shift_on_kick": shift,
    }
    return BassGenerator(
        part_name="bass",
        part_parameters=patterns,
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg={"global_settings": {"key_tonic": "C", "key_mode": "major"}},
    )


def test_kick_lock_velocity():
    gen = make_gen(15)
    section = {
        "section_name": "Test",
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "chord_symbol_for_voicing": "C",
        "part_params": {"bass": {"rhythm_key": "root_quarters", "velocity": 60}},
        "musical_intent": {},
    }
    part = gen.compose(section_data=section, shared_tracks={"kick_offsets": [0, 1, 2, 3]})
    velocities = [n.volume.velocity for n in part.flatten().notes]
    assert velocities == [75, 75, 75, 75]


def test_render_part_kick_alignment():
    gen = BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        emotion_profile_path="data/emotion_profile.yaml",
    )
    part = gen.render_part(
        emotion="joy",
        key_signature="C",
        tempo_bpm=120,
        groove_history=[0.0, 1.0, 2.0, 3.0],
    )
    offsets = [n.offset for n in part.notes]
    for expected, actual in zip([0.0, 1.0, 2.0, 3.0], offsets):
        assert abs(expected - actual) <= 1 / 480
