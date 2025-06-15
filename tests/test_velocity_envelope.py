from music21 import instrument
from generator.piano_generator import PianoGenerator
from utilities.override_loader import load_overrides


class SimplePiano(PianoGenerator):
    def _get_pattern_keys(self, musical_intent, overrides):
        return "rh_test", "lh_test"


def make_gen(rng=None):
    patterns = {
        "rh_test": {
            "pattern": [
                {"offset": i * 0.5, "duration": 0.5, "type": "chord"} for i in range(8)
            ],
            "length_beats": 4.0,
            "velocity": 60,
        },
        "lh_test": {
            "pattern": [{"offset": 0, "duration": 4.0, "type": "root"}],
            "length_beats": 4.0,
            "velocity": 60,
        },
    }
    return SimplePiano(
        part_name="piano",
        part_parameters=patterns,
        default_instrument=instrument.Piano(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg={},
        rng=rng,
    )


def test_intensity_velocity_scale():
    gen = make_gen()
    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "chord_symbol_for_voicing": "C",
        "musical_intent": {"intensity": "high"},
        "part_params": {"piano": {"velocity": 60}},
    }
    parts = gen.compose(section_data=section)
    vels = [n.volume.velocity for n in parts["piano_rh"].flatten().notes]
    avg = sum(vels) / len(vels)
    assert 65 <= avg <= 67


def test_swing_ratio_override(tmp_path):
    path = tmp_path / "ov.yml"
    path.write_text("Sec:\n  piano:\n    swing_ratio: 0.65\n")
    overrides = load_overrides(path)
    gen = make_gen()
    section = {
        "section_name": "Sec",
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "chord_symbol_for_voicing": "C",
        "musical_intent": {},
        "part_params": {"piano": {"velocity": 60}},
    }
    parts = gen.compose(section_data=section, overrides_root=overrides)
    offsets = {round(float(n.offset % 0.5), 2) for n in parts["piano_rh"].flatten().notes}
    # Expect downbeat notes unchanged and off-beats shifted, producing at least two distinct values
    assert any(abs(o - 0.0) < 0.05 for o in offsets)
    assert len(offsets) >= 2
