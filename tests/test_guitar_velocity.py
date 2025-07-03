import yaml
import pytest
from music21 import instrument, harmony
from generator.guitar_generator import GuitarGenerator

@pytest.fixture
def preset_gens(tmp_path):
    data = {
        "standard": {
            "default": [40, 50, 65, 80, 96, 112, 124],
            "power_chord": [45, 55, 70, 85, 100, 118, 127],
        },
        "drop_d": {
            "default": [38, 48, 60, 75, 92, 108, 122],
        },
    }
    f = tmp_path / "presets.yml"
    f.write_text(yaml.safe_dump(data))
    gen_std = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        velocity_preset_path=str(f),
    )
    gen_drop = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        tuning="drop_d",
        velocity_preset_path=str(f),
    )
    return gen_std, gen_drop, f

@pytest.mark.velocity
def test_curve_value_by_tuning(preset_gens):
    gen_std, gen_drop, _ = preset_gens
    assert gen_std.velocity_presets["standard"]["default"][4] > gen_drop.velocity_presets["drop_d"]["default"][4]

@pytest.mark.velocity
def test_curve_applied_to_event(preset_gens):
    gen_std, _, _ = preset_gens
    notes = gen_std._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": "block_chord"},
        {},
        1.0,
        80,
    )
    assert notes[0].volume.velocity == gen_std.default_velocity_curve[0]

@pytest.mark.velocity
def test_fallback_curve(tmp_path):
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        velocity_preset_path=str(tmp_path / "missing.yml"),
    )
    gen.part_parameters["qpat"] = {
        "pattern": [
            {"offset": 0.0, "duration": 1.0},
            {"offset": 1.0, "duration": 1.0},
            {"offset": 2.0, "duration": 1.0},
            {"offset": 3.0, "duration": 1.0},
        ],
        "reference_duration_ql": 4.0,
    }
    sec = {
        "section_name": "A",
        "q_length": 4.0,
        "humanized_duration_beats": 4.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "qpat"}},
        "musical_intent": {},
        "shared_tracks": {},
    }
    part = gen.compose(section_data=sec)
    vels = [n.volume.velocity for n in part.flatten().notes]
    assert all(0 <= v <= 127 for v in vels)
    assert vels == sorted(vels)
