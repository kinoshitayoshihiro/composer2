import yaml
from pathlib import Path
from music21 import instrument, harmony
from generator.guitar_generator import GuitarGenerator


def _basic_section():
    return {
        "section_name": "A",
        "q_length": 4.0,
        "humanized_duration_beats": 4.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {},
        "musical_intent": {},
        "shared_tracks": {},
    }


def test_load_external_patterns(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    data = {"extra_pattern": {"pattern": [{"offset": 0.0, "duration": 1.0}]}}
    file = root / "strum_patterns.yml"
    file.write_text(yaml.safe_dump(data))
    try:
        gen = GuitarGenerator(
            global_settings={},
            default_instrument=instrument.Guitar(),
            part_name="guitar",
            global_tempo=120,
            global_time_signature="4/4",
            global_key_signature_tonic="C",
            global_key_signature_mode="major",
        )
        assert "extra_pattern" in gen.part_parameters
    finally:
        file.unlink()


def test_timing_variation_jitter():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        timing_variation=0.05,
    )
    gen.rng.seed(0)
    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": "block_chord"},
        {},
        1.0,
        80,
    )
    assert notes[0].offset != 0.0


def test_custom_tuning_applied():
    gen_std = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g1",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        tuning=[0, 0, 0, 0, 0, 0],
    )
    gen_drop = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g2",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        tuning=[-2, 0, 0, 0, 0, 0],
    )
    p_std = gen_std._get_guitar_friendly_voicing(harmony.ChordSymbol("E"), 1)[0]
    p_drop = gen_drop._get_guitar_friendly_voicing(harmony.ChordSymbol("E"), 1)[0]
    assert int(p_drop.ps - p_std.ps) == -2


def test_export_musicxml(tmp_path):
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    part = gen.compose(section_data=_basic_section())
    path = tmp_path / "out.xml"
    gen.export_musicxml(str(path))
    assert path.exists() and path.stat().st_size > 0
