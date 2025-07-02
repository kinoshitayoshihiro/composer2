import yaml
from pathlib import Path
from music21 import instrument, harmony
from generator.guitar_generator import GuitarGenerator
from generator.guitar_generator import EXEC_STYLE_BLOCK_CHORD


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
    data = {"extra_pattern": {"pattern": [{"offset": 0.0, "duration": 1.0}]}}
    file = tmp_path / "patterns.yml"
    file.write_text(yaml.safe_dump(data))

    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        external_patterns_path=str(file),
    )
    assert "extra_pattern" in gen.part_parameters


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


def test_tuning_preset_drop_d():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g3",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        tuning="drop_d",
    )
    assert gen.tuning == [-2, 0, 0, 0, 0, 0]
    gen_std = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g4",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        tuning="standard",
    )
    p_std = gen_std._get_guitar_friendly_voicing(harmony.ChordSymbol("E"), 1)[0]
    p_drop = gen._get_guitar_friendly_voicing(harmony.ChordSymbol("E"), 1)[0]
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


def test_export_tab_xml_and_ascii(tmp_path):
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    gen.compose(section_data=_basic_section())

    xml_path = tmp_path / "out.xml"
    gen.export_tab(str(xml_path), format="xml")
    assert xml_path.exists() and xml_path.stat().st_size > 0

    ascii_path = tmp_path / "out.txt"
    gen.export_tab(str(ascii_path), format="ascii")
    assert ascii_path.exists() and ascii_path.stat().st_size > 0


def test_gate_length_variation_range():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        gate_length_variation=0.2,
    )
    gen.rng.seed(1)
    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {},
        1.0,
        80,
    )
    dur = notes[0].quarterLength
    base = 1.0 * 0.9
    assert base * 0.8 <= dur <= base * 1.2


def test_internal_default_patterns():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    keys = {
        "guitar_rhythm_quarter",
        "guitar_rhythm_syncopation",
        "guitar_rhythm_shuffle",
    }
    assert keys.issubset(gen.part_parameters.keys())

    qpat = gen.part_parameters["guitar_rhythm_quarter"]["pattern"]
    assert qpat[0]["offset"] == 0.0
    assert qpat[1]["offset"] == 1.0

