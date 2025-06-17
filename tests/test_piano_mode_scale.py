import json
from pathlib import Path
from music21 import instrument, stream
from generator.piano_generator import PianoGenerator
from utilities.override_loader import load_overrides


class SimplePiano(PianoGenerator):
    pass


def make_gen(rhythm_lib):
    patterns = {
        k: v.model_dump() if hasattr(v, "model_dump") else dict(v)
        for k, v in (rhythm_lib.piano_patterns or {}).items()
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
        rng=__import__("random").Random(0),
    )


def test_dorian_minor_third_lh(rhythm_library):
    gen = make_gen(rhythm_library)
    section = {
        "section_name": "Verse 1",
        "absolute_offset": 0.0,
        "q_length": 2.0,
        "chord_symbol_for_voicing": "G",
        "musical_intent": {"emotion": "quiet_pain", "intensity": "low"},
        "part_params": {"piano": {"rhythm_key_lh": "piano_tense_ostinato_lh"}},
        "mode": "dorian",
    }
    parts = gen.compose(section_data=section)
    lh_notes = [n for n in parts["piano_lh"].flatten().notes]
    names = {n.pitch.name for n in parts["piano_lh"].flatten().notes}
    assert "B-" in names


def test_override_changes_pattern(tmp_path: Path, rhythm_library):
    overrides = {"Verse 1": {"piano": {"rhythm_key": "piano_lh_roots_half"}}}
    ov_path = tmp_path / "ov.json"
    ov_path.write_text(json.dumps(overrides))
    ov_model = load_overrides(ov_path)

    gen = make_gen(rhythm_library)
    section = {
        "section_name": "Verse 1",
        "absolute_offset": 0.0,
        "q_length": 2.0,
        "chord_symbol_for_voicing": "G",
        "musical_intent": {"emotion": "quiet_pain", "intensity": "low"},
        "part_params": {"piano": {"rhythm_key_lh": "piano_tense_ostinato_lh"}},
        "mode": "dorian",
    }
    parts = gen.compose(section_data=section, overrides_root=ov_model)
    names = {n.pitch.name for n in parts["piano_lh"].flat.notes}
    assert "B-" not in names
