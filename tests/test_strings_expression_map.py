import importlib.util
import sys
import types
from pathlib import Path
from music21 import instrument, articulations

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("generator")
pkg.__path__ = [str(ROOT / "generator")]
sys.modules.setdefault("generator", pkg)

_MOD_PATH = ROOT / "generator" / "strings_generator.py"
spec = importlib.util.spec_from_file_location("generator.strings_generator", _MOD_PATH)
strings_module = importlib.util.module_from_spec(spec)
sys.modules["generator.strings_generator"] = strings_module
spec.loader.exec_module(strings_module)
StringsGenerator = strings_module.StringsGenerator


def _basic_section():
    return {
        "section_name": "A",
        "q_length": 2.0,
        "humanized_duration_beats": 2.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {},
        "musical_intent": {},
        "shared_tracks": {},
    }


def _gen(**kwargs):
    return StringsGenerator(
        global_settings={},
        default_instrument=instrument.Violin(),
        part_name="strings",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        **kwargs,
    )


def test_expression_map_soft_legato():
    gen = _gen(expression_maps_path=str(ROOT / "tests" / "data" / "ex_maps.yml"))
    sec = _basic_section()
    sec["part_params"] = {"strings": {"expression_map": "gentle_legato"}}
    parts = gen.compose(section_data=sec)
    cc_vals = [e["val"] for e in parts["violin_i"].extra_cc if e["cc"] == 1]
    assert cc_vals and cc_vals[0] == 20


def test_emotion_map_selection():
    gen = _gen(expression_maps_path=str(ROOT / "tests" / "data" / "ex_maps.yml"))
    sec = _basic_section()
    sec["musical_intent"] = {"emotion": "default", "intensity": "high"}
    parts = gen.compose(section_data=sec)
    vals = [e["val"] for e in parts["violin_i"].extra_cc if e["cc"] == 1]
    assert vals and vals[0] == 80
