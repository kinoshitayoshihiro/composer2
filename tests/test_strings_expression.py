import importlib.util
import sys
import types
from pathlib import Path
from music21 import instrument

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
cc_map = strings_module.cc_map


def _basic_section(length=2.0):
    return {
        "section_name": "A",
        "q_length": length,
        "humanized_duration_beats": length,
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


def test_base_dynamic_mf_vs_p():
    path = ROOT / "tests" / "data" / "expression_map.yml"
    gen = _gen(expression_maps_path=str(path))
    sec = _basic_section()
    sec["musical_intent"] = {"style": "ballad", "intensity": "soft"}
    parts_p = gen.compose(section_data=sec)
    val_p = next(e["val"] for e in parts_p["violin_i"].extra_cc if e["cc"] == cc_map["expression"])
    sec["musical_intent"] = {"style": "dramatic", "intensity": "high"}
    parts_mf = gen.compose(section_data=sec)
    val_mf = next(e["val"] for e in parts_mf["violin_i"].extra_cc if e["cc"] == cc_map["expression"])
    assert val_mf > val_p


def test_crescendo_curve():
    gen = _gen()
    sec = _basic_section(4.0)
    parts = gen.compose(section_data=sec)
    gen.crescendo(parts, 4.0, start_val=40, end_val=90)
    vals = [e["val"] for e in parts["violin_i"].extra_cc if e["cc"] == cc_map["expression"]]
    assert len(vals) >= 3
    assert vals == sorted(vals)


def test_mute_cc_from_map():
    path = ROOT / "tests" / "data" / "expression_map.yml"
    gen = _gen(expression_maps_path=str(path))
    sec = _basic_section()
    sec["musical_intent"] = {"style": "ballad", "intensity": "soft"}
    parts = gen.compose(section_data=sec)
    vals = [e for e in parts["violin_i"].extra_cc if e["cc"] == cc_map["mute_toggle"]]
    assert vals and vals[0]["val"] == 64
