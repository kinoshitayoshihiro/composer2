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
load_cc_map = strings_module.load_cc_map


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


def test_unknown_emotion_fallback(tmp_path):
    gen = _gen(expression_maps_path=str(ROOT / "tests" / "data" / "ex_maps.yml"))
    sec = _basic_section()
    sec["musical_intent"] = {"emotion": "mystery", "intensity": "extreme"}
    parts = gen.compose(section_data=sec)
    vals = [e["val"] for e in parts["violin_i"].extra_cc if e["cc"] == 1]
    assert vals and vals[0] == 20


def test_custom_cc_map_override(tmp_path):
    path = tmp_path / "cc.yml"
    path.write_text("mute_toggle: 21\n")
    load_cc_map(str(path))
    gen = _gen()
    sec = _basic_section(1.0)
    sec["style_params"] = {"mute": True}
    parts = gen.compose(section_data=sec)
    assert any(e["cc"] == 21 for e in parts["violin_i"].extra_cc)


def test_crescendo_long_section():
    gen = _gen()
    sec = _basic_section(8.0)
    parts = gen.compose(section_data=sec)
    gen.crescendo(parts, 8.0, start_val=30, end_val=90)
    vals = [e["val"] for e in parts["violin_i"].extra_cc if e["cc"] == cc_map["expression"]]
    assert vals[0] == 30 and vals[-1] == 90
    assert vals == sorted(vals)
    assert len(vals) > 2
