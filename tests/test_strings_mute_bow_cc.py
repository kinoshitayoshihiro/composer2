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
BowPosition = strings_module.BowPosition


def _basic_section():
    return {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
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


def test_mute_and_bow_cc():
    gen = _gen()
    sec = _basic_section()
    sec["style_params"] = {"mute": True}
    sec["events"] = [{"duration": 1.0, "bow_position": "sul pont."}]
    parts = gen.compose(section_data=sec)
    cc_vals = {(e["cc"], e["val"]) for e in parts["violin_i"].extra_cc}
    assert (20, 127) in cc_vals
    assert any(e["cc"] == 71 and e["val"] >= 99 for e in parts["violin_i"].extra_cc)
