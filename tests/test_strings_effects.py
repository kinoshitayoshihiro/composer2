import importlib.util
import sys
import types
from pathlib import Path
from music21 import instrument

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("generator")
pkg.__path__ = [str(ROOT / "generator")]
sys.modules.setdefault("generator", pkg)

spec = importlib.util.spec_from_file_location("generator.strings_generator", ROOT / "generator" / "strings_generator.py")
strings_module = importlib.util.module_from_spec(spec)
sys.modules["generator.strings_generator"] = strings_module
spec.loader.exec_module(strings_module)
StringsGenerator = strings_module.StringsGenerator
EffectPresetLoader = strings_module.EffectPresetLoader


def _gen():
    return StringsGenerator(
        global_settings={},
        default_instrument=instrument.Violin(),
        part_name="strings",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )


def test_apply_effect_preset(tmp_path):
    cfg = tmp_path / "fx.yml"
    cfg.write_text("hall:\n  ir_file: irs/hall.wav\n  cc:\n    91: 80\n")
    EffectPresetLoader.load(str(cfg))

    gen = _gen()
    sec = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
    }
    parts = gen.compose(section_data=sec)
    part = parts["violin_i"]
    gen.apply_effect_preset(part, "hall")
    assert getattr(part.metadata, "ir_file", None) == "irs/hall.wav"
    assert any(e.get("cc") == 91 and e.get("val") == 80 for e in part.extra_cc)
