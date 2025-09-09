import importlib.util
import sys
import types
from pathlib import Path
from music21 import instrument, note

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


def test_apply_vibrato_range_and_period():
    gen = _gen()
    n = note.Note("C4", quarterLength=1.0)
    gen._apply_vibrato(n, 0.5, 6)
    curve = n.editorial.vibrato_curve
    cents = [c for _, c in curve]
    assert max(cents) <= 50 and min(cents) >= -50
    assert len(curve) >= 5
    assert abs(curve[1][0] - curve[0][0] - 0.1) < 0.01
