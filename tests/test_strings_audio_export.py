import importlib.util
import sys
import types
from pathlib import Path
import pytest
from music21 import instrument
from utilities.audio_env import has_fluidsynth

pytestmark = pytest.mark.requires_audio
sf = pytest.importorskip("soundfile")
if not has_fluidsynth():
    pytest.skip("fluidsynth missing", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("generator")
pkg.__path__ = [str(ROOT / "generator")]
sys.modules.setdefault("generator", pkg)

spec = importlib.util.spec_from_file_location("generator.strings_generator", ROOT / "generator" / "strings_generator.py")
strings_module = importlib.util.module_from_spec(spec)
sys.modules["generator.strings_generator"] = strings_module
spec.loader.exec_module(strings_module)
StringsGenerator = strings_module.StringsGenerator


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


def test_export_audio(tmp_path):
    gen = _gen()
    sec = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
    }
    parts = gen.compose(section_data=sec)
    score = strings_module.stream.Score()
    for p in parts.values():
        score.insert(0, p)
    out = tmp_path / "out.wav"
    audio = gen.export_audio(score, ir_set="room", outfile=out)
    assert out.is_file()
    data, _ = sf.read(out)
    assert len(data) > 0 and audio.shape[0] > 0
