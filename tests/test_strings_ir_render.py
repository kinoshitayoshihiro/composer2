import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

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
        part_name="strings",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )


def _basic_score(gen):
    sec = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
    }
    parts = gen.compose(section_data=sec)
    sc = strings_module.stream.Score()
    for p in parts.values():
        sc.insert(0, p)
    return sc


@pytest.mark.requires_audio
def test_room_norm(tmp_path):
    gen = _gen()
    score = _basic_score(gen)
    audio = gen.export_audio(score, ir_set="room", outfile=None)
    assert audio.shape[0] > 0
    assert np.max(np.abs(audio)) == 1.0


@pytest.mark.requires_audio
def test_hall_outfile(tmp_path):
    gen = _gen()
    score = _basic_score(gen)
    out = tmp_path / "out.wav"
    audio = gen.export_audio(score, ir_set="hall", outfile=out)
    assert out.exists()
    info = sf.info(out)
    assert info.samplerate == 48000
    assert audio.shape[0] > 0
