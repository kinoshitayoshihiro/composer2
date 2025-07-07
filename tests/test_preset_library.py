import copy
import json

import yaml
from music21 import stream

from utilities.mix_profile import export_mix_json
from utilities.tone_shaper import PRESET_LIBRARY, ToneShaper


def test_load_presets_override(tmp_path):
    original = copy.deepcopy(PRESET_LIBRARY)
    cfg = tmp_path / "presets.yml"
    data = {"clean": {"ir_file": str(tmp_path / "c.wav"), "gain_db": -2, "cc": {31: 10}}}
    cfg.write_text(yaml.safe_dump(data))
    try:
        ToneShaper.load_presets(cfg)
        assert PRESET_LIBRARY["clean"]["cc_map"][31] == 10
    finally:
        PRESET_LIBRARY.clear()
        PRESET_LIBRARY.update(original)


def test_load_presets_and_style_hint(tmp_path):
    original = copy.deepcopy(PRESET_LIBRARY)
    cfg = tmp_path / "presets.yml"
    data = {"funk": {"cc_map": {31: 40}}}
    cfg.write_text(yaml.safe_dump(data))
    try:
        ToneShaper.load_presets(cfg)
        ts = ToneShaper()
        preset = ts.choose_preset(style="funk", avg_velocity=70)
        assert preset == "funk"
    finally:
        PRESET_LIBRARY.clear()
        PRESET_LIBRARY.update(original)


def test_choose_preset_high_intensity():
    ts = ToneShaper()
    preset = ts.choose_preset(intensity="high", avg_velocity=90)
    assert preset == "fuzz"


def test_export_mix_json_library_ir(tmp_path):
    original = copy.deepcopy(PRESET_LIBRARY)
    ir_path = tmp_path / "clean.wav"
    ir_path.write_text("dummy")
    PRESET_LIBRARY["clean"]["ir_file"] = str(ir_path)
    try:
        part = stream.Part()
        part.id = "p"
        ts = ToneShaper()
        ts.choose_preset(intensity="low", avg_velocity=40)
        part.tone_shaper = ts
        out = tmp_path / "mix.json"
        export_mix_json(part, out)
        data = json.loads(out.read_text())
        entry = data["p"]
        assert entry["preset"] == "clean"
        assert entry["ir_file"] == str(ir_path)
    finally:
        PRESET_LIBRARY.clear()
        PRESET_LIBRARY.update(original)


def test_load_presets_env_path(tmp_path, monkeypatch):
    original = copy.deepcopy(PRESET_LIBRARY)
    cfg = tmp_path / "envpresets.yml"
    data = {"env_clean": {"cc_map": {31: 64}}}
    cfg.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("PRESET_LIBRARY_PATH", str(cfg))
    try:
        ToneShaper.load_presets(None)
        assert PRESET_LIBRARY["env_clean"]["cc_map"][31] == 64
    finally:
        monkeypatch.delenv("PRESET_LIBRARY_PATH", raising=False)
        PRESET_LIBRARY.clear()
        PRESET_LIBRARY.update(original)
