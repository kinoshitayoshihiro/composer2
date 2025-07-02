from pathlib import Path
from utilities import preset_manager


def test_preset_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(preset_manager, "PRESET_DIR", tmp_path)
    cfg = {"bars": 4, "swing": 0.5}
    preset_manager.save_preset("demo", cfg)
    assert "demo" in preset_manager.list_presets()
    loaded = preset_manager.load_preset("demo")
    assert loaded == cfg
