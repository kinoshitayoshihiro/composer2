import logging
from pathlib import Path
from utilities.effect_preset_loader import EffectPresetLoader


def test_reload(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    cfg = tmp_path / "fx.yml"
    cfg.write_text("hall:\n  ir_file: irs/hall.wav\n")
    EffectPresetLoader.load(str(cfg))
    assert EffectPresetLoader.get("hall")["ir_file"] == "irs/hall.wav"
    cfg.write_text("hall:\n  ir_file: irs/room.wav\n")
    EffectPresetLoader.reload()
    assert EffectPresetLoader.get("hall")["ir_file"] == "irs/room.wav"
    EffectPresetLoader.get("missing")
    assert any("missing" in r.message for r in caplog.records)
