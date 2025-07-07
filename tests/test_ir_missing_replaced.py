import copy

import yaml

from utilities.tone_shaper import PRESET_LIBRARY, ToneShaper


def test_ir_missing_replaced(tmp_path, caplog):
    original = copy.deepcopy(PRESET_LIBRARY)
    cfg = tmp_path / "presets.yml"
    data = {"x": {"ir_file": str(tmp_path / "missing.wav"), "cc_map": {31: 10}}}
    cfg.write_text(yaml.safe_dump(data))
    try:
        ToneShaper.load_presets(cfg)
        assert PRESET_LIBRARY["x"]["ir_file"] is None
        assert "IR file missing" in caplog.text
    finally:
        PRESET_LIBRARY.clear()
        PRESET_LIBRARY.update(original)

