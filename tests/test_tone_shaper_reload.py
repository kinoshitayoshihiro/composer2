import copy

import yaml

from utilities.tone_shaper import PRESET_LIBRARY, ToneShaper


def test_reload_presets(tmp_path):
    original = copy.deepcopy(PRESET_LIBRARY)
    cfg = tmp_path / "presets.yml"
    data = {"jam": {"cc_map": {31: 10}}}
    cfg.write_text(yaml.safe_dump(data))
    try:
        ToneShaper.load_presets(cfg)
        ts = ToneShaper()
        assert ts.choose_preset(style="jam", avg_velocity=70) == "jam"
        data["jam"]["cc_map"][31] = 50
        cfg.write_text(yaml.safe_dump(data))
        ts.reload_presets()
        assert ts.preset_map["jam"][31] == 50
    finally:
        PRESET_LIBRARY.clear()
        PRESET_LIBRARY.update(original)
