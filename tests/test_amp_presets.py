from utilities.tone_shaper import ToneShaper


def test_amp_preset_loading():
    ts = ToneShaper.from_yaml("data/amp_presets.yml")
    assert ts.preset_map["drive"]["amp"] == 90
    assert ts.preset_map["drive"]["reverb"] == 60
    assert ts.ir_map["crunch"].endswith(".wav")


def test_choose_preset_priority():
    ts = ToneShaper({"clean": {"amp": 20}, "drive": {"amp": 90}}, {})
    assert ts.choose_preset("drive", None, 70) == "drive"
    assert ts.choose_preset(None, "high", 110) == "drive"
    assert ts.choose_preset(None, "low", 40) == "clean"


def test_cc_event_generation():
    ts = ToneShaper({"clean": {"amp": 20}}, {})
    ts.choose_preset("clean", None, 80)
    events = ts.to_cc_events()
    assert (0.0, 31, 20) in events
    assert any(e[1] == 93 for e in events)


def test_choose_preset_rules(tmp_path):
    import yaml

    data = {
        "presets": {"clean": 20, "drive": 90},
        "rules": [{"if": "avg_velocity>100", "preset": "drive"}],
    }
    cfg = tmp_path / "amp.yml"
    cfg.write_text(yaml.safe_dump(data))
    ts = ToneShaper.from_yaml(cfg)
    assert ts.choose_preset(None, "low", 120) == "drive"
