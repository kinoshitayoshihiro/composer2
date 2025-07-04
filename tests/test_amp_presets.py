import pytest

from utilities.tone_shaper import ToneShaper


def test_amp_preset_loading():
    ts = ToneShaper.from_yaml("data/amp_presets.yml")
    assert ts.preset_map["drive"]["amp"] == 90
    assert ts.preset_map["drive"]["reverb"] == 60
    assert ts.ir_map["crunch"].suffix == ".wav"


def test_choose_preset_priority():
    ts = ToneShaper({"clean": {"amp": 20}, "drive": {"amp": 90}}, {})
    assert ts.choose_preset("drive", None, 70) == "drive"
    assert ts.choose_preset(None, "high", 110) == "drive"
    assert ts.choose_preset(None, "low", 40) == "clean"


def test_cc_event_generation():
    ts = ToneShaper({"clean": {"amp": 20}}, {})
    events = ts.to_cc_events("clean", "low")
    assert (0.0, 31, 20) in events
    assert any(e[1] == 93 for e in events)


def test_choose_preset_rules(tmp_path):
    import yaml

    data = {
        "presets": {"clean": 20, "drive": 90},
        "rules": [{"if": "avg_velocity>100", "preset": "drive"}],
        "ir": {"clean": "a.wav", "drive": "b.wav"},
    }
    cfg = tmp_path / "amp.yml"
    cfg.write_text(yaml.safe_dump(data))
    ts = ToneShaper.from_yaml(cfg)
    assert ts.choose_preset(None, "low", 120) == "drive"


def test_from_yaml_malformed(tmp_path):
    cfg = tmp_path / "bad.yml"
    cfg.write_text("presets:\n  a: 10\n")
    with pytest.raises(ValueError):
        ToneShaper.from_yaml(cfg)


def test_choose_preset_unknown_intensity():
    ts = ToneShaper({"clean": {"amp": 20}}, {})
    assert ts.choose_preset(None, "extreme", 80) == "clean"


def test_export_audio_with_ir(tmp_path):
    midi = tmp_path / "in.mid"
    wav = tmp_path / "out.wav"
    midi.write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x60MTrk\x00\x00\x00\x04\x00\xFF\x2F\x00")
    ir = tmp_path / "ir.wav"
    import soundfile as sf
    sf.write(ir, [0.0], 44100)
    from utilities import synth
    synth.export_audio(midi, wav, ir_file=ir)
    assert wav.read_bytes().startswith(b"RIFF")
