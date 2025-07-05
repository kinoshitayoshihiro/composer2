import json
from pathlib import Path
from music21 import stream

from utilities.mix_profile import export_mix_json
from utilities.tone_shaper import ToneShaper


def test_export_mix_json(tmp_path):
    part = stream.Part()
    part.id = "g"
    part.extra_cc = [{"time": 0.0, "cc": 91, "val": 80}]
    from music21 import metadata
    part.metadata = metadata.Metadata()
    part.metadata.ir_file = tmp_path / "ir.wav"
    (tmp_path / "ir.wav").write_text("dummy")
    part.tone_shaper = ToneShaper({"clean": {"amp": 20}})
    part.tone_shaper.fx_envelope = part.extra_cc
    out = tmp_path/"mix.json"
    export_mix_json(part, out)
    data = json.loads(out.read_text())
    entry = data["g"]
    assert set(entry).issuperset({"extra_cc", "ir_file", "preset", "fx_cc"})
    assert entry["ir_file"] == str(tmp_path / "ir.wav")
    assert "fx_envelope" not in entry
    times = [e["time"] for e in entry["fx_cc"]]
    assert times == sorted(times)


def test_export_mix_json_fx_env(tmp_path):
    part = stream.Part()
    part.id = "x"
    from music21 import metadata
    part.metadata = metadata.Metadata()
    part.metadata.ir_file = tmp_path / "ir.wav"
    part.metadata.fx_envelope = {"0": {"cc": 91}}
    (tmp_path / "ir.wav").write_text("dummy")
    out = tmp_path / "mix2.json"
    export_mix_json(part, out)
    data = json.loads(out.read_text())
    assert data["x"]["fx_envelope"] == {"0": {"cc": 91}}
