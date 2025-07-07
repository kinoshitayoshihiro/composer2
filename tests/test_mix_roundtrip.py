import json

from music21 import metadata, stream

from utilities.mix_profile import export_mix_json
from utilities.tone_shaper import ToneShaper


def test_mix_roundtrip(tmp_path):
    part = stream.Part()
    part.id = "p"
    part.metadata = metadata.Metadata()
    ts = ToneShaper()
    ts.choose_preset(intensity="low", avg_velocity=40)
    ir_path = tmp_path / "clean.wav"
    ir_path.write_text("x")
    ts.ir_map[ts._selected] = ir_path
    part.tone_shaper = ts
    part.metadata.ir_file = ir_path
    out = tmp_path / "mix.json"
    export_mix_json(part, out)

    loaded = json.loads(out.read_text())
    entry = loaded["p"]
    assert entry["preset"] == ts._selected
    assert entry["ir_file"] == str(part.metadata.ir_file)


