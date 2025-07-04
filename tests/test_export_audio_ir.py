import types
from pathlib import Path

import music21.stream as stream

from utilities import synth


def test_export_audio_uses_metadata(tmp_path, monkeypatch):
    midi = tmp_path / "in.mid"
    wav = tmp_path / "out.wav"
    midi.write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x60MTrk\x00\x00\x00\x04\x00\xFF\x2F\x00")
    from music21 import metadata as m21metadata

    part = stream.Part()
    part.metadata = m21metadata.Metadata()
    part.metadata.ir_file = "dummy_ir.wav"

    calls = {}

    def fake_render_midi(mp, ow, sf2_path=None):
        Path(ow).touch()
        calls["render"] = True
        return Path(ow)

    def fake_render_with_ir(wav_path, ir_path, wav_out):
        calls["ir"] = ir_path
        return Path(wav_out)

    import sys

    monkeypatch.setattr(synth, "render_midi", fake_render_midi)
    if "utilities.convolver" in sys.modules:
        monkeypatch.setattr(sys.modules["utilities.convolver"], "render_with_ir", fake_render_with_ir)
    else:
        monkeypatch.setitem(sys.modules, "utilities.convolver", types.SimpleNamespace(render_with_ir=fake_render_with_ir))

    synth.export_audio(midi, wav, part=part)
    assert calls.get("ir") == "dummy_ir.wav"

