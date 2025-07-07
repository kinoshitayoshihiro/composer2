import json
import sys
import types
from pathlib import Path

def test_synthesize_cli(tmp_path, monkeypatch):
    midi = tmp_path / "test.mid"
    midi.write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xFF/\x00")
    phon = tmp_path / "phon.json"
    phon.write_text(json.dumps(["H", "e", "l", "l", "o"]))
    out_dir = tmp_path / "out"

    calls = {}
    def fake_synth(m, p):
        calls['args'] = (Path(m), p)
        return b"Hello"

    monkeypatch.setitem(sys.modules, 'tts_model', types.SimpleNamespace(synthesize=fake_synth))
    import importlib
    from scripts import synthesize_vocal
    importlib.reload(synthesize_vocal)
    synthesize_vocal.main(["--mid", str(midi), "--phonemes", str(phon), "--out", str(out_dir)])

    out_file = out_dir / "test.wav"
    assert out_file.read_bytes() == b"Hello"
    assert calls['args'] == (midi, ["H", "e", "l", "l", "o"])
