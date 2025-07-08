import sys
import json
import types
from pathlib import Path
import importlib
from generator.vocal_generator import VocalGenerator


def test_cli_no_articulation(tmp_path, monkeypatch):
    # prepare dummy MIDI & phonemes
    midi = tmp_path / "d.mid"
    midi.write_bytes(b"MThd")
    phon = tmp_path / "p.json"
    phon.write_text(json.dumps(["a"]))
    out = tmp_path / "out"
    # stub tts_model to bypass actual TTS
    monkeypatch.setitem(sys.modules, 'tts_model', types.SimpleNamespace(synthesize=lambda *args: b""))
    # run CLI
    sys_argv = [
        "modcompose",
        "sample",
        "--backend",
        "vocal",
        "--no-enable-articulation",
        "--mid",
        str(midi),
        "--phonemes",
        str(phon),
        "--out",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", sys_argv)
    from scripts import synthesize_vocal
    importlib.reload(synthesize_vocal)
    # check generated part has no events
    gen = VocalGenerator(enable_articulation=False)
    part = gen.compose(
        [{"offset": 0, "pitch": "C4", "length": 1.0, "velocity": 80}],
        processed_chord_stream=[],
        humanize_opt=False,
        lyrics_words=["a"],
    )
    assert not getattr(part, "extra_cc", [])
    assert not getattr(part, "pitch_bends", [])
