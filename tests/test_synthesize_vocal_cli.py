import json
import builtins
import importlib
from pathlib import Path

import pytest


def _setup_files(tmp_path: Path):
    midi = tmp_path / "test.mid"
    midi.write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xFF/\x00")
    phon = tmp_path / "phon.json"
    phon.write_text(json.dumps(["a"]))
    return midi, phon


def test_cli_success(tmp_path, monkeypatch):
    midi, phon = _setup_files(tmp_path)
    out_dir = tmp_path / "out"
    model = tmp_path / "model.onnx"

    calls = {}

    def fake_onnx(model_path, midi_path, phonemes):
        calls["args"] = (Path(model_path), Path(midi_path), phonemes)
        return b"OK"

    from generator import vocal_generator

    monkeypatch.setattr(vocal_generator, "synthesize_with_onnx", fake_onnx)
    from scripts import synthesize_vocal
    importlib.reload(synthesize_vocal)

    out = synthesize_vocal.main([
        "--mid",
        str(midi),
        "--phonemes",
        str(phon),
        "--out",
        str(out_dir),
        "--onnx-model",
        str(model),
    ])

    out_file = out_dir / "test.wav"
    assert out_file.read_bytes() == b"OK"
    assert out == out_file
    assert calls["args"] == (model, midi, ["a"])


def test_cli_error(monkeypatch, tmp_path):
    midi, phon = _setup_files(tmp_path)
    out_dir = tmp_path / "out"
    model = tmp_path / "model.onnx"

    def fake_onnx(*a, **k):
        raise RuntimeError("boom")

    from generator import vocal_generator

    monkeypatch.setattr(vocal_generator, "synthesize_with_onnx", fake_onnx)
    from scripts import synthesize_vocal
    importlib.reload(synthesize_vocal)

    with pytest.raises(SystemExit) as exc:
        synthesize_vocal.main([
            "--mid",
            str(midi),
            "--phonemes",
            str(phon),
            "--out",
            str(out_dir),
            "--onnx-model",
            str(model),
        ])
    assert exc.value.code == 1


def test_cli_import_error(monkeypatch, tmp_path):
    midi, phon = _setup_files(tmp_path)
    out_dir = tmp_path / "out"

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tts_model":
            raise ImportError("missing")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from scripts import synthesize_vocal
    importlib.reload(synthesize_vocal)

    with pytest.raises(SystemExit) as exc:
        synthesize_vocal.main([
            "--mid",
            str(midi),
            "--phonemes",
            str(phon),
            "--out",
            str(out_dir),
        ])
    assert exc.value.code == 1


def test_cli_default_tts(tmp_path, monkeypatch):
    # 準備
    midi = tmp_path / "m.mid"
    midi.write_bytes(b"MThd")
    phon = tmp_path / "p.json"
    phon.write_text(json.dumps(["a"]))
    out = tmp_path / "out"
    calls = {}

    # tts_model.synthesize をモック
    def fake_synth(m, p):
        calls['args'] = (Path(m), p)
        return b"TTS"

    import sys, types
    monkeypatch.setitem(sys.modules, 'tts_model', types.SimpleNamespace(synthesize=fake_synth))

    # 再ロードして実行
    from scripts import synthesize_vocal
    import importlib
    importlib.reload(synthesize_vocal)

    result = synthesize_vocal.main([
        "--mid",
        str(midi),
        "--phonemes",
        str(phon),
        "--out",
        str(out),
    ])

    # アサート
    out_file = out / "m.wav"
    assert out_file.read_bytes() == b"TTS"
    assert calls['args'] == (midi, ["a"])
    assert result == out_file
