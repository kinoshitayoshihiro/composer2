import importlib
import types
import sys
from pathlib import Path

import pytest

from generator import vocal_generator


def test_synthesize_with_onnx_success(tmp_path, monkeypatch, caplog):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"stub")
    midi = tmp_path / "test.mid"
    midi.write_bytes(b"MThd")

    calls = {}

    class DummySession:
        def __init__(self, path):
            calls["path"] = Path(path)
        def run(self, outputs, feeds):
            calls["feeds"] = feeds
            return [b"audio"]

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(InferenceSession=DummySession))
    importlib.reload(vocal_generator)

    with caplog.at_level("INFO"):
        result = vocal_generator.synthesize_with_onnx(model, midi, ["a"])
    assert result == b"audio"
    assert calls["path"] == model
    assert "input" in calls["feeds"]
    assert "Starting ONNX synthesis" in caplog.text
    assert "Finished ONNX synthesis" in caplog.text


def test_synthesize_with_onnx_import_error(tmp_path, monkeypatch, caplog):
    if "onnxruntime" in sys.modules:
        del sys.modules["onnxruntime"]
    import builtins
    orig_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("missing")
        return orig_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with caplog.at_level("ERROR"):
        result = vocal_generator.synthesize_with_onnx(tmp_path/"m.onnx", tmp_path/"m.mid", ["a"])
    assert result == b""
    assert "Failed to import onnxruntime" in caplog.text
