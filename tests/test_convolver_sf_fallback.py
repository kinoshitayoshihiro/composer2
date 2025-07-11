import importlib
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest


def write_wav(path: Path, data: np.ndarray, sr: int = 44100) -> None:
    data = np.asarray(data)
    if data.ndim > 1:
        data = data[:, 0]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((data * 32767).astype(np.int16).tobytes())

def test_soundfile_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_sf = importlib.import_module("soundfile")
    monkeypatch.setitem(sys.modules, "soundfile", types.ModuleType("soundfile"))
    monkeypatch.delattr(sys.modules["soundfile"], "__getattr__", raising=False)
    conv = importlib.reload(importlib.import_module("utilities.convolver"))
    inp = tmp_path / "in.wav"
    irp = tmp_path / "ir.wav"
    write_wav(inp, np.zeros(1))
    write_wav(irp, np.zeros(1))
    out = tmp_path / "out.wav"
    with pytest.warns(RuntimeWarning):
        conv.render_with_ir(inp, irp, out)
    assert out.exists()
    monkeypatch.setitem(sys.modules, "soundfile", real_sf)
    importlib.reload(importlib.import_module("utilities.convolver"))
