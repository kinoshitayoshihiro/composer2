import importlib
import sys

import numpy as np
import pytest

from utilities.convolver import render_with_ir

pytestmark = pytest.mark.requires_audio

sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")


def test_impulse_rms(tmp_path):
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t)
    inp = tmp_path / "in.wav"
    ir = tmp_path / "imp.wav"
    sf.write(inp, sine, sr)
    imp = np.zeros(sr)
    imp[0] = 1.0
    sf.write(ir, imp, sr)
    out = tmp_path / "out.wav"
    render_with_ir(inp, ir, out)
    data, _ = sf.read(out)
    rms = np.sqrt(np.mean(np.square(data)))
    assert abs(rms - np.sqrt(0.5)) < 1e-3


def test_gain_db(tmp_path):
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    inp = tmp_path / "in.wav"
    ir = tmp_path / "imp.wav"
    sf.write(inp, tone, sr)
    imp = np.zeros(sr)
    imp[0] = 1.0
    sf.write(ir, imp, sr)
    out = tmp_path / "out.wav"
    render_with_ir(inp, ir, out, gain_db=6.0)
    data, _ = sf.read(out)
    rms = np.sqrt(np.mean(np.square(data)))
    expected = np.sqrt(np.mean(np.square(tone))) * (10 ** (6.0 / 20.0))
    assert abs(rms - expected) < 1e-3


def test_render_missing_ir(tmp_path, caplog):
    sr = 44100
    sine = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, endpoint=False))
    inp = tmp_path / "in.wav"
    sf.write(inp, sine, sr)
    out = tmp_path / "out.wav"
    render_with_ir(inp, tmp_path / "missing.wav", out)
    data, _ = sf.read(out)
    assert np.allclose(data[:, 0], sine, atol=1e-4)


def test_resample_poly_fallback(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "soxr", None)
    conv = importlib.reload(importlib.import_module("utilities.convolver"))
    called = {"ok": False}

    def fake(data, up, down, axis=0):
        called["ok"] = True
        from scipy.signal import resample_poly as rp

        return rp(data, up, down, axis=axis)

    monkeypatch.setattr(conv, "resample_poly", fake)

    sr = 32000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t)
    inp = tmp_path / "in.wav"
    ir = tmp_path / "imp.wav"
    sf.write(inp, sine, sr)
    sf.write(ir, sine, sr // 2)
    out = tmp_path / "out.wav"
    conv.render_with_ir(inp, ir, out)
    assert called["ok"]


def test_downmix_ir_noop():
    from utilities.convolver import _downmix_ir

    mono = np.ones((10, 1), dtype=np.float32)
    stereo = np.ones((10, 2), dtype=np.float32)

    assert _downmix_ir(mono).shape[1] == 1
    assert _downmix_ir(stereo).shape[1] == 2


def test_load_ir_cache_env(monkeypatch):
    monkeypatch.setenv("CONVOLVER_IR_CACHE", "32")
    conv = importlib.reload(importlib.import_module("utilities.convolver"))
    assert conv.load_ir.cache_info().maxsize == 32
