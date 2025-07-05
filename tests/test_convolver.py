import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")

from utilities.convolver import render_with_ir


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
    assert np.allclose(data[:, 0], sine, atol=1e-6)
