from pathlib import Path

import importlib.util
import numpy as np
import soundfile as sf

import pytest

if importlib.util.find_spec("librosa") is None:
    pytest.skip("librosa missing", allow_module_level=True)

from utilities.loudness_normalizer import normalize_wav


def test_normalize_wav(tmp_path: Path) -> None:
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    amp = 10 ** (-3 / 20)
    y = amp * np.sin(2 * np.pi * 1000 * t)
    inp = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    sf.write(inp, y, sr)
    target = -14.0
    normalize_wav(inp, out, target_lufs=target)
    y_norm, _ = sf.read(out)
    # Expect amplitude scaled to roughly 0.28 for -14 LUFS
    max_amp = np.max(np.abs(y_norm))
    assert abs(max_amp - 0.28) < 0.02
