from pathlib import Path

import numpy as np
import soundfile as sf

from utilities.loudness_normalizer import normalize_wav


def test_normalize_wav(tmp_path: Path) -> None:
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 1000 * t)
    inp = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    sf.write(inp, y, sr)
    normalize_wav(inp, out, target_lufs=-20.0)
    y_norm, _ = sf.read(out)
    rms = np.sqrt(np.mean(y_norm ** 2))
    lufs = 20 * np.log10(rms)
    assert abs(lufs - (-20.0)) < 1.0
