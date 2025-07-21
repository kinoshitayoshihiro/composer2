from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mixing_assistant.feature_extractor import extract_features
from scipy.signal import lfilter

pytest.importorskip("pyloudnorm")
pytest.importorskip("librosa")
pytest.importorskip("scipy")


@pytest.fixture(scope="session")
def pink_noise(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sr = 22050
    duration = 10
    n = sr * duration
    white = np.random.randn(n)
    b = [0.049922, -0.095993, 0.050612, -0.004408]
    a = [1, -2.494956, 2.017265, -0.522189]
    pink = lfilter(b, a, white)
    dir_path = tmp_path_factory.mktemp("pink")
    audio_path = dir_path / "pink.wav"
    sf.write(audio_path, pink, sr)
    return audio_path


def test_extract_features(pink_noise: Path) -> None:
    audio_path = pink_noise

    feats = extract_features(audio_path)
    keys = [
        "spectral_centroid_mean",
        "spectral_flatness_db",
        "spectral_rolloff95",
        "zero_cross_rate_mean",
        "loudness_i",
        "loudness_range",
        "crest_factor",
    ]
    for key in keys:
        assert key in feats
        assert math.isfinite(feats[key])
    assert math.isfinite(feats["spectral_rolloff95"]) and math.isfinite(
        feats["zero_cross_rate_mean"]
    )
