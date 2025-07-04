from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

if importlib.util.find_spec("librosa") is None:
    pytest.skip("librosa missing", allow_module_level=True)

from utilities.peak_extractor import PeakExtractorConfig, extract_peaks


def _make_wav(path: Path) -> None:
    sr = 16000
    dur = 2.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * 440 * t)
    burst = int(0.01 * sr)
    times = np.linspace(0.2, 1.8, 5)
    for tt in times:
        start = int(tt * sr)
        noise = 0.7079 * np.random.randn(burst)
        y[start:start + burst] += noise
    sf.write(path, y, sr)


def test_extract_peaks_accuracy(tmp_path: Path) -> None:
    wav = tmp_path / "v.wav"
    _make_wav(wav)
    cfg = PeakExtractorConfig(sr=16000, frame_length=512, hop_length=128)
    peaks = extract_peaks(wav, cfg)
    expected = np.linspace(0.2, 1.8, 5)
    assert 4 <= len(peaks) <= 6
    for p, e in zip(peaks[:5], expected):
        assert abs(p - e) < 0.015


def test_cli_creates_json(tmp_path: Path) -> None:
    wav = tmp_path / "v.wav"
    _make_wav(wav)
    out_json = tmp_path / "p.json"
    subprocess.run(
        ["python", "-m", "modular_composer.cli", "peaks", str(wav), "-o", str(out_json)],
        check=True,
    )
    assert out_json.exists()
    with out_json.open() as fh:
        data = json.load(fh)
    assert data == sorted(data)

