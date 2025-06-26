import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from utilities.consonant_extract import (
    EssentiaUnavailable,
    detect_consonant_peaks,
)

np.random.seed(0)


def synth_voice(peaks: list[float], sr: int = 16000, length: float = 1.0) -> np.ndarray:
    """Return synthetic signal with short noise bursts at ``peaks`` seconds."""
    n = int(length * sr)
    sig = np.zeros(n)
    for t in peaks:
        idx = int(t * sr)
        if 0 <= idx < n:
            sig[idx: idx + int(0.01 * sr)] += np.random.randn(int(0.01 * sr)) * 0.5
    return sig


def write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((data * 32767).astype(np.int16).tobytes())


@pytest.mark.parametrize("peaks", [[0.2, 0.6, 0.8]])
def test_detect_consonant_peaks(peaks):
    sig = synth_voice(peaks)
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "test.wav"
        write_wav(wav_path, sig)
        detected = detect_consonant_peaks(wav_path, thr=0.2)
        assert len(detected) >= len(peaks)


def test_cli_outputs_json(tmp_path: Path) -> None:
    wav = tmp_path / "v.wav"
    sig = synth_voice([0.1, 0.5])
    write_wav(wav, sig)
    out_json = tmp_path / "p.json"
    subprocess.run(
        ["python", "-m", "modular_composer.cli", "peaks", str(wav), "-o", str(out_json)],
        check=True,
    )
    assert out_json.exists()
    with out_json.open() as fh:
        data = json.load(fh)
    assert data == sorted(data)


@pytest.mark.ess_optional
def test_essentia_backend(tmp_path: Path) -> None:
    wav = tmp_path / "v.wav"
    sig = synth_voice([0.1, 0.5])
    write_wav(wav, sig)
    try:
        peaks = detect_consonant_peaks(wav, thr=0.2, algo="essentia")
    except EssentiaUnavailable:
        pytest.skip("Essentia not installed")
    assert len(peaks) >= 2
