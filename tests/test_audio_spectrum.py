import wave
from pathlib import Path

import numpy as np
import pytest


def fft_magnitude(path: Path) -> np.ndarray:
    with wave.open(str(path), 'rb') as wf:
        data = wf.readframes(wf.getnframes())
        samples = np.frombuffer(data, dtype=np.int16)
    spectrum = np.abs(np.fft.rfft(samples))
    return spectrum / np.max(spectrum)


@pytest.mark.parametrize("style", ["rock_drive_loop", "brush_light_loop"])
def test_spectrum_match(style: str) -> None:
    base = Path("data/golden/wav") / f"{style}.wav"
    curr = Path("tmp") / f"{style}.wav"
    if not base.exists():
        pytest.skip(f"{base} missing; baseline not committed")
    if not curr.exists():
        pytest.skip(f"{curr} not generated; run audio regression first")
    spec_base = fft_magnitude(base)
    spec_curr = fft_magnitude(curr)
    n = min(len(spec_base), len(spec_curr))
    error = np.mean(np.abs(spec_base[:n] - spec_curr[:n]))
    assert error < 0.05, f"Spectral diff {error:.3f} exceeds 5%"
