from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def normalize_wav(path_in: str | Path, path_out: str | Path, target_lufs: float = -14.0) -> None:
    """Normalize WAV loudness to ``target_lufs`` (approximate)."""
    y, sr = librosa.load(path_in, sr=None)
    rms = np.sqrt(np.mean(y ** 2)) or 1e-9
    current_lufs = 20 * np.log10(rms)
    gain = 10 ** ((target_lufs - current_lufs) / 20)
    y_norm = y * gain
    sf.write(path_out, y_norm, sr)

__all__ = ["normalize_wav"]
