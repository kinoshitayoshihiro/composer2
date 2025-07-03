from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


DEFAULT_TARGET_LUFS_MAP = {"verse": -16.0, "chorus": -12.0}


def normalize_wav(
    path_in: str | Path,
    path_out: str | Path,
    target_lufs: float | None = None,
    *,
    section: str | None = None,
    target_lufs_map: dict[str, float] | None = None,
) -> None:
    """Normalize WAV loudness to ``target_lufs`` or map-based value."""
    if target_lufs_map is None:
        target_lufs_map = DEFAULT_TARGET_LUFS_MAP
    if target_lufs is None:
        target_lufs = float(target_lufs_map.get(section, -14.0))
    y, sr = librosa.load(path_in, sr=None)
    rms = np.sqrt(np.mean(y ** 2)) or 1e-9
    current_lufs = 20 * np.log10(rms)
    gain = 10 ** ((target_lufs - current_lufs) / 20)
    y_norm = y * gain
    sf.write(path_out, y_norm, sr)

__all__ = ["normalize_wav"]
