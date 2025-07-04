from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf

try:
    import pyloudnorm as pyln  # type: ignore
except Exception:  # pragma: no cover - optional
    pyln = None  # type: ignore

try:
    from pydub import AudioSegment  # type: ignore
except Exception:  # pragma: no cover - optional
    AudioSegment = None  # type: ignore

DEFAULT_TARGET_LUFS_MAP: dict[str, float] = {"verse": -16.0, "chorus": -12.0}


def _measure_lufs(samples: np.ndarray, sr: int) -> float:
    if pyln is not None:
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(samples))
    rms = np.sqrt(np.mean(samples ** 2)) or 1e-9
    return 20 * np.log10(rms)


def _apply_gain(samples: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10 ** (gain_db / 20)
    return samples * gain


def normalize_wav(
    wav_path: Path,
    *,
    section: str,
    target_lufs_map: Mapping[str, float] | None = None,
) -> None:
    """Normalize ``wav_path`` to the loudness of ``section``."""
    target_lufs_map = target_lufs_map or DEFAULT_TARGET_LUFS_MAP
    target = float(target_lufs_map.get(section, -14.0))

    if AudioSegment is not None:
        audio = AudioSegment.from_file(wav_path)
        array = np.array(audio.get_array_of_samples()).astype(float)
        if audio.channels > 1:
            array = array.reshape((-1, audio.channels)).mean(axis=1)
        array /= 1 << (8 * audio.sample_width - 1)
        lufs = _measure_lufs(array, audio.frame_rate)
        gain = target - lufs
        audio = audio.apply_gain(gain)
        audio.export(wav_path, format="wav")
        return

    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    lufs = _measure_lufs(y, sr)
    gain = target - lufs
    y_norm = _apply_gain(y, gain)
    sf.write(wav_path, y_norm, sr)

__all__ = ["normalize_wav"]
