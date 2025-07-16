from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import librosa
import logging


def extract_pitch(wav_path: Path) -> List[float]:
    """Return pitch contour for *wav_path* using librosa's YIN algorithm."""
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    if np.all(np.isnan(f0)):
        logging.warning("No pitch detected in %s", wav_path)
        return []
    return f0.astype(float).tolist()


def extract_amplitude_envelope(wav_path: Path) -> List[float]:
    """Return amplitude envelope derived from short-time energy."""
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    envelope = librosa.feature.rms(y=y)[0]
    return envelope.astype(float).tolist()


def map_envelope_to_velocity(envelope: List[float], min_vel: int = 30, max_vel: int = 127) -> List[int]:
    """Map a normalized amplitude *envelope* to MIDI velocity values."""
    if not envelope:
        return []
    arr = np.asarray(envelope, dtype=float)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    scaled = arr * (max_vel - min_vel) + min_vel
    return scaled.round().clip(min_vel, max_vel).astype(int).tolist()

__all__ = [
    "extract_pitch",
    "extract_amplitude_envelope",
    "map_envelope_to_velocity",
]
