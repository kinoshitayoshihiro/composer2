from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass
class PeakExtractorConfig:
    sr: int = 22050
    frame_length: int = 2048
    hop_length: int = 512
    rms_smooth_ms: float = 20.0
    threshold_db: float = -20.0
    min_distance_ms: float = 30.0


def extract_peaks(
    wav_path: Path | str,
    cfg: PeakExtractorConfig = PeakExtractorConfig(),
) -> list[float]:
    """Return strictly sorted RMS peak times in seconds."""
    path = Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(path)

    y, sr = librosa.load(path.as_posix(), sr=cfg.sr, mono=True)
    if y.size == 0:
        raise ValueError("empty audio")

    rms = librosa.feature.rms(y=y, frame_length=cfg.frame_length, hop_length=cfg.hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    win = max(1, int(round(cfg.rms_smooth_ms / 1000 * sr / cfg.hop_length)))
    rms_db = uniform_filter1d(rms_db, win)
    above = rms_db > cfg.threshold_db
    onset_idx = np.where(np.logical_and(above[1:], ~above[:-1]))[0] + 1
    times = librosa.frames_to_time(onset_idx, sr=sr, hop_length=cfg.hop_length)
    min_dist = cfg.min_distance_ms / 1000.0
    filtered: list[float] = []
    last = -np.inf
    for t in times:
        if t - last >= min_dist:
            filtered.append(float(t))
            last = t
    return filtered

