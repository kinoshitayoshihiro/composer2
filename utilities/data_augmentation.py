from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import librosa
import soundfile as sf


def pitch_shift(wav: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Return *wav* shifted by *n_steps* semitones."""
    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=n_steps)


def time_stretch(wav: np.ndarray, rate: float) -> np.ndarray:
    """Return time-stretched audio."""
    return librosa.effects.time_stretch(wav, rate)


def add_gaussian_noise(wav: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix Gaussian noise with given signal-to-noise ratio in dB."""
    sig_power = np.mean(wav ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(scale=np.sqrt(noise_power), size=len(wav))
    return wav + noise


def augment_wav_dir(
    src_dir: Path,
    dst_dir: Path,
    shifts: List[float],
    stretches: List[float],
    snrs: List[float],
    *,
    progress: bool = False,
) -> None:
    """Augment all WAV files under *src_dir* and save to *dst_dir*."""
    if not src_dir.exists():
        raise ValueError(f"Source directory {src_dir} does not exist")
    paths = list(src_dir.rglob("*.wav"))
    if not paths:
        raise ValueError(f"No WAV files found in {src_dir}")
    iterator: Iterable[Path] = paths
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(paths, desc="augment")
        except Exception:
            pass
    for path in iterator:
        y, sr = librosa.load(path, sr=None, mono=True)
        for sh in shifts:
            y_ps = pitch_shift(y, sr=sr, n_steps=sh) if sh else y
            for st in stretches:
                y_ts = time_stretch(y_ps, st) if st != 1.0 else y_ps
                for snr in snrs:
                    y_no = add_gaussian_noise(y_ts, snr) if snr else y_ts
                    rel = path.relative_to(src_dir)
                    out = dst_dir / rel.parent / f"{rel.stem}_ps{sh}_ts{st}_sn{snr}.wav"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(out, y_no, sr)

__all__ = [
    'pitch_shift',
    'time_stretch',
    'add_gaussian_noise',
    'augment_wav_dir',
]
