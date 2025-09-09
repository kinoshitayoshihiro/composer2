"""Dataset utilities for bass timbre style-transfer experiments.

This module provides helpers to align source/target audio pairs and compute
melspectrogram training examples.  The dataset itself optionally caches raw
mels in ``root/.mel_cache`` for faster subsequent loading.

Specification reference: Timbre Style-Transfer Phase 0.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - for type hints
    pass

try:  # pragma: no cover - fallback for environments without project logger
    from utilities.logger import get_logger  # type: ignore[import]
except Exception:  # pragma: no cover - minimal stub
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

SR = 24000
HOP = 512
N_FFT = 2048
N_MELS = 128

logger = get_logger(__name__)


@dataclass
class TimbrePair:
    id: str
    src_path: Path
    tgt_path: Path
    midi_path: Path | None
    tgt_suffix: str

    """Grouping of source/target audio files belonging to the same phrase."""


def find_pairs(root: Path, src_suffix: str, tgt_suffixes: Sequence[str]) -> list[TimbrePair]:
    """Discover :class:`TimbrePair` objects within ``root``.

    Parameters
    ----------
    root
        Directory containing ``*.wav`` files named ``<id>__<suffix>.wav``.
    src_suffix
        File suffix for source audio.
    tgt_suffixes
        Iterable of possible target suffixes.

    Returns
    -------
    list[TimbrePair]
        All valid source/target pairings discovered under ``root``.
    """

    pairs: list[TimbrePair] = []
    groups: dict[str, dict[str, Path]] = {}
    for wav in root.glob("*.wav"):
        if "__" not in wav.stem:
            continue
        pid, suffix = wav.stem.split("__", 1)
        groups.setdefault(pid, {})[suffix] = wav
    for pid, mapping in groups.items():
        src = mapping.get(src_suffix)
        if src is None:
            logger.warning("Missing source %s for id %s", src_suffix, pid)
            continue
        midi = root / f"{pid}__{src_suffix}.mid"
        midi_path = midi if midi.exists() else None
        for tgt_suffix in tgt_suffixes:
            tgt = mapping.get(tgt_suffix)
            if tgt is None:
                logger.warning("Missing target %s for id %s", tgt_suffix, pid)
                continue
            pairs.append(TimbrePair(pid, src, tgt, midi_path, tgt_suffix))
    return pairs


def _resample_mono(path: Path) -> np.ndarray:
    """Return mono audio at :data:`SR`."""
    try:
        import librosa  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - handled by importorskip in tests
        raise RuntimeError("librosa is required") from exc

    y, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def _onset_env(y: np.ndarray) -> np.ndarray:
    """Compute onset strength envelope."""
    try:
        import librosa  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - handled by importorskip in tests
        raise RuntimeError("librosa is required") from exc

    cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP))
    env = librosa.onset.onset_strength(S=cqt, sr=SR, hop_length=HOP)
    return env.astype(np.float32)


def _xcorr_offset(a: np.ndarray, b: np.ndarray) -> int:
    """Return sample offset maximizing cross-correlation."""
    corr = np.correlate(a, b, mode="full")
    return int(corr.argmax() - len(b) + 1)


def _shift_audio(y: np.ndarray, frames: int) -> np.ndarray:
    """Shift audio by ``frames`` hops with zero padding."""
    shift = frames * HOP
    if shift > 0:
        y = y[shift:]
    elif shift < 0:
        y = np.pad(y, (-shift, 0))[: len(y)]
    return y


def _compute_mel(y: np.ndarray) -> np.ndarray:
    """Compute mel-spectrogram in decibel scale."""
    try:
        import librosa  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - handled by importorskip in tests
        raise RuntimeError("librosa is required") from exc

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
    )
    return librosa.power_to_db(mel).astype(np.float32)


def _normalize(mel: np.ndarray) -> NDArray[np.float32]:
    """Scale ``mel`` to ``0-1`` range.

    A silent input yields an array filled with ``0.5`` so that networks learn a
    neutral mapping.
    """

    mn = float(mel.min())
    mx = float(mel.max())
    if mx - mn < 1e-8:
        return np.zeros_like(mel, dtype=np.float32) + 0.5
    return ((mel - mn) / (mx - mn)).astype(np.float32)


def _fix_length(mel: np.ndarray, max_len: int) -> np.ndarray:
    """Trim mel to ``max_len`` frames if necessary."""
    if mel.shape[1] > max_len:
        mel = mel[:, :max_len]
    return mel


def _align_audio(src: np.ndarray, tgt: np.ndarray, midi: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Align ``src`` and ``tgt`` using onsets and optional MIDI.

    Raises
    ------
    RuntimeError
        If ``librosa`` is not installed.
    """
    try:
        import librosa  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - handled by importorskip in tests
        raise RuntimeError("librosa is required") from exc

    env_src = _onset_env(src)
    env_tgt = _onset_env(tgt)
    if midi is not None:
        try:
            import pretty_midi  # type: ignore[import]

            pm = pretty_midi.PrettyMIDI(str(midi))
            times: list[float] = [n.start for inst in pm.instruments for n in inst.notes]
        except Exception:
            times = []
        if times:
            click = librosa.clicks(times=times, sr=SR, hop_length=HOP, length=max(len(src), len(tgt)))
            env_click = _onset_env(click)
            offset_src = _xcorr_offset(env_src, env_click)
            offset_tgt = _xcorr_offset(env_tgt, env_click)
            src = _shift_audio(src, -offset_src)
            tgt = _shift_audio(tgt, -offset_tgt)
            env_src = _onset_env(src)
            env_tgt = _onset_env(tgt)
    _, wp = librosa.sequence.dtw(env_src[np.newaxis, :], env_tgt[np.newaxis, :])
    wp = np.array(wp)[::-1]
    src_idx = np.clip(wp[:, 0], 0, len(env_src) - 1)
    tgt_idx = np.clip(wp[:, 1], 0, len(env_tgt) - 1)
    mel_src_full = _compute_mel(src)[:, : len(env_src)]
    mel_tgt_full = _compute_mel(tgt)[:, : len(env_tgt)]
    mel_src = mel_src_full[:, src_idx]
    mel_tgt = mel_tgt_full[:, tgt_idx]
    return mel_src, mel_tgt


def compute_mel_pair(
    src_path: Path,
    tgt_path: Path,
    midi_path: Path | None,
    max_len: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return aligned mel-spectrograms for a pair of audio files."""

    src = _resample_mono(src_path)
    tgt = _resample_mono(tgt_path)
    mel_src, mel_tgt = _align_audio(src, tgt, midi_path)
    mel_src = _fix_length(mel_src, max_len)
    mel_tgt = _fix_length(mel_tgt, max_len)
    return mel_src, mel_tgt


class BassTimbreDataset:
    """Iterable dataset of aligned mel-spectrogram pairs."""
    def __init__(
        self,
        root: Path,
        src_suffix: str = "wood",
        tgt_suffixes: Sequence[str] | None = None,
        max_len: int = 30_000,
        cache: bool = True,
    ) -> None:
        """Create a dataset rooted at ``root``."""
        self.root = Path(root)
        self.src_suffix = src_suffix
        self.tgt_suffixes = list(tgt_suffixes or ["synth"])
        self.max_len = max_len
        self.cache = cache
        self.pairs = find_pairs(self.root, self.src_suffix, self.tgt_suffixes)
        self.cache_dir = self.root / ".mel_cache"
        if self.cache:
            self.cache_dir.mkdir(exist_ok=True)
            self.write_cache()

    def _cache_path(self, pair: TimbrePair) -> Path:
        """Return cache path for ``pair``."""
        return self.cache_dir / f"{pair.id}__{self.src_suffix}->{pair.tgt_suffix}.npy"

    def write_cache(self) -> None:
        """Compute and store raw mel pairs under :attr:`cache_dir`."""
        for pair in self.pairs:
            path = self._cache_path(pair)
            if path.exists():
                continue
            path.parent.mkdir(exist_ok=True)
            mel_src, mel_tgt = compute_mel_pair(pair.src_path, pair.tgt_path, pair.midi_path, self.max_len)
            np.save(path, np.stack([mel_src, mel_tgt]))

    def __len__(self) -> int:
        """Number of discovered pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, NDArray[np.float32]]:
        """Return normalized mel pair ``idx`` as torch tensors.

        Mel arrays are pitch-major (``128 × T``).
        """
        import torch

        pair = self.pairs[idx]
        path = self._cache_path(pair)
        if self.cache and path.exists():
            arr = np.load(path)
            mel_src, mel_tgt = arr[0], arr[1]
        else:
            mel_src, mel_tgt = compute_mel_pair(pair.src_path, pair.tgt_path, pair.midi_path, self.max_len)
            if self.cache:
                path.parent.mkdir(exist_ok=True)
                np.save(path, np.stack([mel_src, mel_tgt]))
        mel_src = _normalize(mel_src)
        mel_tgt = _normalize(mel_tgt)
        return {
            "src": torch.tensor(mel_src, dtype=torch.float32),
            "tgt": torch.tensor(mel_tgt, dtype=torch.float32),
            "id": pair.id,
        }

__all__ = [
    "BassTimbreDataset",
    "find_pairs",
    "compute_mel_pair",
    "TimbrePair",
]
