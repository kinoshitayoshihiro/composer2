from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import logging

import numpy as np

SR = 24000
HOP = 512
N_FFT = 2048
N_MELS = 128


@dataclass
class TimbrePair:
    id: str
    src_path: Path
    tgt_path: Path
    midi_path: Optional[Path]
    tgt_suffix: str


def find_pairs(root: Path, src_suffix: str, tgt_suffixes: Sequence[str]) -> List[TimbrePair]:
    pairs: List[TimbrePair] = []
    groups: dict[str, dict[str, Path]] = {}
    for wav in root.glob("*.wav"):
        if "__" not in wav.stem:
            continue
        pid, suffix = wav.stem.split("__", 1)
        groups.setdefault(pid, {})[suffix] = wav
    for pid, mapping in groups.items():
        src = mapping.get(src_suffix)
        if src is None:
            logging.warning("Missing source %s for id %s", src_suffix, pid)
            continue
        midi = root / f"{pid}__{src_suffix}.mid"
        midi_path = midi if midi.exists() else None
        for tgt_suffix in tgt_suffixes:
            tgt = mapping.get(tgt_suffix)
            if tgt is None:
                logging.warning("Missing target %s for id %s", tgt_suffix, pid)
                continue
            pairs.append(TimbrePair(pid, src, tgt, midi_path, tgt_suffix))
    return pairs


def _resample_mono(path: Path) -> np.ndarray:
    import librosa

    y, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def _onset_env(y: np.ndarray) -> np.ndarray:
    import librosa

    cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP))
    env = librosa.onset.onset_strength(S=cqt, sr=SR, hop_length=HOP)
    return env.astype(np.float32)


def _xcorr_offset(a: np.ndarray, b: np.ndarray) -> int:
    corr = np.correlate(a, b, mode="full")
    return int(corr.argmax() - len(b) + 1)


def _shift_audio(y: np.ndarray, frames: int) -> np.ndarray:
    shift = frames * HOP
    if shift > 0:
        y = y[shift:]
    elif shift < 0:
        y = np.pad(y, (-shift, 0))[: len(y)]
    return y


def _compute_mel(y: np.ndarray) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
    )
    return librosa.power_to_db(mel).astype(np.float32)


def _normalize(mel: np.ndarray) -> np.ndarray:
    mn = float(mel.min())
    mx = float(mel.max())
    if mx - mn < 1e-8:
        return np.zeros_like(mel)
    return ((mel - mn) / (mx - mn)).astype(np.float32)


def _fix_length(mel: np.ndarray, max_len: int) -> np.ndarray:
    if mel.shape[1] > max_len:
        mel = mel[:, :max_len]
    return mel


def _align_audio(src: np.ndarray, tgt: np.ndarray, midi: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    import librosa

    env_src = _onset_env(src)
    env_tgt = _onset_env(tgt)
    if midi is not None:
        try:
            import pretty_midi

            pm = pretty_midi.PrettyMIDI(str(midi))
            times: List[float] = [n.start for inst in pm.instruments for n in inst.notes]
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


def compute_mel_pair(src_path: Path, tgt_path: Path, midi_path: Optional[Path], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    src = _resample_mono(src_path)
    tgt = _resample_mono(tgt_path)
    mel_src, mel_tgt = _align_audio(src, tgt, midi_path)
    mel_src = _fix_length(mel_src, max_len)
    mel_tgt = _fix_length(mel_tgt, max_len)
    return mel_src, mel_tgt


class BassTimbreDataset:
    def __init__(
        self,
        root: Path,
        src_suffix: str = "wood",
        tgt_suffixes: Sequence[str] | None = None,
        max_len: int = 30_000,
        cache: bool = True,
    ) -> None:
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
        return self.cache_dir / f"{pair.id}__{self.src_suffix}->{pair.tgt_suffix}.npy"

    def write_cache(self) -> None:
        for pair in self.pairs:
            path = self._cache_path(pair)
            if path.exists():
                continue
            path.parent.mkdir(exist_ok=True)
            mel_src, mel_tgt = compute_mel_pair(pair.src_path, pair.tgt_path, pair.midi_path, self.max_len)
            np.save(path, np.stack([mel_src, mel_tgt]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
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
