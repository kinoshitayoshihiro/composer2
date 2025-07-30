# utilities/consonant_extract.py
"""Consonant peak extraction utilities.

This module provides a small CLI and a helper function to detect
fricative/consonant-like peaks from WAV files.  The implementation
relies on ``librosa`` for onset strength calculation with optional
high-pass/low-pass filtering.  Essentia spectral flux can be used
when the library is available.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


class EssentiaUnavailable(RuntimeError):
    """Raised when the Essentia backend is requested but not installed."""

_DEF_WIN = 2048
_DEF_HOP = 512
_DEF_THR = 0.25


def _apply_filters(
    y: np.ndarray,
    sr: int,
    hpf: float | None = None,
    lpf: float | None = None,
) -> np.ndarray:
    nyq = 0.5 * sr
    if hpf is not None:
        if hpf >= nyq:
            raise ValueError("hpf cutoff must be below Nyquist")
        b, a = butter(2, hpf / nyq, btype="highpass")
        padlen = 3 * (max(len(a), len(b)) - 1)
        if len(y) <= padlen:
            logging.warning(
                "input length (%d) <= padlen (%d); skipping high-pass filter",
                len(y),
                padlen,
            )
        else:
            y = filtfilt(b, a, y)
    if lpf is not None:
        if lpf >= nyq:
            raise ValueError("lpf cutoff must be below Nyquist")
        b, a = butter(2, lpf / nyq, btype="lowpass")
        padlen = 3 * (max(len(a), len(b)) - 1)
        if len(y) <= padlen:
            logging.warning(
                "input length (%d) <= padlen (%d); skipping low-pass filter",
                len(y),
                padlen,
            )
        else:
            y = filtfilt(b, a, y)
    return y


def detect_consonant_peaks(
    wav_path: str | Path,
    *,
    algo: str = "librosa",
    window: int = _DEF_WIN,
    hop: int = _DEF_HOP,
    thr: float = _DEF_THR,
    hpf: float | None = None,
    lpf: float | None = None,
) -> list[float]:
    """Return peak times in seconds from ``wav_path``."""
    path = Path(wav_path)
    y, sr = librosa.load(str(path), sr=None, mono=True)
    y = _apply_filters(y, sr, hpf, lpf)

    algo = algo.lower()
    if algo == "librosa":
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop, n_fft=window
        )
    elif algo == "essentia":
        try:
            import essentia.standard as ess
        except Exception as exc:  # pragma: no cover - optional dependency
            raise EssentiaUnavailable("Essentia not installed") from exc
        flux = ess.SpectralFlux()
        spectrum = ess.Spectrum()
        frames = ess.FrameGenerator(y, frameSize=window, hopSize=hop, startFromZero=True)
        onset_env = [flux(spectrum(f)) for f in frames]
        onset_env = np.array(onset_env)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    if onset_env.size == 0:
        return []
    thr_val = thr * float(np.max(onset_env))
    peaks, _ = find_peaks(onset_env, height=thr_val)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    return times.tolist()


def extract_to_json(
    wav_path: str | Path,
    out_path: str | Path,
    **kwargs: object,
) -> list[float]:
    times = detect_consonant_peaks(wav_path, **kwargs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(times, indent=2))
    return times


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract consonant peaks")
    parser.add_argument("input_wav")
    parser.add_argument("-o", "--output", default="consonant_peaks.json")
    parser.add_argument("--algo", default="librosa")
    parser.add_argument("--window", type=int, default=_DEF_WIN)
    parser.add_argument("--hop", type=int, default=_DEF_HOP)
    parser.add_argument("--thr", type=float, default=_DEF_THR)
    parser.add_argument("--hpf", type=float, default=None)
    parser.add_argument("--lpf", type=float, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    extract_to_json(
        args.input_wav,
        args.output,
        algo=args.algo,
        window=args.window,
        hop=args.hop,
        thr=args.thr,
        hpf=args.hpf,
        lpf=args.lpf,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
