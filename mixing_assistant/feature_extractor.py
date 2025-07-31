from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm

__all__ = ["extract_features", "main"]

_meters: dict[int, object] = {}
EXPECTED_KEYS = [
    "spectral_centroid_mean",
    "spectral_flatness_db",
    "spectral_rolloff95",
    "zero_cross_rate_mean",
    "loudness_i",
    "loudness_range",
    "crest_factor",
]


def extract_features(
    audio_path: Path,
    *,
    n_mels: int = 128,
    frame_len: float = 0.0464,
    hop_len: float = 0.0232,
    loudness_range_s: float = 3.0,
) -> dict[str, float]:
    try:
        librosa = importlib.import_module("librosa")
        pyln = importlib.import_module("pyloudnorm")
    except Exception:
        return {k: float("nan") for k in EXPECTED_KEYS}

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception:
        return {k: float("nan") for k in EXPECTED_KEYS}

    return _compute_features(y, sr, n_mels, frame_len, hop_len, loudness_range_s, librosa, pyln)


def _compute_features(
    y: np.ndarray,
    sr: int,
    n_mels: int,
    frame_len: float,
    hop_len: float,
    loudness_range_s: float,
    librosa: object,
    pyln: object,
) -> dict[str, float]:
    frame_length = int(sr * frame_len)
    hop_length = int(sr * hop_len)
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
    )
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=frame_length, hop_length=hop_length
    )
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    peak = np.max(np.abs(y))
    rms_mean = float(np.mean(rms))
    crest = float(20 * np.log10(peak / (rms_mean + 1e-12)))
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, roll_percent=0.95
    )
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )
    meter = _meters.get(sr)
    if meter is None:
        meter = pyln.Meter(sr)
        _meters[sr] = meter
    loudness_i = float(meter.integrated_loudness(y))
    step = int(loudness_range_s * sr)
    hop = int(sr * 0.1)
    lufs = [
        meter.integrated_loudness(y[i : i + step])
        for i in range(0, len(y) - step + 1, hop)
    ]
    loudness_range = (
        float(np.percentile(lufs, 95) - np.percentile(lufs, 10)) if lufs else 0.0
    )
    return {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_flatness_db": float(librosa.amplitude_to_db(flatness).mean()),
        "spectral_rolloff95": float(np.mean(rolloff)),
        "zero_cross_rate_mean": float(np.mean(zcr)),
        "loudness_i": loudness_i,
        "loudness_range": loudness_range,
        "crest_factor": crest,
    }


def _worker(path: Path) -> dict[str, float]:
    return extract_features(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio features")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("mix_features.parquet"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    audio_files = sorted(Path(args.root).rglob("*.wav"))
    workers = args.workers or min(8, os.cpu_count() or 1)
    ctx = mp.get_context("spawn") if sys.platform == "darwin" else None
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        results = list(tqdm(ex.map(_worker, audio_files), total=len(audio_files)))
    df = pd.DataFrame(results)
    df.to_parquet(args.out)


if __name__ == "__main__":
    main()
