from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("breath")

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional
    sf = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

try:  # pragma: no cover - optional
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional
    ort = None  # type: ignore


def infer_breath_mask(
    wav: Path | str,
    hop_ms: float = 10.0,
    thr_offset_db: float = -30.0,
    percentile: float = 95.0,
    onnx_path: str | Path | None = None,
) -> np.ndarray:
    """Return boolean mask of low-energy frames in ``wav``."""
    env_path = os.getenv("BREATH_ONNX_PATH")
    path = Path(onnx_path or env_path) if onnx_path or env_path else None
    if path is not None:
        try:
            return infer_breath_mask_onnx(
                wav,
                hop_ms=hop_ms,
                thr_offset_db=thr_offset_db,
                percentile=percentile,
                path=path,
            )
        except Exception as exc:
            logger.warning("onnx failed: %s; falling back to numpy", exc)
    return _infer_breath_mask_numpy(wav, hop_ms, thr_offset_db, percentile)


def _infer_breath_mask_numpy(
    wav: Path | str,
    hop_ms: float,
    thr_offset_db: float,
    percentile: float,
) -> np.ndarray:
    if sf is None:
        raise ImportError("soundfile is required")

    y, sr = sf.read(str(wav))
    if y.ndim > 1:
        logger.warning("stereo detected; converting to mono")
        y = y.mean(axis=1)
    hop = max(1, int(sr * hop_ms / 1000))
    frames = int(np.ceil(len(y) / hop))
    pad = frames * hop - len(y)
    if pad:
        y_pad = np.pad(y, (0, pad))
    else:
        y_pad = y
    energies = np.abs(y_pad[: frames * hop].reshape(frames, hop)).mean(axis=1)
    eps = max(1e-10, float(energies.max()) * 1e-6)
    e_db = 10 * np.log10(energies + eps)
    thr = np.percentile(e_db, percentile) + thr_offset_db
    mask = e_db < thr

    if torch is not None:
        with torch.no_grad():
            logits = torch.from_numpy(e_db).float() * 0.1
            prob = torch.sigmoid(logits).numpy()
        mask &= prob < 0.5

    return mask


def infer_breath_mask_onnx(
    wav: Path | str,
    *,
    hop_ms: float = 10.0,
    thr_offset_db: float = -30.0,
    percentile: float = 95.0,
    path: Path,
) -> np.ndarray:
    """Placeholder ONNX implementation.

    If ``onnxruntime`` is unavailable, ``ImportError`` is raised.
    """
    if ort is None:
        raise ImportError("onnxruntime is required")
    if not path.is_file():
        raise FileNotFoundError(str(path))
    session = ort.InferenceSession(str(path))
    logger.debug("infer_breath_mask_onnx stub running with %s", path)
    _ = session  # unused, placeholder
    return _infer_breath_mask_numpy(wav, hop_ms, thr_offset_db, percentile)


__all__ = ["infer_breath_mask", "infer_breath_mask_onnx"]
