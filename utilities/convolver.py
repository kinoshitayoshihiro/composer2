from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve


def render_with_ir(wav_path: str | Path, ir_path: str | Path, wav_out: str | Path) -> Path:
    """Convolve ``wav_path`` with ``ir_path`` and write to ``wav_out``."""
    wav_path = Path(wav_path)
    ir_path = Path(ir_path)
    wav_out = Path(wav_out)

    y, sr = sf.read(wav_path)
    ir, ir_sr = sf.read(ir_path)
    if ir_sr != sr:
        raise ValueError("Sample rate mismatch")
    if y.ndim > 1:
        y = y[:, 0]
    if ir.ndim > 1:
        ir = ir[:, 0]
    conv = fftconvolve(y, ir)[: len(y)]
    sf.write(wav_out, conv, sr)
    return wav_out

__all__ = ["render_with_ir"]
