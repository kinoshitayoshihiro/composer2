from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

try:
    import pyloudnorm as pyln  # type: ignore
except Exception:  # pragma: no cover - optional
    pyln = None  # type: ignore
try:
    import soxr  # type: ignore
except Exception:  # pragma: no cover - optional
    soxr = None  # type: ignore
from tqdm import tqdm


logger = logging.getLogger(__name__)


def _write_gain(data: np.ndarray, sr: int, path: Path, gain_db: float) -> None:
    """Write *data* to *path* applying gain and normalisation."""
    if gain_db:
        data = data * (10 ** (gain_db / 20.0))
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak
    sf.write(path, data, sr)


def render_with_ir(
    input_wav: str | Path,
    ir_wav: str | Path,
    out_wav: str | Path,
    gain_db: float = 0.0,
    *,
    lufs_target: float | None = None,
    progress: bool = False,
) -> None:
    """Convolve ``input_wav`` with ``ir_wav`` and write to ``out_wav``.

    If sample rates differ, both are resampled to 44100 Hz using soxr when
    available. Mono IRs are broadcast to stereo. The result is normalised,
    optionally amplified, and loudness-adjusted toward ``lufs_target``.
    """

    inp = Path(input_wav)
    irp = Path(ir_wav)
    out = Path(out_wav)

    try:
        y, sr = sf.read(inp, always_2d=True)
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - best effort
        logger.warning("Failed to read WAV: %s", exc)
        return

    if not irp.is_file():
        logger.warning("IR file missing: %s", ir_wav)
        _write_gain(y, sr, out, gain_db)
        return

    try:
        ir, ir_sr = sf.read(irp, always_2d=True)
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - best effort
        logger.warning("Failed to read IR: %s", exc)
        _write_gain(y, sr, out, gain_db)
        return

    if sr != ir_sr:
        if soxr is None:
            logger.warning("Sample rate mismatch: %s vs %s", sr, ir_sr)
            target_sr = sr
        else:
            target_sr = 44100
            y = soxr.resample(y, sr, target_sr, quality="mq")
            ir = soxr.resample(ir, ir_sr, target_sr, quality="mq")
        sr = ir_sr = target_sr

    if ir.shape[1] == 1 and y.shape[1] > 1:
        ir = np.broadcast_to(ir, (ir.shape[0], y.shape[1]))
    if y.shape[1] == 1 and ir.shape[1] > 1:
        y = np.broadcast_to(y, (y.shape[0], ir.shape[1]))

    channels = max(y.shape[1], ir.shape[1])
    out_data = []
    bar = tqdm(total=channels, disable=not progress, desc="IR", leave=False)
    for ch in range(channels):
        s = y[:, ch] if y.shape[1] > 1 else y[:, 0]
        h = ir[:, ch] if ir.shape[1] > 1 else ir[:, 0]
        conv = fftconvolve(s, h)[: len(s)]
        out_data.append(conv)
        bar.update(1)
    bar.close()

    data = np.stack(out_data, axis=1)
    rms_in = np.sqrt(np.mean(np.square(y)))
    rms_out = np.sqrt(np.mean(np.square(data)))
    if rms_out > 0:
        data = data * (rms_in / rms_out)

    lufs = None
    if lufs_target is not None and pyln is not None:
        meter = pyln.Meter(sr)
        lufs = float(meter.integrated_loudness(data))
        diff = lufs_target - lufs
        diff = min(3.0, max(-3.0, diff))
        data = data * (10 ** (diff / 20.0))
        lufs = float(meter.integrated_loudness(data))

    _write_gain(data, sr, out, gain_db)
    if lufs is None and lufs_target is not None:
        lufs = lufs_target
    logger.info(
        "\u2713 Convolved %.1f s IR \u2192 %s (%.1f LUFS)",
        len(ir) / sr,
        out,
        lufs if lufs is not None else -0.0,
    )


__all__ = ["render_with_ir"]
