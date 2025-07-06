from __future__ import annotations

from pathlib import Path
import logging

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional
    sf = None  # type: ignore
from scipy.signal import fftconvolve

try:
    import pyloudnorm as pyln  # type: ignore
except Exception:  # pragma: no cover - optional
    pyln = None  # type: ignore
try:
    import soxr  # type: ignore
except Exception:  # pragma: no cover - optional
    soxr = None  # type: ignore
try:
    if sf is not None:
        sf.default_subtype("WAV", "FLOAT")
except Exception:
    pass
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional

    class _NoTqdm:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return

        def update(self, *args: object, **kwargs: object) -> None:
            return

        def close(self) -> None:
            return

    def tqdm(*args: object, **kwargs: object) -> _NoTqdm:  # type: ignore
        return _NoTqdm()


logger = logging.getLogger(__name__)


def _write_gain(data: np.ndarray, sr: int, path: Path, gain_db: float) -> None:
    """Write *data* to *path* applying gain and normalisation."""
    if sf is None:
        raise RuntimeError("soundfile is required for WAV rendering")
    if gain_db:
        data = data * (10 ** (gain_db / 20.0))
    peak = np.max(np.abs(data))
    if peak > 1.0:
        data = data / peak
    sf.write(path, data.astype(np.float32), sr, subtype="FLOAT")


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

    if sf is None:
        raise RuntimeError("soundfile is required for render_with_ir")
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
        if y.shape[1] == 1:
            y = np.broadcast_to(y, (y.shape[0], 2))
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


def load_ir(path: str) -> tuple[np.ndarray, int]:
    """Return IR data and sample rate."""
    if sf is None:
        raise RuntimeError("soundfile is required for load_ir")
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), int(sr)


def convolve_ir(
    audio: np.ndarray, ir: np.ndarray, block_size: int = 2**14
) -> np.ndarray:
    """Overlap-add FFT convolution returning the input length."""
    if audio.ndim == 1:
        audio = audio[:, None]
    if ir.ndim == 1:
        ir = ir[:, None]
    out_len = audio.shape[0] + ir.shape[0] - 1
    channels = max(audio.shape[1], ir.shape[1])
    result = np.zeros((out_len, channels), dtype=np.float64)
    fft_size = 1 << int(np.ceil(np.log2(block_size + ir.shape[0] - 1)))
    H = np.fft.rfft(ir, fft_size, axis=0)
    pos = 0
    while pos < audio.shape[0]:
        chunk = audio[pos : pos + block_size]
        pad = np.zeros((fft_size, audio.shape[1]), dtype=np.float64)
        pad[: chunk.shape[0]] = chunk
        X = np.fft.rfft(pad, axis=0)
        y = np.fft.irfft(X * H, axis=0)[:fft_size]
        result[pos : pos + fft_size, : audio.shape[1]] += y
        pos += block_size
    return result[: audio.shape[0]].astype(np.float32)


def render_wav(
    midi_path: str,
    ir_path: str,
    out_path: str,
    sf2: str | None = None,
    **mix_opts,
) -> Path:
    """Render ``midi_path`` with ``fluidsynth`` and apply ``ir_path``."""
    from utilities.synth import render_midi

    if sf is None:
        raise RuntimeError("soundfile is required for render_wav")

    tmp = Path(out_path).with_suffix(".dry.wav")
    render_midi(midi_path, tmp, sf2_path=sf2)
    audio, sr = sf.read(tmp, dtype="float32")
    ir, ir_sr = load_ir(ir_path)
    if sr != ir_sr:
        from scipy.signal import resample

        ir = resample(ir, int(len(ir) * sr / ir_sr))
    if ir.ndim == 1:
        ir = ir[:, None]
    out = convolve_ir(audio, ir, block_size=mix_opts.get("block_size", 2**14))
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak
    sf.write(out_path, (out * 32767).astype(np.int16), sr)
    tmp.unlink(missing_ok=True)
    return Path(out_path)


__all__ = ["render_with_ir", "load_ir", "convolve_ir", "render_wav"]
