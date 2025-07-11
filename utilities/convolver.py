from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from music21 import stream

try:
    import soundfile as sf  # type: ignore
    if not all(hasattr(sf, a) for a in ("read", "write")):
        raise ImportError
except Exception:  # pragma: no cover - optional
    sf = None  # type: ignore

try:
    import soxr  # type: ignore
except Exception:  # pragma: no cover - optional
    soxr = None  # type: ignore

import os
import shutil
import warnings
from functools import lru_cache
from typing import Literal

from scipy.signal import resample_poly

try:
    import pyloudnorm as pyln  # type: ignore
except Exception:  # pragma: no cover - optional
    pyln = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional

    class _NoTqdm:
        def __init__(self, *a: object, **k: object) -> None: ...
        def update(self, *a: object, **k: object) -> None: ...
        def close(self) -> None: ...

    def tqdm(*a: object, **k: object) -> _NoTqdm:  # type: ignore
        return _NoTqdm()


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _resample(data: np.ndarray, src: int, dst: int, *, quality: str) -> np.ndarray:
    if src == dst:
        return data
    if soxr is not None:
        qmap = {"fast": soxr.QQ, "high": soxr.HQ, "ultra": soxr.VHQ}
        return soxr.resample(data, src, dst, quality=qmap.get(quality, soxr.QQ))
    window = ("kaiser", 16.0) if quality == "ultra" else ("kaiser", 8.0)
    from fractions import Fraction

    frac = Fraction(dst, src).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    try:
        res = resample_poly(data.astype(np.float64), up, down, axis=0, window=window)
    except TypeError:  # pragma: no cover - older SciPy
        res = resample_poly(data.astype(np.float64), up, down, axis=0)
    return res.astype(data.dtype, copy=False)


def _rms(arr: np.ndarray) -> float:
    """Return root-mean-square of ``arr``."""

    return float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))


def _mix_to_stereo(ir: np.ndarray, method: str = "rms") -> np.ndarray:
    """Return ``ir`` mixed down to stereo."""

    if ir.ndim == 1:
        ir = ir[:, None]
    if ir.shape[1] == 1:
        return np.repeat(ir, 2, axis=1).astype(np.float32, copy=False)

    if ir.shape[1] == 2:
        left = ir[:, 0]
        right = ir[:, 1]
        return np.stack([left, right], axis=1).astype(np.float32, copy=False)

    if method == "rms":
        weights = np.array([_rms(ir[:, i]) for i in range(ir.shape[1])], dtype=np.float64)
        if np.all(weights == 0):
            weights = np.ones_like(weights)
        weights = weights / np.sum(weights)
        mix = np.sum(ir * weights[None, :], axis=1)
        orig_pow = float(np.sum(np.square(ir, dtype=np.float64)))
        pre_pow = float(np.sum(np.square(mix, dtype=np.float64)))
        if pre_pow > 0:
            mix = mix * np.sqrt(orig_pow / (pre_pow * 2.0))
        stereo = np.stack([mix, mix], axis=1)
        return stereo.astype(np.float32, copy=False)

    mid = ir.shape[1] // 2
    left = ir[:, :mid].mean(axis=1)
    right = ir[:, mid:].mean(axis=1)
    return np.stack([left, right], axis=1).astype(np.float32, copy=False)


def _downmix_ir(
    ir: np.ndarray, mode: Literal["auto", "stereo", "none"] = "auto"
) -> np.ndarray:
    """Return IR down-mixed according to ``mode``."""
    if mode == "none":
        return ir
    if mode == "stereo":
        return _mix_to_stereo(ir)
    # auto
    if ir.ndim == 1 or ir.shape[1] <= 2:
        return ir
    return _mix_to_stereo(ir)


def _fft_convolve(sig: np.ndarray, ir: np.ndarray) -> np.ndarray:
    n = len(sig) + len(ir) - 1
    nfft = _next_pow2(n)
    S = np.fft.rfft(sig, nfft)
    H = np.fft.rfft(ir, nfft)
    y = np.fft.irfft(S * H, nfft)[:n]
    return y


def convolve_ir(
    audio: np.ndarray,
    ir: np.ndarray,
    block_size: int = 2**14,
    *,
    progress: bool = False,
) -> np.ndarray:
    """Overlap-add FFT convolution returning ``len(audio)+len(ir)-1`` samples."""
    if audio.ndim == 1:
        audio = audio[:, None]
    if ir.ndim == 1:
        ir = ir[:, None]
    out_len = audio.shape[0] + ir.shape[0] - 1
    channels = max(audio.shape[1], ir.shape[1])
    if ir.shape[0] < 2**14:
        result = []
        for ch in range(channels):
            s = audio[:, ch] if audio.shape[1] > 1 else audio[:, 0]
            h = ir[:, ch] if ir.shape[1] > 1 else ir[:, 0]
            result.append(_fft_convolve(s, h))
        return np.stack(result, axis=1)[:out_len]

    fft_size = _next_pow2(ir.shape[0] * 2)
    hop = fft_size - ir.shape[0] + 1
    H = np.fft.rfft(ir, fft_size, axis=0)
    result = np.zeros((out_len + hop, channels), dtype=np.float64)
    pos = 0
    bar = tqdm(total=audio.shape[0], disable=not progress, desc="IR", leave=False)
    while pos < audio.shape[0]:
        chunk = audio[pos : pos + hop]
        buf = np.zeros((fft_size, audio.shape[1]), dtype=np.float64)
        buf[: chunk.shape[0]] = chunk
        X = np.fft.rfft(buf, axis=0)
        y = np.fft.irfft(X * H, axis=0)
        result[pos : pos + fft_size, : audio.shape[1]] += y
        pos += hop
        bar.update(chunk.shape[0])
    bar.close()
    return result[:out_len].astype(np.float32)


_IR_CACHE_SIZE = int(os.environ.get("CONVOLVER_IR_CACHE", "8"))

@lru_cache(maxsize=_IR_CACHE_SIZE)
def load_ir(path: str) -> tuple[np.ndarray, int]:
    """Return IR data and sample rate from ``path``."""
    if sf is None:
        raise RuntimeError("soundfile is required for load_ir")
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.astype(np.float32), int(sr)


def _apply_tpdf_dither(data: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth == 32:
        return data
    if bit_depth not in (16, 24):
        return data
    if bit_depth == 24:
        # soundfile expects 24-bit PCM data in an int32 container
        lsb = 1.0 / (2**31)
    else:
        lsb = 1.0 / (2**15)
    noise = (np.random.random(data.shape) - 0.5) + (np.random.random(data.shape) - 0.5)
    return data + noise * lsb


def _fade_tail(
    data: np.ndarray,
    drop_db: float,
    sr: int,
    ms: float = 10.0,
    *,
    max_len: int | None = None,
) -> np.ndarray:
    if data.ndim > 1:
        mag = np.max(np.abs(data), axis=1)
    else:
        mag = np.abs(data)
    peak = float(np.max(mag))
    if peak == 0:
        return data
    thresh = peak * (10 ** (drop_db / 20.0))
    idx = np.where(mag > thresh)[0]
    if idx.size == 0:
        start = 0
        fade_len = len(data)
    else:
        start = idx[-1]
        fade_len = min(len(data) - start, int(sr * ms / 1000.0))
    if fade_len <= 0:
        return data
    fade = np.linspace(1.0, 0.0, fade_len)
    if data.ndim > 1:
        fade = fade[:, None]
    out = data.copy()
    out[start : start + fade_len] *= fade
    end = start + fade_len
    if max_len is not None:
        end = min(end, max_len)
    return out[:end]


def _quantize_pcm(data: np.ndarray, bit_depth: int) -> tuple[np.ndarray, str]:
    """Return integer PCM array and subtype."""
    if bit_depth == 32:
        return data.astype(np.float32), "FLOAT"
    if bit_depth == 24:
        max_val = float(2**31 - 1)
    else:
        max_val = float(2 ** (bit_depth - 1) - 1)
    min_val = -max_val - 1
    q = np.clip(np.round(data * max_val), min_val, max_val)
    if bit_depth == 24:
        return q.astype(np.int32), "PCM_24"
    return q.astype(np.int16), "PCM_16"


def render_with_ir(
    input_wav: str | Path,
    ir_wav: str | Path,
    out_wav: str | Path,
    *,
    lufs_target: float | None = None,
    gain_db: float | None = None,
    block_size: int = 2**14,
    bit_depth: int = 24,
    quality: str = "fast",
    oversample: int = 1,
    normalize: bool = True,
    dither: bool = True,
    downmix: Literal["auto", "stereo", "none"] = "auto",
    tail_db_drop: float = -60.0,
    progress: bool = False,
) -> Path:
    """Convolve ``input_wav`` with ``ir_wav`` and write to ``out_wav``.

    ``downmix`` controls handling of multi-channel impulse responses.
    """

    if sf is None:
        warnings.warn(
            "soundfile not installed; skipping convolution",
            RuntimeWarning,
        )
        shutil.copyfile(input_wav, out_wav)
        return Path(out_wav)

    inp = Path(input_wav)
    irp = Path(ir_wav)
    out = Path(out_wav)
    gain_db = gain_db or 0.0

    dither = bool(dither)
    if not normalize:
        dither = False

    try:
        y, sr = sf.read(inp, always_2d=True)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to read WAV: %s", exc)
        return out

    if not irp.is_file():
        logger.warning("IR file missing: %s", ir_wav)
        if y.shape[1] == 1:
            y = np.broadcast_to(y, (y.shape[0], 2))
        if dither:
            y = _apply_tpdf_dither(y, bit_depth)
        pcm, subtype = _quantize_pcm(y, bit_depth)
        sf.write(out, pcm, sr, subtype=subtype)
        return out

    ir, ir_sr = load_ir(str(irp))
    target_sr = 44100
    y = _resample(y, sr, target_sr, quality=quality)
    ir = _resample(ir, ir_sr, target_sr, quality=quality)
    ir = _downmix_ir(ir, downmix)
    sr = target_sr

    if oversample > 1:
        y = _resample(y, sr, sr * oversample, quality=quality)
        ir = _resample(ir, sr, sr * oversample, quality=quality)
        sr *= oversample

    if ir.shape[1] == 1 and y.shape[1] > 1:
        ir = np.broadcast_to(ir, (ir.shape[0], y.shape[1]))
    if y.shape[1] == 1 and ir.shape[1] > 1:
        y = np.broadcast_to(y, (y.shape[0], ir.shape[1]))

    orig_len = y.shape[0]
    data = convolve_ir(y, ir, block_size=block_size, progress=progress)
    if oversample > 1:
        data = _resample(data, sr, sr // oversample, quality=quality)
        sr //= oversample

    data = _fade_tail(data, tail_db_drop, sr, max_len=orig_len)

    if gain_db:
        data = data * (10 ** (gain_db / 20.0))

    if normalize:
        peak = float(np.max(np.abs(data)))
        if peak > 1.0:
            data = data / peak

    if lufs_target is not None:
        if pyln is None:
            logger.warning("pyloudnorm not installed; skipping LUFS normalization")
        else:
            try:
                meter = pyln.Meter(sr)
                diff = lufs_target - float(meter.integrated_loudness(data))
                diff = max(-3.0, min(3.0, diff))
                data = data * (10 ** (diff / 20.0))
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("pyloudnorm failed: %s", exc)

    if dither:
        data = _apply_tpdf_dither(data, bit_depth)
    pcm, subtype = _quantize_pcm(data, bit_depth)
    sf.write(out, pcm, sr, subtype=subtype)
    return out


def normalize_velocities(parts: list[stream.Part] | dict[str, stream.Part]) -> None:
    """Scale note velocities of ``parts`` so averages match."""
    if isinstance(parts, dict):
        all_parts = list(parts.values())
    else:
        all_parts = list(parts)
    if not all_parts:
        return
    avgs = []
    for p in all_parts:
        vals = [n.volume.velocity or 0 for n in p.recurse().notes if n.volume]
        if vals:
            avgs.append(sum(vals) / len(vals))
    if not avgs:
        return
    target = sum(avgs) / len(avgs)
    for p in all_parts:
        vals = [n.volume.velocity or 0 for n in p.recurse().notes if n.volume]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        if avg == 0:
            continue
        scale = target / avg
        for n in p.recurse().notes:
            if n.volume is None:
                continue
            val = int(max(1, min(127, round(n.volume.velocity * scale))))
            n.volume.velocity = val


def render_wav(
    midi_path: str,
    ir_path: str,
    out_path: str,
    sf2: str | None = None,
    *,
    parts: Iterable[stream.Part] | dict[str, stream.Part] | None = None,
    quality: str = "fast",
    bit_depth: int = 24,
    oversample: int = 1,
    normalize: bool = True,
    dither: bool = True,
    downmix: Literal["auto", "stereo", "none"] = "auto",
    tail_db_drop: float = -60.0,
    **kw: Any,
) -> Path:
    """Render ``midi_path`` with ``fluidsynth`` and apply ``ir_path``."""
    from utilities.synth import render_midi

    if sf is None:
        warnings.warn(
            "soundfile not installed; skipping render_wav",
            RuntimeWarning,
        )
        Path(out_path).touch()
        return Path(out_path)

    tmp_midi: Path | None = None
    if parts is not None:
        normalize_velocities(
            list(parts.values()) if isinstance(parts, dict) else list(parts)
        )
        score = stream.Score()
        for p in parts.values() if isinstance(parts, dict) else parts:
            score.insert(0, p)
        tmp_midi = Path(out_path).with_suffix(".norm.mid")
        score.write("midi", fp=str(tmp_midi))
        midi_in = tmp_midi
    else:
        midi_in = Path(midi_path)

    tmp = Path(out_path).with_suffix(".dry.wav")
    render_midi(midi_in, tmp, sf2_path=sf2)
    if "mix_opts" in kw:
        warnings.warn(
            "'mix_opts' dict is deprecated; pass options as keyword arguments",
            DeprecationWarning,
            stacklevel=2,
        )
        mo = kw.pop("mix_opts") or {}
        if isinstance(mo, Mapping):
            kw.update(mo)
    render_with_ir(
        tmp,
        ir_path,
        out_path,
        quality=quality,
        bit_depth=bit_depth,
        oversample=oversample,
        normalize=normalize,
        dither=dither,
        downmix=downmix,
        tail_db_drop=tail_db_drop,
        **kw,
    )
    tmp.unlink(missing_ok=True)
    if tmp_midi is not None:
        tmp_midi.unlink(missing_ok=True)
    return Path(out_path)


__all__ = [
    "render_with_ir",
    "load_ir",
    "convolve_ir",
    "normalize_velocities",
    "render_wav",
]
