from __future__ import annotations

import logging
import os
import shutil
import warnings
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from music21 import stream
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------#
# Optional native-deps (失敗しても graceful-degradation)
# ---------------------------------------------------------------------------#
try:
    import soundfile as sf  # type: ignore

    if not all(hasattr(sf, a) for a in ("read", "write")):
        raise ImportError
except Exception:  # pragma: no cover
    sf = None  # type: ignore

try:
    import soxr  # type: ignore
except Exception:  # pragma: no cover
    soxr = None  # type: ignore

try:
    import pyloudnorm as pyln  # type: ignore
except Exception:  # pragma: no cover
    pyln = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover

    class _NoTqdm:
        def __init__(self, *a: object, **k: object) -> None: ...
        def update(self, *a: object, **k: object) -> None: ...
        def close(self) -> None: ...

    def tqdm(*a: object, **k: object) -> _NoTqdm:  # type: ignore
        return _NoTqdm()


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------#
# Utility helpers
# ---------------------------------------------------------------------------#


def _next_pow2(n: int) -> int:
    """Return the next power-of-two ≥ *n*."""
    return 1 << (n - 1).bit_length()


def _resample(
    data: np.ndarray,
    src: int,
    dst: int,
    *,
    quality: str = "fast",
) -> np.ndarray:
    """Resample *data* from *src* Hz → *dst* Hz with optional quality."""
    if src == dst:
        return data
    if soxr is not None:  # fast/high/ultra ⇄ soxr quality map
        qmap = {"fast": soxr.QQ, "high": soxr.HQ, "ultra": soxr.VHQ}
        return soxr.resample(data, src, dst, quality=qmap.get(quality, soxr.QQ))

    # Fallback: scipy polyphase resampler
    window = ("kaiser", 16.0) if quality == "ultra" else ("kaiser", 8.0)
    from fractions import Fraction

    frac = Fraction(dst, src).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    try:
        res = resample_poly(data.astype(np.float64), up, down, axis=0, window=window)
    except TypeError:  # older SciPy
        res = resample_poly(data.astype(np.float64), up, down, axis=0)
    return res.astype(data.dtype, copy=False)


def _rms(arr: np.ndarray) -> float:
    """Root-mean-square utility."""
    return float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))


_STEREO_METHODS: Final = frozenset({"mean", "rms"})


def _mix_to_stereo(ir: np.ndarray, method: str = "rms") -> np.ndarray:
    """
    Down-mix multi-channel IR → stereo.

    Parameters
    ----------
    ir
        (N, C) IR array.
    method
        ``"rms"`` – RMS weighted mix (energy preserving).  
        ``"mean"`` – Simple arithmetic mean.
    """
    if method not in _STEREO_METHODS:
        raise ValueError(f"`method` must be one of {_STEREO_METHODS}")

    if ir.ndim == 1:
        ir = ir[:, None]

    # Mono → duplicate
    if ir.shape[1] == 1:
        return np.repeat(ir, 2, axis=1).astype(np.float32, copy=False)

    # Already stereo
    if ir.shape[1] == 2:
        return ir.astype(np.float32, copy=False)

    # >2ch
    if method == "rms":
        weights = np.array([_rms(ir[:, i]) for i in range(ir.shape[1])], dtype=np.float64)
        if np.allclose(weights, 0):
            weights[:] = 1.0
        weights /= weights.sum()
        mix = np.sum(ir * weights[None, :], axis=1)

        # Normalize to preserve power split to 2 ch
        orig_pow = float(np.sum(np.square(ir, dtype=np.float64)))
        pre_pow = float(np.sum(np.square(mix, dtype=np.float64)))
        if pre_pow > 0:
            mix *= np.sqrt(orig_pow / (pre_pow * 2.0))
        stereo = np.stack([mix, mix], axis=1)
        return stereo.astype(np.float32, copy=False)

    # mean – split in half to L/R then average
    mid = ir.shape[1] // 2
    l_chunk, r_chunk = ir[:, :mid], ir[:, mid:]
    l_mix = l_chunk.mean(axis=1)
    r_mix = r_chunk.mean(axis=1)
    return np.stack([l_mix, r_mix], axis=1).astype(np.float32, copy=False)


def _downmix_ir(ir: np.ndarray, mode: Literal["auto", "stereo", "none"] = "auto") -> np.ndarray:
    """
    Down-mix IR according to *mode*.

    * ``"none"``   – no change.  
    * ``"stereo"`` – force stereo.  
    * ``"auto"``   – <=2 ch keep, else stereo mix.
    """
    if mode == "none":
        return ir
    if mode == "stereo":
        return _mix_to_stereo(ir)
    # auto
    if ir.ndim == 1 or ir.shape[1] <= 2:
        return ir
    return _mix_to_stereo(ir)


def _fft_convolve(sig: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Single-channel FFT convolution."""
    n = len(sig) + len(ir) - 1
    nfft = _next_pow2(n)
    S = np.fft.rfft(sig, nfft)
    H = np.fft.rfft(ir, nfft)
    return np.fft.irfft(S * H, nfft)[:n]


def convolve_ir(
    audio: np.ndarray,
    ir: np.ndarray,
    block_size: int = 2**14,
    *,
    progress: bool = False,
) -> np.ndarray:
    """Overlap-add FFT convolution (handles multi-channel)."""
    if audio.ndim == 1:
        audio = audio[:, None]
    if ir.ndim == 1:
        ir = ir[:, None]

    out_len = audio.shape[0] + ir.shape[0] - 1
    channels = max(audio.shape[1], ir.shape[1])

    # Small IR → simple mode
    if ir.shape[0] < 2**14:
        res = [
            _fft_convolve(
                audio[:, ch] if audio.shape[1] > 1 else audio[:, 0],
                ir[:, ch] if ir.shape[1] > 1 else ir[:, 0],
            )
            for ch in range(channels)
        ]
        return np.stack(res, axis=1)[:out_len]

    # Large IR → block FFT
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


# ---------------------------------------------------------------------------#
# IR cache
# ---------------------------------------------------------------------------#
_IR_CACHE_SIZE = int(os.environ.get("CONVOLVER_IR_CACHE", "8"))


@lru_cache(maxsize=_IR_CACHE_SIZE)
def load_ir(path: str) -> tuple[np.ndarray, int]:
    """Load IR WAV → (data, sr)."""
    if sf is None:
        raise RuntimeError("soundfile is required for load_ir")
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.astype(np.float32), int(sr)


# ---------------------------------------------------------------------------#
# Rendering helpers
# ---------------------------------------------------------------------------#
def _apply_tpdf_dither(data: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth == 32:
        return data
    if bit_depth not in (16, 24):
        return data
    lsb = 1.0 / (2**31) if bit_depth == 24 else 1.0 / (2**15)
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
    """Quick tail-fade to avoid clicks."""
    mag = np.max(np.abs(data), axis=1) if data.ndim > 1 else np.abs(data)
    peak = float(np.max(mag))
    if peak == 0:
        return data
    thresh = peak * (10 ** (drop_db / 20.0))
    idx = np.where(mag > thresh)[0]
    start = idx[-1] if idx.size else 0
    fade_len = min(len(data) - start, int(sr * ms / 1000.0))
    if fade_len <= 0:
        return data
    fade = np.linspace(1.0, 0.0, fade_len)[:, None] if data.ndim > 1 else np.linspace(1.0, 0.0, fade_len)
    out = data.copy()
    out[start : start + fade_len] *= fade
    end = min(start + fade_len, max_len) if max_len is not None else start + fade_len
    return out[:end]


def _quantize_pcm(data: np.ndarray, bit_depth: int) -> tuple[np.ndarray, str]:
    if bit_depth == 32:
        return data.astype(np.float32), "FLOAT"
    max_val = float(2**31 - 1) if bit_depth == 24 else float(2 ** (bit_depth - 1) - 1)
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
    """
    Convolve *input_wav* with *ir_wav* and write *out_wav*.

    Fallbacks gracefully if optional libs are missing.
    """
    if sf is None:  # pragma: no cover
        warnings.warn("soundfile not installed – skipping convolution", RuntimeWarning)
        shutil.copyfile(input_wav, out_wav)
        return Path(out_wav)

    inp, irp, out = Path(input_wav), Path(ir_wav), Path(out_wav)
    gain_db = gain_db or 0.0
    dither = bool(dither and normalize)

    # ------------------------------------------------------------------- load
    try:
        y, sr = sf.read(inp, always_2d=True)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read WAV: %s", exc)
        return out

    if not irp.is_file():
        logger.warning("IR file missing: %s – dry render", ir_wav)
        if y.shape[1] == 1:
            y = np.broadcast_to(y, (y.shape[0], 2))
        if dither:
            y = _apply_tpdf_dither(y, bit_depth)
        pcm, subtype = _quantize_pcm(y, bit_depth)
        sf.write(str(out), pcm, sr, subtype=subtype)
        return out

    ir, ir_sr = load_ir(str(irp))

    # ---------------------------------------------------------------- resample
    target_sr = 44100
    y = _resample(y, sr, target_sr, quality=quality)
    ir = _resample(ir, ir_sr, target_sr, quality=quality)
    ir = _downmix_ir(ir, downmix)
    sr = target_sr

    if oversample > 1:
        y = _resample(y, sr, sr * oversample, quality=quality)
        ir = _resample(ir, sr, sr * oversample, quality=quality)
        sr *= oversample

    # ---------------------------------------------------------------- convo
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

    # ---------------------------------------------------------------- gain / LUFS
    if gain_db:
        data *= 10 ** (gain_db / 20.0)

    if normalize:
        peak = float(np.max(np.abs(data)))
        if peak > 1.0:
            data /= peak

    if lufs_target is not None and pyln is not None:
        try:
            meter = pyln.Meter(sr)
            diff = lufs_target - float(meter.integrated_loudness(data))
            diff = max(-3.0, min(3.0, diff))
            data *= 10 ** (diff / 20.0)
        except Exception as exc:  # pragma: no cover
            logger.debug("pyloudnorm failed: %s", exc)
    elif lufs_target is not None:
        logger.warning("pyloudnorm not installed; LUFS normalization skipped")

    # ---------------------------------------------------------------- dither / write
    if dither:
        data = _apply_tpdf_dither(data, bit_depth)
    pcm, subtype = _quantize_pcm(data, bit_depth)
    sf.write(str(out), pcm, sr, subtype=subtype)
    return out


# ---------------------------------------------------------------------------#
# External helpers
# ---------------------------------------------------------------------------#
def normalize_velocities(parts: list[stream.Part] | dict[str, stream.Part]) -> None:
    """Normalize average note velocities across parts."""
    all_parts = list(parts.values()) if isinstance(parts, dict) else list(parts)
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
            n.volume.velocity = int(max(1, min(127, round(n.volume.velocity * scale))))  # type: ignore[arg-type]


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
    """
    Render *midi_path* with Fluidsynth then apply IR.

    If *parts* is supplied, velocities are normalized and a temporary MIDI
    is rendered instead.
    """
    from utilities.synth import render_midi  # lazy import

    if sf is None:  # pragma: no cover
        warnings.warn("soundfile not installed – skipping render_wav", RuntimeWarning)
        Path(out_path).touch()
        return Path(out_path)

    tmp_midi: Path | None = None
    if parts is not None:
        normalize_velocities(parts)
        score = stream.Score()
        for p in parts.values() if isinstance(parts, Mapping) else parts:
            score.insert(0, p)
        tmp_midi = Path(out_path).with_suffix(".norm.mid")
        score.write("midi", fp=str(tmp_midi))
        midi_in = tmp_midi
    else:
        midi_in = Path(midi_path)

    dry_wav = Path(out_path).with_suffix(".dry.wav")
    render_midi(midi_in, dry_wav, sf2_path=sf2)

    # legacy 'mix_opts' shim
    if "mix_opts" in kw:
        warnings.warn("'mix_opts' dict is deprecated; pass keyword args instead", DeprecationWarning)
        if isinstance(kw["mix_opts"], Mapping):
            kw.update(kw.pop("mix_opts"))

    render_with_ir(
        dry_wav,
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
    dry_wav.unlink(missing_ok=True)
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
