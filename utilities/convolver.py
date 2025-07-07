from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from music21 import stream

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
        from scipy.signal import resample_poly  # type: ignore
    except Exception:  # pragma: no cover - optional
        resample_poly = None  # type: ignore
else:
    resample_poly = None  # type: ignore
try:
    if sf is not None:
        try:
            sf.default_subtype("WAV", subtype="FLOAT")  # type: ignore[arg-type]
        except TypeError:
            # Older soundfile versions only expose a mapping
            if hasattr(sf, "_default_subtypes"):
                sf._default_subtypes["WAV"] = "FLOAT"  # type: ignore[attr-defined]
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


def _write_gain(
    data: np.ndarray, sr: int, path: Path, gain_db: float, bit_depth: str
) -> None:
    """Write *data* to *path* applying gain and normalisation."""
    if sf is None:
        raise RuntimeError("soundfile is required for WAV rendering")
    if gain_db:
        data = data * (10 ** (gain_db / 20.0))
    peak = np.max(np.abs(data))
    if peak > 1.0:
        data = data / peak
    subtype_map = {"16": "PCM_16", "24": "PCM_24", "32f": "FLOAT"}
    subtype = subtype_map.get(bit_depth, "PCM_16")
    sf.write(path, data.astype(np.float32), sr, subtype=subtype)


def render_with_ir(
    input_wav: str | Path,
    ir_wav: str | Path,
    out_wav: str | Path,
    *,
    lufs_target: float | None = None,
    gain_db: float | None = None,
    block_size: int = 2**14,
    bit_depth: str = "16",
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
    if gain_db is None:
        gain_db = 0.0

    try:
        y, sr = sf.read(inp, always_2d=True)
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - best effort
        logger.warning("Failed to read WAV: %s", exc)
        return

    if not irp.is_file():
        logger.warning("IR file missing: %s", ir_wav)
        if y.shape[1] == 1:
            y = np.broadcast_to(y, (y.shape[0], 2))
        _write_gain(y, sr, out, gain_db, bit_depth)
        return

    try:
        ir, ir_sr = sf.read(irp, always_2d=True)
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - best effort
        logger.warning("Failed to read IR: %s", exc)
        _write_gain(y, sr, out, gain_db, bit_depth)
        return

    if sr != ir_sr:
        target_sr = 44100
        if soxr is not None:
            y = soxr.resample(y, sr, target_sr, quality="mq")
            ir = soxr.resample(ir, ir_sr, target_sr, quality="mq")
        else:
            if resample_poly is None:
                from scipy.signal import resample_poly as _rp

                y = _rp(y, target_sr, sr, axis=0)
                ir = _rp(ir, target_sr, ir_sr, axis=0)
            else:
                y = resample_poly(y, target_sr, sr, axis=0)
                ir = resample_poly(ir, target_sr, ir_sr, axis=0)
        sr = ir_sr = target_sr

    if ir.shape[1] == 1 and y.shape[1] > 1:
        ir = np.broadcast_to(ir, (ir.shape[0], y.shape[1]))
    if y.shape[1] == 1 and ir.shape[1] > 1:
        y = np.broadcast_to(y, (y.shape[0], ir.shape[1]))

    if y.shape[0] > 2**18:
        data = convolve_ir(y, ir, block_size=block_size, progress=progress)
    else:
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

    _write_gain(data, sr, out, gain_db, bit_depth)
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
    audio: np.ndarray,
    ir: np.ndarray,
    block_size: int = 2**14,
    *,
    progress: bool = False,
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
    bar = tqdm(total=audio.shape[0], disable=not progress, desc="IR", leave=False)
    while pos < audio.shape[0]:
        chunk = audio[pos : pos + block_size]
        pad = np.zeros((fft_size, audio.shape[1]), dtype=np.float64)
        pad[: chunk.shape[0]] = chunk
        X = np.fft.rfft(pad, axis=0)
        y = np.fft.irfft(X * H, axis=0)[:fft_size]
        result[pos : pos + fft_size, : audio.shape[1]] += y
        pos += block_size
        bar.update(chunk.shape[0])
    bar.close()
    return result[: audio.shape[0]].astype(np.float32)


def normalize_velocities(parts: list[stream.Part] | dict[str, stream.Part]) -> None:
    """Scale note velocities of *parts* so averages match."""
    if isinstance(parts, dict):
        all_parts = list(parts.values())
    else:
        all_parts = list(parts)
    if not all_parts:
        return
    avgs = []
    for p in all_parts:
        vals = [n.volume.velocity or 0 for n in p.recurse().notes if n.volume]
        if not vals:
            continue
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
    parts: list[stream.Part] | dict[str, stream.Part] | None = None,
    **mix_opts,
) -> Path:
    """Render ``midi_path`` with ``fluidsynth`` and apply ``ir_path``."""
    from utilities.synth import render_midi

    if sf is None:
        raise RuntimeError("soundfile is required for render_wav")

    tmp_midi: Path | None = None
    if parts is not None:
        normalize_velocities(parts)
        score = stream.Score()
        for p in (parts.values() if isinstance(parts, dict) else parts):
            score.insert(0, p)
        tmp_midi = Path(out_path).with_suffix(".norm.mid")
        score.write("midi", fp=str(tmp_midi))
        midi_in = tmp_midi
    else:
        midi_in = Path(midi_path)

    tmp = Path(out_path).with_suffix(".dry.wav")
    render_midi(midi_in, tmp, sf2_path=sf2)
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
