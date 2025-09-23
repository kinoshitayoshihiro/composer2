from __future__ import annotations

import math
from typing import List, Tuple

try:  # Optional dependency
    import numpy as np
except Exception:  # pragma: no cover - numpy may be absent
    np = None  # type: ignore

try:  # Optional dependency
    import librosa  # type: ignore
except Exception:  # pragma: no cover - librosa may be absent
    librosa = None  # type: ignore


def energy_to_cc11(
    y: "np.ndarray | List[float]",
    sr: int,
    hop_ms: int = 10,
    smooth_ms: int = 80,
    *,
    strategy: str = "energy",
) -> List[Tuple[float, int]]:
    """Return ``(time, value)`` pairs for CC11 derived from audio energy.

    Parameters
    ----------
    y:
        Mono audio samples.
    sr:
        Sample rate of ``y``.
    hop_ms:
        Hop size in milliseconds for the analysis frames.
    smooth_ms:
        Exponential moving average window in milliseconds.
    strategy:
        Either ``"energy"`` (short‑time energy) or ``"rms"``.  ``"rms"``
        requires ``librosa``; when unavailable, falls back to ``"energy"``.
    """

    if np is None:
        return []
    hop = max(1, int(sr * hop_ms / 1000))
    rms = None
    if librosa is not None:
        try:
            if strategy == "rms":
                rms = librosa.rms(y=y, hop_length=hop)[0]
            else:
                rms = librosa.feature.rms(
                    y=y, frame_length=2 * hop, hop_length=hop, center=True
                )[0]
        except Exception:  # pragma: no cover - fallback below
            rms = None
    if rms is None:
        win = hop
        padded = np.pad(np.asarray(y, dtype=float), (win // 2, win // 2), mode="constant")
        rms = np.sqrt(
            np.convolve(padded ** 2, np.ones(win) / win, mode="valid")[::hop]
        )
    if not rms.size:
        return []
    x = rms - rms.min()
    if x.max() > 0:
        x = x / x.max()
    alpha = 1.0
    if smooth_ms > 0:
        alpha = 1 - math.exp(-hop_ms / smooth_ms)
    ema = []
    z = 0.0
    for v in x:
        z = alpha * v + (1 - alpha) * z
        ema.append(z)
    if not ema:
        return []
    ema_arr = np.asarray(ema, dtype=float)
    times = (np.arange(ema_arr.size, dtype=float) * hop) / float(sr)
    if smooth_ms > 0 and ema_arr.size:
        window = int(round(sr * smooth_ms / 1000.0))
        window = max(1, min(window, ema_arr.size))
        if window > 1:
            pad = window // 2
            padded = np.pad(ema_arr, (pad, pad), mode="edge")
            smoothed = np.convolve(padded, np.ones(window) / float(window), mode="valid")
            if smoothed.size > ema_arr.size:
                smoothed = smoothed[: ema_arr.size]
            ema_arr = smoothed
        step = max(1, int(round(smooth_ms / hop_ms))) if hop_ms > 0 else 1
        if step > 1 and ema_arr.size > 2:
            idx = list(range(0, ema_arr.size, step))
            last_idx = ema_arr.size - 1
            if idx[-1] != last_idx:
                idx.append(last_idx)
            dedup_idx: list[int] = []
            prev_idx = -1
            for i in idx:
                if 0 <= i < ema_arr.size and i != prev_idx:
                    dedup_idx.append(i)
                    prev_idx = i
            if dedup_idx:
                ema_arr = ema_arr[dedup_idx]
                times = times[dedup_idx]
    clipped = np.clip(ema_arr, 0.0, 1.0)
    values = np.rint(clipped * 127.0).astype(int)
    if values.size == 0:
        return []
    times_list = times.tolist()
    vals_list = values.tolist()
    events: List[Tuple[float, int]] = []
    first_val = int(vals_list[0])
    events.append((float(times_list[0]), first_val))
    prev_val = first_val
    for idx in range(1, len(vals_list) - 1):
        val = int(vals_list[idx])
        if val != prev_val:
            events.append((float(times_list[idx]), val))
            prev_val = val
    if len(vals_list) > 1:
        last_val = int(vals_list[-1])
        last_time = float(times_list[-1])
        if last_val != prev_val:
            events.append((last_time, last_val))
        else:
            events[-1] = (last_time, events[-1][1])
    return events


def infer_cc64_from_overlaps(
    notes: List["pretty_midi.Note"], threshold: float
) -> List[Tuple[float, int]]:
    """Infer simple sustain pedal events from note overlaps."""
    if threshold <= 0 or not notes:
        return []
    try:
        import pretty_midi
    except Exception:  # pragma: no cover - pretty_midi may be absent
        return []
    events: List[Tuple[float, int]] = []
    sorted_notes = sorted(notes, key=lambda n: (float(n.start), float(n.end)))
    for a, b in zip(sorted_notes, sorted_notes[1:]):
        gap = float(b.start) - float(a.end)
        if 0 < gap <= threshold:
            events.append((float(a.end), 127))
            events.append((float(b.start), 0))
    dedup: List[Tuple[float, int]] = []
    prev_val: int | None = None
    for t, v in sorted(events):
        if prev_val is not None and v == prev_val:
            continue
        dedup.append((t, v))
        prev_val = v
    return dedup
