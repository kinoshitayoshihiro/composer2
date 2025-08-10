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
    return [(i * hop / sr, int(round(127 * v))) for i, v in enumerate(ema)]


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
