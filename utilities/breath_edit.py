from __future__ import annotations

import logging
from array import array
from functools import reduce
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - optional progress bar
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional
    tqdm = None  # type: ignore
    logging.getLogger("breath").debug("tqdm not found; progress bar disabled")

logger = logging.getLogger("breath")

try:
    import pydub
    from pydub import AudioSegment

    pydub_version = getattr(pydub, "__version__", "0")
except Exception:  # pragma: no cover - optional
    AudioSegment = None  # type: ignore
    pydub_version = "0"


def _concat_legacy_no_xfade(segs: list[AudioSegment]) -> AudioSegment:
    base = segs[0]
    arr = array(base.array_type)
    for s in segs:
        arr.extend(s.get_array_of_samples())
    return base._spawn(arr.tobytes())


def process_breath(
    wav_in: Path | str,
    wav_out: Path | str,
    mask: NDArray[np.bool_],
    mode: str,
    attenuate_gain_db: float,
    crossfade_ms: int,
    hop_ms: float | None = None,
) -> None:
    """Edit breaths in ``wav_in`` according to ``mask`` and save to ``wav_out``."""
    if AudioSegment is None:
        raise ImportError("pydub required")

    audio = AudioSegment.from_file(wav_in)
    hop = hop_ms or len(audio) / len(mask)
    frames = int(np.ceil(len(audio) / hop))
    if len(mask) < frames:
        mask = np.pad(mask, (0, frames - len(mask)), False)
    elif len(mask) > frames:
        extra = mask[frames:]
        mask = mask[:frames]
        if extra.any():
            mask[-1] = True

    def _segment(start: int, end: int) -> AudioSegment:
        return audio[int(start * hop) : int(end * hop)]

    segments: list[AudioSegment] = []
    gaps: list[int] = []
    last_gap = 0
    pbar = tqdm(total=len(mask)) if tqdm is not None else None
    i = 0
    while i < len(mask):
        start = i
        cur = mask[i]
        while i < len(mask) and mask[i] == cur:
            i += 1
        seg = _segment(start, i)
        if cur:
            if mode == "attenuate":
                seg = seg.apply_gain(attenuate_gain_db)
                segments.append(seg)
                gaps.append(last_gap)
                last_gap = 0
            elif mode == "keep":
                segments.append(seg)
                gaps.append(last_gap)
                last_gap = 0
            else:
                last_gap += len(seg)
        else:
            segments.append(seg)
            gaps.append(last_gap)
            last_gap = 0
        if pbar is not None:
            pbar.update(i - start)
    if pbar is not None:
        pbar.close()

    if mode == "remove":
        kept = [seg for seg in segments if len(seg) > 0]
        if not kept:
            out = AudioSegment.silent(duration=0)
        else:
            ver = tuple(int(v) for v in pydub_version.split(".")[:2])
            if ver < (0, 25) and crossfade_ms == 0:
                out = _concat_legacy_no_xfade(kept)
            else:
                out = kept[0]
                for seg, gap in zip(kept[1:], gaps[1:]):
                    xfade = 0
                    if crossfade_ms > 0 and len(out) > 0 and len(seg) > 0:
                        if len(seg) < 2 * crossfade_ms:
                            fade_ms = int(min(5, len(seg) // 4))
                            if fade_ms > 0:
                                out = out.fade_out(fade_ms)
                                seg = seg.fade_in(fade_ms)
                        elif gap < 2 * crossfade_ms:
                            out = out.fade_out(crossfade_ms)
                            seg = seg.fade_in(crossfade_ms)
                        else:
                            xfade = int(min(crossfade_ms, len(out), len(seg)))
                    out = out.append(seg, crossfade=xfade)
    else:
        if not segments:
            out = AudioSegment.silent(duration=0)
        else:
            ver = tuple(int(v) for v in pydub_version.split(".")[:2])
            if ver < (0, 25):
                out = _concat_legacy_no_xfade(segments)
            else:
                if len(segments) > 2000:
                    out = sum(segments[1:], segments[0])
                else:
                    out = reduce(
                        lambda a, b: a.append(b, crossfade=0), segments[1:], segments[0]
                    )

    out.export(wav_out, format="wav")
    logger.info("wrote %s", wav_out)


__all__ = ["process_breath"]
