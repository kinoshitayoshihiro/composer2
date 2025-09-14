from __future__ import annotations
import pretty_midi  # type: ignore
from typing import Tuple, List, Optional

from .consts import DAMP_INST_NAME

EPS = 1e-9


def clip_note_interval(start_t: float, end_t: float, *, eps: float = EPS) -> Tuple[float, float]:
    if start_t < 0:
        start_t = 0.0
    if end_t < start_t + eps:
        end_t = start_t + eps
    return start_t, end_t


def _append_phrase(inst, pitch: int, start: float, end: float, vel: int,
                   merge_gap_sec: float, release_sec: float, min_len_sec: float,
                   stats=None):
    if end <= start + EPS:
        return
    end -= release_sec
    start, end = clip_note_interval(start, end, eps=EPS)
    if min_len_sec > 0.0 and end < start + min_len_sec - EPS:
        end = start + min_len_sec
    if merge_gap_sec < 0 or not inst.notes:
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))
        return
    if inst.notes[-1].pitch == pitch and (start - inst.notes[-1].end) <= merge_gap_sec + EPS:
        inst.notes[-1].end = max(inst.notes[-1].end, end)
        if stats is not None:
            stats.setdefault('merge_events', []).append({'pitch': pitch, 'start': start})
    else:
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))


def _legato_merge_chords(inst, merge_gap_sec: float):
    last_by_pitch = {}
    merged = []
    for n in sorted(inst.notes, key=lambda x: (x.pitch, x.start)):
        prev = last_by_pitch.get(n.pitch)
        if prev and (n.start - prev.end) <= merge_gap_sec + EPS:
            prev.end = max(prev.end, n.end)
        else:
            merged.append(n)
            last_by_pitch[n.pitch] = n
    inst.notes = sorted(merged, key=lambda x: x.start)


def emit_damping(pm_out, mode: str, **kw):
    """Emit damping control changes into `pm_out`.

    Parameters
    ----------
    pm_out : pretty_midi.PrettyMIDI
        MIDI container to modify.
    mode : str
        'none', 'fixed', or 'vocal'.
    kw : dict
        Additional options depending on mode.
    """
    if mode == "none":
        return pm_out
    cc_num = int(kw.get('cc', 64))
    channel = int(kw.get('channel', 0))
    inst = pretty_midi.Instrument(program=0, name=DAMP_INST_NAME)
    inst.midi_channel = channel
    if mode == "fixed":
        val = int(kw.get('value', 0))
        inst.control_changes.append(pretty_midi.ControlChange(cc_num, val, 0.0))
        pm_out.instruments.append(inst)
        return pm_out
    if mode == "vocal":
        ratios: List[float] = kw.get('vocal_ratios') or []
        downbeats: List[float] = kw.get('downbeats') or []
        if not downbeats or not ratios:
            return pm_out
        smooth = int(kw.get('smooth', 1))
        if smooth > 1:
            ratios = _moving_average(ratios, smooth)
        clip = kw.get('clip', (0, 127))
        min_beats = float(kw.get('min_beats', 0.0))
        deadband = float(kw.get('deadband', 0.0))
        last_val: Optional[int] = None
        last_t = -1e9
        for i, r in enumerate(ratios):
            if i >= len(downbeats):
                break
            val = int(round(max(clip[0], min(clip[1], r * 127))))
            t = downbeats[i]
            if last_val is not None:
                if abs(val - last_val) <= deadband:
                    continue
                if t - last_t < min_beats:
                    continue
            inst.control_changes.append(pretty_midi.ControlChange(cc_num, val, t))
            last_val = val
            last_t = t
        if inst.control_changes:
            pm_out.instruments.append(inst)
        return pm_out
    return pm_out


def _moving_average(vals: List[float], k: int) -> List[float]:
    out: List[float] = []
    for i in range(len(vals)):
        s = 0.0
        c = 0
        for j in range(i - k + 1, i + k):
            if 0 <= j < len(vals):
                s += vals[j]
                c += 1
        out.append(s / c if c else 0.0)
    return out
