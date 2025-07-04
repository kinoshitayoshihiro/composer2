# cython: boundscheck=False, wraparound=False
import math
import random
import copy
from music21 import volume

def apply_swing(part_stream, double swing_ratio, int subdiv=8):
    if subdiv <= 0:
        return
    step = 4.0 / subdiv
    pair = step * 2.0
    tol = step * 0.1
    for n in part_stream.recurse().notes:
        pos = float(n.offset)
        pair_start = math.floor(pos / pair) * pair
        within = pos - pair_start
        if abs(within - step) < tol:
            n.offset = pair_start + pair * swing_ratio

cpdef humanize_velocities(part_stream, int amount=4,
                          bint use_expr_cc11=False,
                          bint use_aftertouch=False,
                          str expr_curve="cubic-in",
                          int kick_leak_jitter=0):
    cdef list kicks = []
    cdef double thr = 0.0
    cdef bint need_jitter = kick_leak_jitter != 0
    if need_jitter:
        for n in part_stream.recurse().notes:
            if int(n.pitch.midi) == 36:
                kicks.append(float(n.offset))
        bpm_el = part_stream.recurse().getElementsByClass('MetronomeMark').first()
        bpm = bpm_el.number if bpm_el is not None else 120
        thr = 0.1 * bpm / 60.0

    cdef double last_cc = 0.0
    for n in part_stream.recurse().notes:
        vel = getattr(n.volume, 'velocity', None)
        if vel is None:
            vel = 64
            if n.volume is None:
                n.volume = volume.Volume(velocity=vel)
            else:
                n.volume.velocity = vel
        delta = random.randint(-amount, amount)
        new_vel = max(1, min(127, vel + delta))
        if need_jitter and int(n.pitch.midi) in (42, 44, 46):
            for k in kicks:
                if abs(float(n.offset) - k) <= thr:
                    jitter = random.randint(-kick_leak_jitter, kick_leak_jitter)
                    new_vel = max(1, min(127, new_vel + jitter))
                    break
        n.volume.velocity = new_vel
        cc_val = new_vel
        if expr_curve == "cubic-in":
            cc_val = int(max(1, min(127, (new_vel / 127.0) ** 3 * 127)))
        cc_val = max(1, min(127, cc_val))
        if cc_val < last_cc:
            cc_val = last_cc
        if use_expr_cc11 or use_aftertouch:
            if not hasattr(part_stream, "extra_cc"):
                part_stream.extra_cc = []
        if use_expr_cc11:
            part_stream.extra_cc.append({"time": n.offset, "cc": 11, "val": cc_val})
        if use_aftertouch:
            part_stream.extra_cc.append({"time": n.offset, "cc": 74, "val": cc_val})
        last_cc = cc_val

def apply_envelope(part_stream, int start, int end, double scale):
    """Cython-accelerated envelope scaling."""
    for n in part_stream.recurse().notes:
        pos = float(n.offset)
        if start <= pos < end:
            vel = getattr(n.volume, 'velocity', None)
            if vel is None:
                continue
            new_vel = int(max(1, min(127, round(vel * scale))))
            n.volume.velocity = new_vel

def apply_velocity_histogram(part_stream, histogram):
    if not histogram:
        return part_stream
    choices = [int(v) for v in histogram.keys()]
    weights = [float(w) for w in histogram.values()]
    for n in part_stream.recurse().notes:
        vel = random.choices(choices, weights)[0]
        if n.volume is None:
            n.volume = volume.Volume(velocity=vel)
        else:
            n.volume.velocity = vel
    return part_stream

def timing_correct_part(part_stream, double alpha):
    new_part = copy.deepcopy(part_stream)
    notes = list(new_part.recurse().notes)
    if not notes:
        return new_part
    ema = float(notes[0].offset) - round(float(notes[0].offset))
    for n in notes:
        target = round(float(n.offset))
        delta = float(n.offset) - target
        ema += alpha * (delta - ema)
        n.offset = target + ema
    return new_part
