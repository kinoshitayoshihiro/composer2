# cython: boundscheck=False, wraparound=False
import random
from music21 import volume, note


def postprocess_kick_lock(part_stream, kick_offsets, int vel_shift, double eps=0.03):
    """Boost velocity of notes near kick offsets."""
    if not kick_offsets:
        return
    for n in part_stream.recurse().notes:
        pos = float(n.offset)
        for k in kick_offsets:
            if abs(pos - float(k)) < eps:
                if n.volume is None:
                    n.volume = volume.Volume(velocity=64)
                n.volume.velocity = min(127, (n.volume.velocity or 64) + vel_shift)
                break

def velocity_random_walk(part_stream, double bar_len, int step_range, rnd=None):
    """Apply bar-by-bar velocity fluctuation."""
    cdef double current_bar = 0.0
    cdef int state = 0
    notes = sorted(part_stream.recurse().notes, key=lambda n: n.offset)
    rng = rnd or random
    for n in notes:
        while float(n.offset) >= current_bar + bar_len - 1e-6:
            current_bar += bar_len
            step = rng.randint(-step_range, step_range)
            state = max(-step_range, min(step_range, state + step))
        if n.volume is None:
            n.volume = volume.Volume(velocity=64)
        n.volume.velocity = max(1, min(127, (n.volume.velocity or 64) + state))


def apply_velocity_curve(events, curve):
    """Scale event velocity factors by ``curve`` list."""
    if not curve:
        return
    for i in range(len(events)):
        ev = events[i]
        scale = float(curve[min(i, len(curve)-1)])
        ev['velocity_factor'] = ev.get('velocity_factor', 1.0) * scale


from music21 import stream

def insert_melody_notes(part, notes, offsets, durations, int velocity, double density, rnd=None):
    """Insert notes into ``part`` with given offsets and durations."""
    rng = rnd or random
    cdef int i
    for i in range(len(notes)):
        if rng.random() > density:
            continue
        n = notes[i]
        n.quarterLength = durations[i]
        if n.volume is None:
            n.volume = volume.Volume(velocity=velocity)
        else:
            n.volume.velocity = velocity
        part.insert(offsets[i], n)
