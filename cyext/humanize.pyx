# cython: boundscheck=False, wraparound=False
import math
import random
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

def humanize_velocities(part_stream, int amount=4):
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
        n.volume.velocity = new_vel
