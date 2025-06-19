import random

from generator.drum_generator import AccentMapper

def test_velocity_random_walk_per_bar():
    rng = random.Random(0)
    mapper = AccentMapper(rng=rng)
    base = 80
    walk_range = 8
    velocities = []
    for _ in range(8):
        mapper.begin_bar(walk_range=walk_range)
        velocities.append(mapper.get_velocity(0.0, base, clamp=(base - walk_range, base + walk_range)))
    diffs = [abs(velocities[i] - velocities[i - 1]) for i in range(1, len(velocities))]
    assert all(base - walk_range <= v <= base + walk_range for v in velocities)
    assert all(d <= 8 for d in diffs)

