import random
from generator.drum_generator import AccentMapper

def test_velocity_random_walk_per_bar():
    random.seed(0)
    mapper = AccentMapper()
    base = 80
    walk_range = 4
    velocities = []
    for _ in range(8):
        mapper.begin_bar(walk_range=walk_range)
        velocities.append(mapper.get_velocity(0.0, base))
    diffs = [abs(velocities[i] - velocities[i-1]) for i in range(1, len(velocities))]
    assert all(d <= walk_range for d in diffs)
    assert sum(diffs) / len(diffs) > 0

