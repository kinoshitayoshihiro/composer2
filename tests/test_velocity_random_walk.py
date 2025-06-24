import random

from utilities.accent_mapper import AccentMapper

def test_velocity_random_walk_per_bar():
    rng = random.Random(0)
    heatmap = {0: 10}
    settings = {"accent_threshold": 0.0, "ghost_density_range": (0.3, 0.8)}
    mapper = AccentMapper(heatmap, settings, rng=rng)
    base = 80
    velocities = []
    for _ in range(8):
        mapper.begin_bar()
        velocities.append(mapper.accent(0, base))
    diffs = [abs(velocities[i] - velocities[i - 1]) for i in range(1, len(velocities))]
    assert all(base <= v <= int(base * 1.2 + 1) for v in velocities)
    assert all(d <= 16 for d in diffs)

