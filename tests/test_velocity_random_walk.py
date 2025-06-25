import random

from utilities.accent_mapper import AccentMapper

def test_velocity_random_walk_per_bar():
    rng = random.Random(0)
    heatmap = {0: 10}
    settings = {
        "accent_threshold": 0.0,
        "ghost_density_range": (0.3, 0.8),
        "random_walk_step": 8,
    }
    mapper = AccentMapper(heatmap, settings, rng=rng)
    base = 80
    velocities = []
    for _ in range(8):
        mapper.begin_bar()
        velocities.append(mapper.accent(0, base, apply_walk=True))
    step_range = settings["random_walk_step"]
    base_after_accent = int(round(base * 1.2))
    diffs = [abs(velocities[i] - velocities[i - 1]) for i in range(1, len(velocities))]
    assert all(
        base_after_accent - step_range <= v <= base_after_accent + step_range
        for v in velocities
    )
    assert all(d <= 2 * step_range for d in diffs)

