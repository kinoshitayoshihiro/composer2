import random
from generator.drum_generator import AccentMapper


def test_velocity_random_walk_basic():
    mapper = AccentMapper()
    random.seed(0)
    base = 64
    velocities = []
    for _ in range(8):
        mapper.begin_bar(walk_range=4)
        velocities.append(mapper.get_velocity(0.0, base))
    # differences between consecutive bars should not exceed walk_range
    for a, b in zip(velocities, velocities[1:]):
        assert abs(b - a) <= 4
    avg_dev = sum(abs(v - base) for v in velocities) / len(velocities)
    assert avg_dev > 0
