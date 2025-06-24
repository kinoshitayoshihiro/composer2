from utilities.accent_mapper import AccentMapper
import random


def test_no_accent_below_threshold():
    heatmap = {0: 2, 1: 1}
    gs = {"accent_threshold": 0.8, "ghost_density_range": (0.3, 0.5)}
    am = AccentMapper(heatmap, gs, rng=random.Random(0))
    assert am.accent(1, 80) == 80


def test_accent_above_threshold():
    heatmap = {5: 10, 0: 1}
    gs = {"accent_threshold": 0.2, "ghost_density_range": (0.3, 0.5)}
    am = AccentMapper(heatmap, gs, rng=random.Random(0))
    out = am.accent(5, 80)
    assert out > 80


def test_ghost_hat_density_limits():
    heatmap = {}
    gs = {"accent_threshold": 0.5, "ghost_density_range": (0.3, 0.5)}
    am = AccentMapper(heatmap, gs, rng=random.Random(0))
    results = [am.maybe_ghost_hat(i) for i in range(1000)]
    density = sum(results) / len(results)
    assert 0.2 <= density <= 0.6

