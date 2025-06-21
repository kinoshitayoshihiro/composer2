import random

from utilities.velocity_smoother import EmaSmoother


def test_velocity_ema_reduces_diff():
    rng = random.Random(0)
    raw = [rng.randint(50, 120) for _ in range(20)]
    smoother = EmaSmoother()
    smoothed = [smoother.smooth(v) for v in raw]
    diff_raw = [abs(raw[i] - raw[i-1]) for i in range(1, len(raw))]
    diff_smooth = [abs(smoothed[i] - smoothed[i-1]) for i in range(1, len(smoothed))]
    avg_raw = sum(diff_raw) / len(diff_raw)
    avg_smooth = sum(diff_smooth) / len(diff_smooth)
    assert avg_smooth <= avg_raw * 0.85
