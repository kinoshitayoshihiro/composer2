from utilities.velocity_smoother import VelocitySmoother


def test_velocity_smoother_adapts_alpha() -> None:
    raw = [80, 82, 81, 83, 120]
    smoother = VelocitySmoother()
    out = [smoother.smooth(v) for v in raw]
    # first four values should stay near each other
    diffs = [abs(out[i] - out[i - 1]) for i in range(1, 4)]
    assert max(diffs) < 3
    # last value should react to the large jump
    assert out[-1] > 100

