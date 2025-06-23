from utilities.velocity_smoother import VelocitySmoother
from utilities.velocity_smoother import EMASmoother
import statistics


def test_velocity_smoother_adapts_alpha() -> None:
    raw = [80, 82, 81, 83, 120]
    smoother = VelocitySmoother()
    out = [smoother.smooth(v) for v in raw]
    # first four values should stay near each other
    diffs = [abs(out[i] - out[i - 1]) for i in range(1, 4)]
    assert max(diffs) < 3
    # last value should react to the large jump
    assert out[-1] > 100


def test_velocity_spike_reduction() -> None:
    raw = [64, 65, 120, 66, 67]
    sm = EMASmoother(window=16)
    out = [sm.smooth(v) for v in raw]
    stdev_raw = statistics.pstdev(raw)
    stdev_out = statistics.pstdev(out)
    assert stdev_out <= stdev_raw * 0.7


def test_velocity_invariance_on_flat_line() -> None:
    raw = [80] * 16
    sm = EMASmoother(window=16)
    out = [sm.smooth(v) for v in raw]
    assert out == raw
