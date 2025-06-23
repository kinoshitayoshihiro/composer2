import random
import statistics
from utilities.velocity_smoother import EMASmoother


def test_ema_smoother_reduces_variance() -> None:
    rng = random.Random(0)
    vals = [rng.randint(40, 100) for _ in range(32)]
    smoother = EMASmoother()
    out = [smoother.update(v) for v in vals]
    assert statistics.pvariance(out) < statistics.pvariance(vals)
