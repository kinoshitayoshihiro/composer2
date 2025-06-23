import numpy as np
from utilities.velocity_smoother import EMASmoother


def test_adaptive_alpha_variance() -> None:
    low_vals = [60 + (i % 3 - 1) for i in range(32)]
    high_vals = [60 if i % 2 == 0 else 100 for i in range(32)]

    sm = EMASmoother()
    out_low = [sm.update(v) for v in low_vals]
    var_low_ratio = np.var(out_low) / np.var(low_vals)

    sm.reset()
    out_high = [sm.update(v) for v in high_vals]
    var_high_ratio = np.var(out_high) / np.var(high_vals)

    assert var_high_ratio > var_low_ratio
