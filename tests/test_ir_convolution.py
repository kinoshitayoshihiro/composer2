import numpy as np
from utilities.convolver import convolve_ir


def test_fft_equals_numpy():
    rng = np.random.default_rng(0)
    audio = rng.normal(size=32).astype(np.float32)
    ir = rng.normal(size=8).astype(np.float32)
    out = convolve_ir(audio, ir)
    ref = np.convolve(audio, ir)[: len(audio) + len(ir) - 1]
    assert np.allclose(out[:, 0], ref, atol=1e-6)
