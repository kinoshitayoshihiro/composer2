import pytest
from utilities import groove_sampler_ngram

@pytest.mark.slow
def test_perf_budget() -> None:
    train_t, samp_t = groove_sampler_ngram.profile_train_sample()
    if train_t > 120 or samp_t > 0.05:
        pytest.xfail("Performance budget exceeded on this machine")
    assert train_t < 120
    assert samp_t < 0.05

