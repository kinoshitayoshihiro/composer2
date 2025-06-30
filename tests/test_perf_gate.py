import time
from utilities import groove_sampler_ngram as gs

def test_perf_gate() -> None:
    t_train, t_sample_bar = gs.profile_train_sample()
    assert t_train < 5.0
    assert t_sample_bar * 256 < 5.0

