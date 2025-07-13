import numpy as np
from random import Random
from utilities import groove_sampler_v2


def _make_model() -> groove_sampler_v2.NGramModel:
    freq = [ {0: np.array([6, 3, 1], dtype=np.uint32)} ]
    bucket = {0: np.array([6, 3, 1], dtype=np.uint32)}
    ctx = [{0: 0}]
    prob = [np.array([[0.6, 0.3, 0.1]], dtype=np.float32)]
    return groove_sampler_v2.NGramModel(
        n=1,
        resolution=16,
        resolution_coarse=16,
        state_to_idx={(0,0,'a'):0,(0,1,'b'):1,(0,2,'c'):2},
        idx_to_state=[(0,0,'a'),(0,1,'b'),(0,2,'c')],
        freq=freq,
        bucket_freq=bucket,
        ctx_maps=ctx,
        prob_paths=None,
        prob=prob,
    )


def test_topk_and_topp() -> None:
    model = _make_model()
    rng = Random(0)
    results_k = {groove_sampler_v2.sample_next(model, [], 0, rng, top_k=1) for _ in range(20)}
    assert results_k == {0}
    rng = Random(0)
    results_p = {groove_sampler_v2.sample_next(model, [], 0, rng, top_p=0.5) for _ in range(20)}
    assert results_p == {0}

