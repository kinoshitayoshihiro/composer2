from random import Random

from utilities import groove_sampler_ngram as n


def _simple_model() -> n.Model:
    return {
        "version": n.VERSION,
        "resolution": n.RESOLUTION,
        "order": 1,
        "freq": {0: {(): {(0, "kick"): 1}}},
        "prob": {0: {(): {(0, "kick"): 0.0}}},
        "mean_velocity": {},
        "vel_deltas": {},
        "micro_offsets": {},
        "aux_cache": {},
        "use_sha1": False,
        "num_tokens": 1,
        "train_perplexity": 0.0,
        "train_seconds": 0.0,
    }


def test_lin_prob_lru() -> None:
    model = _simple_model()
    rng = Random(0)
    n._lin_prob.clear()
    for i in range(3000):
        hist = [(i, "kick")]
        n._sample_next(hist, model, rng, top_k=None)
    assert len(n._lin_prob) <= n.MAX_CACHE
