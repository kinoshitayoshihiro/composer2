import pytest
from utilities import groove_sampler_ngram


def test_schema_token_injected(monkeypatch):
    seen = []

    def dummy_next(history, model, rng, **kw):
        seen.append(list(history))
        return (0, "kick")

    monkeypatch.setattr(groove_sampler_ngram, "_sample_next", dummy_next)
    model = {
        "version": 1,
        "resolution": 16,
        "order": 2,
        "freq": {},
        "prob": {0: {(): {(0, "kick"): 0.0}}},
        "mean_velocity": {},
        "vel_deltas": {},
        "micro_offsets": {},
        "vel_bigrams": {},
        "micro_bigrams": {},
        "aux_cache": {},
        "use_sha1": False,
        "num_tokens": 0,
        "train_perplexity": 0.0,
        "train_seconds": 0.0,
    }
    groove_sampler_ngram.sample(model, bars=1, rhythm_schema="<straight8>")
    assert seen and seen[0][0] == (-1, "<straight8>")
