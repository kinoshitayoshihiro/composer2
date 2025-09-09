import random
from utilities import groove_sampler


def test_given_empty_model_when_generate_bar_then_returns_empty():
    model = {"n": 2, "freq": {}, "unigram": {}}
    events = groove_sampler.generate_bar(
        [], model, random.Random(0), resolution=16
    )
    assert events == []
