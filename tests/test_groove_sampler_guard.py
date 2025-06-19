import random
from utilities import groove_sampler


def test_no_infinite_loop():
    model = {"n": 2, "freq": {}}
    events = groove_sampler.generate_bar([], model, random.Random(0), resolution=16)
    assert events == []
