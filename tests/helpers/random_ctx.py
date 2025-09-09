from __future__ import annotations

import contextlib
import random


@contextlib.contextmanager
def seeded_random(seed: int):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)
