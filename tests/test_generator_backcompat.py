from __future__ import annotations

import warnings

from generator import GuitarGenerator


def test_guitar_generator_positional_args():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        gen = GuitarGenerator({}, None)
        assert gen is not None
        assert any(issubclass(ww.category, DeprecationWarning) for ww in w)
