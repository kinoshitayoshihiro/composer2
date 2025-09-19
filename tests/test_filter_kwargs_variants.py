import functools

from utilities.audio_to_midi_batch import _filter_kwargs


def f(_a, *, x=None):
    return _a, x


def g(_a, **kw):
    return _a, kw


def test_filter_kwargs_basic_and_drop_none():
    kw = {"x": 1, "y": 2, "z": None}
    assert _filter_kwargs(f, kw) == {"x": 1}
    assert _filter_kwargs(g, kw) == {"x": 1, "y": 2}


def test_filter_kwargs_partial_unwrap():
    p = functools.partial(f, 0)
    kw = {"x": 5, "y": 9}
    assert _filter_kwargs(p, kw) == {"x": 5}
