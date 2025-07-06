import pytest
from music21 import stream

import utilities.cc_tools as cc


def test_merge_cc_events_override() -> None:
    base = [(0.0, 11, 40), (1.0, 11, 50)]
    more = [(1.0, 11, 70)]
    merged = cc.merge_cc_events(base, more)
    assert merged == [(0.0, 11, 40), (1.0, 11, 70)]


def test_merge_cc_events_mixed_input() -> None:
    base = [{"time": 0.0, "cc": 11, "val": 10}, (1.0, 11, 20)]
    more = [(0.5, 11, 15)]
    merged = cc.merge_cc_events(base, more, as_dict=True)
    assert merged == [
        {"time": 0.0, "cc": 11, "val": 10},
        {"time": 0.5, "cc": 11, "val": 15},
        {"time": 1.0, "cc": 11, "val": 20},
    ]


def test_finalize_cc_events_sort_and_convert() -> None:
    p = stream.Part()
    p._extra_cc = {(0.5, 11, 70)}
    p.extra_cc = [{"time": 0.0, "cc": 11, "val": 60}]
    res = cc.finalize_cc_events(p)
    assert res == [
        {"time": 0.0, "cc": 11, "val": 60},
        {"time": 0.5, "cc": 11, "val": 70},
    ]
    assert not hasattr(p, "_extra_cc")
