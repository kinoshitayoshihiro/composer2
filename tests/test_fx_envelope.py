import pytest
from music21 import stream

from utilities.fx_envelope import apply
from utilities.cc_tools import to_sorted_dicts


@pytest.mark.parametrize("shape", ["lin", "exp", "log"])
def test_monotonic(shape):
    part = stream.Part()
    env = {0.0: {"type": "reverb", "start": 0, "end": 100, "duration_ql": 1, "shape": shape}}
    apply(part, env, bpm=120)
    events = to_sorted_dicts(part._extra_cc)
    values = [e["val"] for e in events]
    assert values == sorted(values)


def test_bpm_variation_counts():
    env = {0.0: {"cc": 91, "start": 0, "end": 100, "duration_ql": 1.0}}
    counts = []
    edges = []
    for bpm in [60, 120, 180]:
        part = stream.Part()
        apply(part, env, bpm=bpm)
        events = to_sorted_dicts(part._extra_cc)
        counts.append(len(events))
        edges.append((events[0]["val"], events[-1]["val"]))
    assert len(set(counts)) == 3
    assert len(set(edges)) == 1


def test_validation_errors():
    p = stream.Part()
    with pytest.raises(ValueError):
        apply(p, {0.0: {"start": 0, "end": 100}})
    with pytest.raises(ValueError):
        apply(p, {0.0: {"type": "reverb", "start": 0, "end": 128}})
    with pytest.raises(ValueError):
        apply(p, {0.0: {"type": "reverb", "start": 0, "end": 100, "shape": "bad"}})
