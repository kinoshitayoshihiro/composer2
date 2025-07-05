import pytest
pytest.importorskip("music21")
from music21 import note, stream, volume

from utilities import humanizer


def test_gliss_cc_injected(monkeypatch):
    humanizer.load_profiles({"test": {"gliss_pairs": True}})
    p = stream.Part()
    n1 = note.Note(60, quarterLength=0.5)
    n1.volume = volume.Volume(velocity=80)
    n2 = note.Note(62, quarterLength=0.5)
    n2.volume = volume.Volume(velocity=80)
    p.insert(0.0, n1)
    p.insert(0.5, n2)
    humanizer.apply(p, "test")
    ccs = getattr(p, "extra_cc", [])
    assert {"time": 0.0, "cc": 5, "val": 63} in ccs
    assert {"time": 0.0, "cc": 65, "val": 127} in ccs
