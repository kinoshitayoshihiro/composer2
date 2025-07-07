import random
from music21 import note

from generator.articulation import ArticulationEngine, QL_32ND, CC_AFTERTOUCH


def test_gliss_note_count() -> None:
    eng = ArticulationEngine()
    events = eng.generate_gliss("C4", "C5", 1.0)
    assert len(events) == int(1.0 / QL_32ND)


def test_trill_aftertouch() -> None:
    eng = ArticulationEngine(random.Random(0))
    events = eng.generate_trill("C4", 0.5)
    mids = [n.pitch.midi for n, _ in events]
    assert len(events) == int(0.5 / QL_32ND)
    assert all(mids[i] != mids[i + 1] for i in range(len(mids) - 1))
    cc = eng.cc_events
    assert cc and all(e[1] == CC_AFTERTOUCH for e in cc)
