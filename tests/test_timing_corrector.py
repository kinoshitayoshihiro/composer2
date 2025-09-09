from music21 import note, stream
import statistics
from utilities.timing_corrector import TimingCorrector


def test_timing_corrector_reduces_jitter() -> None:
    part = stream.Part()
    offsets = [0.0, 1.1, 2.05, 3.0]
    for o in offsets:
        n = note.Note('C4', quarterLength=1.0)
        n.offset = o
        part.insert(o, n)
    tc = TimingCorrector(alpha=0.5)
    corrected = tc.correct_part(part)
    assert corrected.notes[0].offset == offsets[0]
    orig_res = [o - round(o) for o in offsets]
    new_res = [n.offset - round(n.offset) for n in corrected.notes]
    assert statistics.pstdev(new_res) < statistics.pstdev(orig_res)
