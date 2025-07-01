from music21 import note, stream
import random
from utilities.humanizer import apply_velocity_histogram


def test_velocity_histogram_sampling() -> None:
    part = stream.Part()
    for i in range(100):
        n = note.Note('C4', quarterLength=1.0)
        n.offset = i
        part.insert(i, n)
    hist = {90: 0.7, 110: 0.3}
    random.seed(0)
    apply_velocity_histogram(part, hist)
    vels = [n.volume.velocity for n in part.notes]
    count_90 = vels.count(90)
    ratio = count_90 / len(vels)
    assert 0.5 < ratio < 0.9


def test_velocity_histogram_reproducible() -> None:
    part = stream.Part()
    for i in range(10):
        n = note.Note('C4', quarterLength=1.0)
        n.offset = i
        part.insert(i, n)
    hist = {90: 0.7, 110: 0.3}
    import copy
    part2 = copy.deepcopy(part)
    random.seed(42)
    apply_velocity_histogram(part, hist)
    vels1 = [n.volume.velocity for n in part.notes]
    random.seed(42)
    apply_velocity_histogram(part2, hist)
    vels2 = [n.volume.velocity for n in part2.notes]
    assert vels1 == vels2
