from music21 import note, stream, volume
import utilities.humanizer as hum
from utilities.humanizer import _humanize_velocities
import pytest


def _make_part() -> stream.Part:
    p = stream.Part()
    for i in range(10):
        n = note.Note("C4", quarterLength=0.5)
        n.volume = volume.Volume(velocity=30 + i * 5)
        n.offset = i * 0.5
        p.append(n)
    return p


@pytest.mark.parametrize("curve", ["linear", "cubic-in"])
def test_expr_curve_monotonic(curve: str) -> None:
    part = _make_part()
    _humanize_velocities(part, amount=0, global_settings={"use_expr_cc11": True}, expr_curve=curve)
    vals = [ev["val"] for ev in getattr(part, "extra_cc", []) if ev.get("cc") == 11]
    assert len(vals) == 10
    assert vals == sorted(vals)
    if curve == "cubic-in":
        assert vals[0] < vals[1] < vals[-1]
