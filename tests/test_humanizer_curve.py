import pytest
pytest.importorskip("music21")
from music21 import note, stream, volume
import utilities.humanizer as hum
from utilities.humanizer import _humanize_velocities


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


def test_curve_difference() -> None:
    part_a = _make_part()
    part_b = _make_part()
    _humanize_velocities(part_a, amount=0, global_settings={"use_expr_cc11": True}, expr_curve="linear")
    _humanize_velocities(part_b, amount=0, global_settings={"use_expr_cc11": True}, expr_curve="cubic-in")
    vals_a = [ev["val"] for ev in part_a.extra_cc if ev["cc"] == 11]
    vals_b = [ev["val"] for ev in part_b.extra_cc if ev["cc"] == 11]
    assert vals_a != vals_b


def test_kick_leak_jitter() -> None:
    p = stream.Part()
    k = note.Note("C2", quarterLength=0.5)
    k.pitch.midi = 36
    h = note.Note("F#2", quarterLength=0.5)
    h.pitch.midi = 42
    k.volume = volume.Volume(velocity=64)
    h.volume = volume.Volume(velocity=64)
    p.insert(0.0, k)
    p.insert(0.05, h)
    _humanize_velocities(p, amount=0, kick_leak_jitter=3)
    assert 61 <= h.volume.velocity <= 67
