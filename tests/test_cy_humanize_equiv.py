import pytest
from music21 import note, stream, volume

from utilities.humanizer import _apply_swing, _humanize_velocities

try:
    from cyext.humanize import apply_swing as cy_apply_swing
    from cyext.humanize import humanize_velocities as cy_humanize
except Exception:  # pragma: no cover - optional
    cy_apply_swing = None
    cy_humanize = None


def _make_part():
    p = stream.Part()
    for i in range(8):
        n = note.Note(60)
        n.volume = volume.Volume(velocity=64)
        n.quarterLength = 0.25
        n.offset = i * 0.25
        p.append(n)
    return p


@pytest.mark.skipif(cy_apply_swing is None, reason="cython ext not built")
def test_apply_swing_equiv():
    p1 = _make_part()
    p2 = _make_part()
    _apply_swing(p1, 0.6, subdiv=8)
    cy_apply_swing(p2, 0.6, 8)
    offs1 = [n.offset for n in p1.notes]
    offs2 = [n.offset for n in p2.notes]
    assert offs1 == pytest.approx(offs2)


@pytest.mark.skipif(cy_humanize is None, reason="cython ext not built")
def test_humanize_vel_equiv():
    p1 = _make_part()
    p2 = _make_part()
    settings = {"use_expr_cc11": True, "use_aftertouch": True}
    _humanize_velocities(p1, amount=5, global_settings=settings)
    cy_humanize(p2, 5, True, True)
    v1 = [n.volume.velocity for n in p1.notes]
    v2 = [n.volume.velocity for n in p2.notes]
    assert v1 == v2
    assert getattr(p1, "extra_cc", []) == getattr(p2, "extra_cc", [])
