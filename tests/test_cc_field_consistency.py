from music21 import note, stream

from utilities.humanizer import _humanize_velocities


def test_extra_cc_keys_consistent() -> None:
    part = stream.Part()
    n = note.Note("C4", quarterLength=1.0)
    n.volume.velocity = 90
    part.insert(0.0, n)
    settings = {"use_expr_cc11": True, "use_aftertouch": True}
    _humanize_velocities(part, amount=3, global_settings=settings)
    for ev in getattr(part, "extra_cc", []):
        assert "cc" in ev and "val" in ev
        assert "number" not in ev and "value" not in ev
