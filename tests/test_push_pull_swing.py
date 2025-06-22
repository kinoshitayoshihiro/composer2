from utilities.timing_utils import _combine_timing


def _eval(rel, pp, swing, bpm=120):
    beat_len = 1.0
    return _combine_timing(
        rel,
        beat_len,
        swing_ratio=swing,
        swing_type="eighth",
        push_pull_curve=pp,
        tempo_bpm=bpm,
    )


def test_push_pull_only():
    pp = [20, -20, 0, 0]
    assert _eval(0.0, pp, 0.5) > 0.0
    assert _eval(1.0, pp, 0.5) < 1.0


def test_push_pull_and_swing():
    pp = [0, 0, 0, 0]
    straight = _eval(0.5, pp, 0.5)
    swung = _eval(0.5, pp, 0.66)
    assert swung > straight


def test_sixteenth_swing():
    pp = [0, 0, 0, 0]
    beat_len = 1.0
    base = _combine_timing(
        0.25,
        beat_len,
        swing_ratio=0.5,
        swing_type="sixteenth",
        push_pull_curve=pp,
        tempo_bpm=120,
    )
    swung = _combine_timing(
        0.25,
        beat_len,
        swing_ratio=0.66,
        swing_type="sixteenth",
        push_pull_curve=pp,
        tempo_bpm=120,
    )
    assert swung > base

