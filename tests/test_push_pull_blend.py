import pytest
from utilities.timing_utils import _combine_timing


def test_push_pull_velocity_blend():
    max_ms = 80
    curve = [max_ms, -max_ms, 0, 0]
    res1 = _combine_timing(
        0.0,
        1.0,
        swing_ratio=0.5,
        swing_type="eighth",
        push_pull_curve=curve,
        tempo_bpm=120,
        max_push_ms=max_ms,
        vel_range=(0.9, 1.1),
        return_vel=True,
    )
    assert pytest.approx(res1.vel_scale, rel=1e-6) == 1.1

    res2 = _combine_timing(
        1.0,
        1.0,
        swing_ratio=0.5,
        swing_type="eighth",
        push_pull_curve=curve,
        tempo_bpm=120,
        max_push_ms=max_ms,
        vel_range=(0.9, 1.1),
        return_vel=True,
    )
    assert pytest.approx(res2.vel_scale, rel=1e-6) == 0.9

    curve2 = [-max_ms, 0, max_ms, 0]
    prev = 1.0
    alpha = 0.5
    scales = []
    for pos in [0.0, 1.0, 2.0]:
        blend = _combine_timing(
            pos,
            1.0,
            swing_ratio=0.5,
            swing_type="eighth",
            push_pull_curve=curve2,
            tempo_bpm=120,
            max_push_ms=max_ms,
            vel_range=(0.9, 1.1),
            return_vel=True,
        )
        smooth = prev * (1 - alpha) + blend.vel_scale * alpha
        scales.append(smooth)
        prev = blend.vel_scale

    diff1 = scales[1] - scales[0]
    diff2 = scales[2] - scales[1]
    assert diff1 <= diff2
