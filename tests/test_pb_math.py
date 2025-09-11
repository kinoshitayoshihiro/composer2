import pytest
from utilities.pb_math import (
    RAW_MAX,
    RAW_CENTER,
    DELTA_MAX,
    norm_to_raw,
    raw_to_norm,
    clip_delta,
    BendRange,
    norm_to_pb,
    pb_to_norm,
    semi_to_pb,
    pb_to_semi,
    PITCHWHEEL_CENTER,
    PB_MIN,
    PB_MAX,
)


def test_constants():
    assert RAW_MAX == 16383
    assert RAW_CENTER == 8192
    assert DELTA_MAX == 8191


def test_norm_roundtrip_center():
    raw = norm_to_raw(0.0)
    assert raw == RAW_CENTER
    norm = raw_to_norm(raw)
    assert abs(norm) < 1e-9


def test_norm_roundtrip_edges():
    assert norm_to_raw(1.0) == RAW_CENTER + DELTA_MAX
    assert norm_to_raw(-1.0) == RAW_CENTER - DELTA_MAX
    assert norm_to_raw(2.0) == RAW_CENTER + DELTA_MAX  # clip
    assert norm_to_raw(-2.0) == RAW_CENTER - DELTA_MAX


def test_raw_to_norm_bounds():
    assert raw_to_norm(-100) == pytest.approx(-1.0, abs=1e-3)
    assert raw_to_norm(RAW_MAX + 100) == pytest.approx(1.0, abs=1e-3)


def test_clip_delta():
    assert clip_delta(DELTA_MAX + 10) == DELTA_MAX
    assert clip_delta(-DELTA_MAX - 10) == -DELTA_MAX


def test_bend_range():
    br = BendRange(semitones=2.0)
    assert br.cents_to_norm(0.0) == 0.0
    assert br.cents_to_norm(200.0) == 1.0
    assert br.cents_to_norm(-200.0) == -1.0
    assert br.norm_to_cents(1.0) == 200.0
    assert br.norm_to_cents(-1.0) == -200.0


def test_norm_pb_roundtrip():
    arr = norm_to_pb([-1.0, 1.0])
    assert arr[0] == PB_MIN
    assert arr[1] == PB_MAX
    back = pb_to_norm(arr)
    assert pytest.approx(back[0]) == -1.0
    assert pytest.approx(back[1]) == 1.0


def test_raw_roundtrip_new():
    raw_min = norm_to_raw(-1.0)
    raw_max = norm_to_raw(1.0)
    assert raw_min == PITCHWHEEL_CENTER + PB_MIN
    assert raw_max == PITCHWHEEL_CENTER + PB_MAX
    assert pytest.approx(raw_to_norm(raw_min)) == -1.0
    assert pytest.approx(raw_to_norm(raw_max)) == 1.0


def test_semi_range():
    for rng in [1, 2, 12]:
        assert semi_to_pb(rng, rng) == PB_MAX
        assert semi_to_pb(-rng, rng) == PB_MIN
        assert pytest.approx(pb_to_semi(PB_MAX, rng)) == rng
        assert pytest.approx(pb_to_semi(PB_MIN, rng)) == -rng

