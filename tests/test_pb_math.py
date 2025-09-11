from utilities.pb_math import RAW_MAX, RAW_CENTER, DELTA_MAX, norm_to_raw, raw_to_norm, clip_delta, BendRange


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
    assert raw_to_norm(-100) == -1.0
    assert raw_to_norm(RAW_MAX + 100) == 1.0


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

