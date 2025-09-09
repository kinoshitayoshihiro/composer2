import random
from utilities.vibrato_engine import generate_vibrato, generate_gliss, generate_trill
from utilities import pb_math


def test_generate_vibrato_alternating() -> None:
    events = generate_vibrato(0.25, 1.0, 5.0)
    assert len(events) == 5
    types = [e[0] for e in events]
    assert types == ["pitch_wheel", "aftertouch", "pitch_wheel", "aftertouch", "pitch_wheel"]
    values = [e[2] for e in events]
    assert all(pb_math.PB_MIN <= v <= pb_math.PB_MAX for v in values)


def test_generate_gliss_linear() -> None:
    events = generate_gliss(60, 64, 0.5)
    assert len(events) == 5
    pitches = [p for p, _ in events]
    assert pitches[0] == 60 and pitches[-1] == 64
    assert all(pitches[i] <= pitches[i + 1] for i in range(len(pitches) - 1))


def test_generate_trill_velocity_range() -> None:
    random.seed(0)
    events = generate_trill(60, 0.5, rate=8.0)
    assert len(events) == 5
    pitches = [p for p, _, _ in events]
    assert all(p in (59, 61) for p in pitches)
    for i in range(len(pitches) - 1):
        assert pitches[i] != pitches[i + 1]
    velocities = [v for _, _, v in events]
    assert all(59 <= v <= 69 for v in velocities)
