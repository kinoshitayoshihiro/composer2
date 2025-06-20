import pytest
from utilities.fill_dsl import parse


def test_run_tokens_order_and_velocity():
    ev_up = parse("RUNUP", length_beats=2.0, velocity_factor=1.0)
    ev_down = parse("RUNDOWN", length_beats=2.0, velocity_factor=1.0)

    assert [e["instrument"] for e in ev_up] == ["tom1", "tom2", "tom3", "crash"]
    assert [e["instrument"] for e in ev_down] == ["tom3", "tom2", "tom1", "kick"]

    step = 2.0 / 4
    assert pytest.approx([0.0, step, step * 2, step * 3]) == [
        e["offset"] for e in ev_up
    ]
    assert pytest.approx([0.0, step, step * 2, step * 3]) == [
        e["offset"] for e in ev_down
    ]

    up_vel = [e["velocity_factor"] for e in ev_up]
    down_vel = [e["velocity_factor"] for e in ev_down]
    assert up_vel == sorted(up_vel)
    assert down_vel == sorted(down_vel, reverse=True)


def test_run_token_custom_length():
    events = parse("RUNUPx5", length_beats=3.0, velocity_factor=1.0)
    instruments = [e["instrument"] for e in events]
    assert instruments == ["tom1", "tom2", "tom3", "tom1", "tom2", "crash"]
    step = 3.0 / 6
    assert pytest.approx([i * step for i in range(6)]) == [e["offset"] for e in events]
    vel = [e["velocity_factor"] for e in events]
    assert vel == sorted(vel)
