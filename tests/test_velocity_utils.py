from utilities.velocity_utils import scale_velocity


def test_scale_velocity_basic():
    assert scale_velocity(100, 0.5) == 50
    assert scale_velocity(120, 1.2) == 127
    assert scale_velocity(1, 0.0) == 1
