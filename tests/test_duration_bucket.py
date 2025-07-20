from utilities.duration_bucket import to_bucket


def test_duration_bucket_boundaries() -> None:
    assert to_bucket(0.1) == 0
    assert to_bucket(0.25) == 1
    assert to_bucket(0.6) == 3
    assert to_bucket(4.0) == 5
    assert to_bucket(10.0) == 7
