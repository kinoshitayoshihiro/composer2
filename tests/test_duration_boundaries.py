import utilities.duration_bucket as db


def test_duration_boundaries() -> None:
    for i, bound in enumerate(db._BOUNDS):
        assert db.to_bucket(bound - 1e-6) == i
        assert db.to_bucket(bound) == i + 1
