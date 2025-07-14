import pytest

from utilities.progression_templates import get_progressions


@pytest.mark.parametrize(
    "bucket,mode",
    [
        ("soft_reflective", "major"),
        ("soft_reflective", "minor"),
        ("_default", "major"),
        ("_default", "minor"),
        ("unknown", "minor"),
    ],
)
def test_progressions(bucket: str, mode: str) -> None:
    progs = get_progressions(bucket, mode=mode)
    assert isinstance(progs, list)
    assert len(progs) >= 3
