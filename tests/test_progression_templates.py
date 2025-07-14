import pytest

from utilities.progression_templates import get_progressions, _load


@pytest.mark.parametrize(
    "bucket,mode",
    [
        ("soft_reflective", "major"),
        ("soft_reflective", "minor"),
    ],
)
def test_lookup(bucket: str, mode: str) -> None:
    lst = get_progressions(bucket, mode=mode)
    assert isinstance(lst, list) and lst


def test_cache() -> None:
    id1 = id(_load())
    id2 = id(_load())
    assert id1 == id2


@pytest.mark.parametrize(
    "bucket,mode",
    [
        ("missing", "major"),
        ("soft_reflective", "dorian"),
    ],
)
def test_keyerror(bucket: str, mode: str) -> None:
    with pytest.raises(KeyError):
        get_progressions(bucket, mode=mode)
