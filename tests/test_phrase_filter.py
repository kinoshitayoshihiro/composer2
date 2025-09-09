import pytest

pytest.importorskip("hdbscan")


def test_cluster_phrases():
    from utilities.phrase_filter import cluster_phrases

    ev1 = [
        {"instrument": "kick", "offset": 0.0},
        {"instrument": "hh", "offset": 0.25},
        {"instrument": "snare", "offset": 0.5},
    ]
    ev2 = [
        {"instrument": "kick", "offset": 0.0},
        {"instrument": "hh", "offset": 0.25},
        {"instrument": "snare", "offset": 0.5},
    ]
    ev3 = [
        {"instrument": "kick", "offset": 0.0},
        {"instrument": "hh", "offset": 0.25},
        {"instrument": "snare", "offset": 0.75},
    ]

    mask = cluster_phrases([ev1, ev2, ev3], n=2)
    assert len(mask) == 3
    assert mask[0] is True
