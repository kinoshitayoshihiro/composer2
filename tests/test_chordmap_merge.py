from utilities.chordmap_merge import merge_chordmaps

BASE = [
    {"bar": 1, "root": "C", "quality": "maj7", "dur": 1},
    {"bar": 2, "root": "D", "quality": "7", "dur": 1, "bass": "E"},
    {"bar": 3, "root": "F", "quality": "maj7", "dur": 1},
    {"bar": 4, "root": "A", "quality": "7", "dur": 1},
]

NARR = [
    {"bar": 1, "op": "add_tension", "add": ["9", "9", "11", "13"]},
    {"bar": 2, "op": "slash", "bass": "G"},
    {"bar": 3, "op": "substitute", "to": {"root": "G", "quality": "m7"}},
    {"bar": 4, "op": "cadence_lock"},
    {"bar": 4, "op": "substitute", "to": {"quality": "m9"}},
]

POLICY = {
    "root_change_allowed": False,
    "tension_limits": {"max_stack": 2},
    "bass_guard": True,
    "strong_beat_lock": True,
    "weight_threshold": 0.0,
}


def test_merge_rules_minimal():
    merged = merge_chordmaps(BASE, NARR, POLICY)
    by_bar = {c["bar"]: c for c in merged}

    bar1 = by_bar[1]
    assert bar1.get("tensions") in (["9", "11"], ["11", "9"])

    bar2 = by_bar[2]
    assert bar2.get("bass") == "E"

    bar3 = by_bar[3]
    assert bar3["root"] == "F"
    assert "m7" in bar3["quality"]

    bar4 = by_bar[4]
    assert bar4["root"] == "A"
    assert bar4["quality"] == "7"
