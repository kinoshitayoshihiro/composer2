from generators import chord_generator as cg


def test_progression_lookup(monkeypatch):
    monkeypatch.setattr(cg, "get_progressions", lambda b, mode="major": ["I IV V I"])
    prog = cg._pick_progression("soft_reflective")
    assert prog == "I IV V I"
