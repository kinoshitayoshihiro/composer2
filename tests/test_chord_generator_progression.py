import importlib.util
from pathlib import Path

from tests.helpers.random_ctx import seeded_random

spec = importlib.util.spec_from_file_location(
    "chord_generator", Path("generators/chord_generator.py")
)
cg = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(cg)


def test_pick_progression(monkeypatch):
    monkeypatch.setattr(cg, "get_progressions", lambda b, mode="major": ["I IV V I"])
    with seeded_random(0):
        assert cg._pick_progression("soft_reflective") == "I IV V I"
