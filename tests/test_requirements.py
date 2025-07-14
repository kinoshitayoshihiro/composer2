import importlib
from pathlib import Path

import pretty_midi


def test_no_duplicate_packages():
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    names: list[str] = []
    for line in req.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = (
            line.replace("~=", "==")
            .replace(">=", "==")
            .split("==", 1)[0]
            .strip()
        )
        names.append(name.lower())
    assert len(names) == len(set(names))


def test_pretty_midi_version():
    parts = tuple(int(p) for p in pretty_midi.__version__.split(".")[:2])
    assert parts >= (0, 3)


def test_public_docstrings():
    modules = ["utilities.vocal_sync", "utilities.progression_templates"]
    for name in modules:
        mod = importlib.import_module(name)
        for attr_name, obj in vars(mod).items():
            if attr_name.startswith("_") or not callable(obj):
                continue
            if getattr(obj, "__module__", None) != name:
                continue
            assert obj.__doc__, f"{attr_name} missing docstring"
            assert "Parameters" in obj.__doc__, f"{attr_name} missing Parameters"
