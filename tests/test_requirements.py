import importlib
from pathlib import Path

# Ensure any version shims from sitecustomize are applied before
# importing optional dependencies.  This is necessary when running the
# tests with a plain `pytest` invocation where the repository root is
# not automatically on ``PYTHONPATH`` and therefore ``sitecustomize``
# would not be imported implicitly.
try:  # pragma: no cover - if sitecustomize is missing this is a no-op
    import sitecustomize  # type: ignore
except Exception:  # pragma: no cover - ignore any import issues
    pass

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
    assert parts >= (0, 2)


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
