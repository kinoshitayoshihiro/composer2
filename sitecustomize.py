"""Project-local stub loaded automatically by Python at start-up.
Silences heavy optional dependencies in minimal CI environments.
"""
import importlib.machinery
import importlib.util
import sys
import types  # pragma: no cover

for _name in ("pkg_resources", "yaml", "scipy", "scipy.signal", "music21"):
    if importlib.util.find_spec(_name) is None:
        mod = types.ModuleType(_name)
        mod.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
        sys.modules.setdefault(_name, mod)  # pragma: no cover
