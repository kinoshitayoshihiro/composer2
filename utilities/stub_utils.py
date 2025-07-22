"""Helpers for installing lightweight stub modules in CI."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from typing import List


class _dummy_class:
    """Class that ignores all attribute access."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __getattr__(self, name: str):
        return self


class _dummy_module(types.ModuleType):
    """Module that returns itself for unknown attributes."""

    def __getattr__(self, name: str):
        attr = _dummy_module(f"{self.__name__}.{name}")
        setattr(self, name, attr)
        return attr


def _create_module(name: str) -> types.ModuleType:
    mod = _dummy_module(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return mod


def _ensure_module(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = _create_module(name)
    sys.modules[name] = mod
    if "." in name:
        parent_name, child = name.rsplit(".", 1)
        parent = _ensure_module(parent_name)
        setattr(parent, child, mod)
    return mod


def _populate_music21(mod: types.ModuleType) -> None:
    submods = [
        "pitch",
        "note",
        "stream",
        "converter",
        "instrument",
        "articulations",
        "harmony",
        "key",
        "meter",
        "interval",
    ]
    for sub in submods:
        _ensure_module(f"music21.{sub}")
    cls_map = {
        "pitch": ["Pitch"],
        "note": ["Note"],
        "stream": ["Stream"],
        "converter": ["Converter"],
        "instrument": [
            "Instrument",
            "Guitar",
            "Violin",
            "AcousticBass",
            "AltoSaxophone",
            "Piano",
        ],
        "articulations": ["Articulation"],
        "harmony": ["ChordSymbol"],
        "key": ["Key"],
        "meter": ["TimeSignature"],
        "interval": ["Interval"],
    }
    for mod_name, classes in cls_map.items():
        sm = sys.modules[f"music21.{mod_name}"]
        for cls_name in classes:
            cls = type(cls_name, (_dummy_class,), {})
            setattr(sm, cls_name, cls)


def _populate_scipy(mod: types.ModuleType) -> None:
    sig = _ensure_module("scipy.signal")

    def _empty_array():
        try:
            import numpy as np  # type: ignore

            return np.array([])
        except Exception:
            return []

    def hamming(*_a, **_k):
        return _empty_array()

    def butter(*_a, **_k):
        return _empty_array(), _empty_array()

    def lfilter(*_a, **_k):
        return _empty_array()

    sig.hamming = hamming
    sig.butter = butter
    sig.lfilter = lfilter
    mod.signal = sig


SPECIAL_POPULATORS = {
    "music21": _populate_music21,
    "scipy": _populate_scipy,
    "yaml": lambda m: setattr(m, "safe_load", lambda *_a, **_k: {}),
}


def install_stubs(names: List[str], force: bool = False) -> None:
    """Install lightweight stub modules for missing packages."""

    for name in names:
        if importlib.util.find_spec(name) is not None and not force:
            continue
        mod = _ensure_module(name)
        pop = SPECIAL_POPULATORS.get(name)
        if pop is not None:
            pop(mod)

