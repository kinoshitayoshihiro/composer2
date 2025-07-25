"""Helpers for installing lightweight stub modules in CI."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from collections.abc import Callable, Iterable

STUB_MODULES = [
    "pkg_resources",
    "yaml",
    "scipy",
    "scipy.signal",
    "music21",
    "pretty_midi",
    "soundfile",
    "torchcrf",
    "torch_crf",
    "TorchCRF",
]


class _dummy_class:
    """Class that ignores all attribute access."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __getattr__(self, name: str):
        return self

    def __call__(self, *args: object, **kwargs: object) -> _dummy_class:
        return self

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"<Dummy {self.__class__.__name__}>"


class _dummy_module(types.ModuleType):
    """Module that returns itself for unknown attributes."""

    def __getattr__(self, name: str):
        attr_name = f"{self.__name__}.{name}"
        attr = _dummy_module(attr_name)
        setattr(self, name, attr)
        sys.modules[attr_name] = attr
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
        "expressions",
        "chord",
        "tempo",
        "volume",
        "midi",
        "exceptions21",
        "scale",
        "common",
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
        "expressions": ["Expression"],
        "chord": ["Chord"],
        "key": ["Key"],
        "meter": ["TimeSignature"],
        "interval": ["Interval"],
        "tempo": ["MetronomeMark"],
        "volume": ["Volume"],
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


def _populate_pretty_midi(mod: types.ModuleType) -> None:
    setattr(mod, "PrettyMIDI", _dummy_class)


def _populate_soundfile(mod: types.ModuleType) -> None:
    setattr(mod, "read", lambda *_a, **_k: ([], 44100))


SPECIAL_POPULATORS = {
    "music21": _populate_music21,
    "scipy": _populate_scipy,
    "yaml": lambda m: setattr(m, "safe_load", lambda *_a, **_k: {}),
    "pretty_midi": _populate_pretty_midi,
    "soundfile": _populate_soundfile,
}


def register_populator(name: str, fn: Callable[[types.ModuleType], None]) -> None:
    SPECIAL_POPULATORS[name] = fn


def install_stubs(
    names: list[str] = STUB_MODULES, force_names: Iterable[str] | None = None
) -> None:
    """Install lightweight stub modules for missing packages."""

    force_set = set(force_names or [])
    for name in names:
        if importlib.util.find_spec(name) is not None and name not in force_set:
            continue
        mod = _ensure_module(name)
        pop = SPECIAL_POPULATORS.get(name)
        if pop is not None:
            pop(mod)
