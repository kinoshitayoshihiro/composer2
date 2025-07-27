"""Helpers for installing lightweight stub modules in CI."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from collections.abc import Callable, Iterable
from pathlib import Path

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
    "pandas",
    "joblib",
    "sklearn",
    "numba",
    "uvicorn",
    "websockets",
    "mido",
    "tomli",
    "numpy",
    "typing_extensions",
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

    # support "|" (PEP604) and generics
    def __or__(self, other):
        return self

    __ror__ = __or__

    @classmethod
    def __class_getitem__(cls, _item):
        return cls

    @classmethod
    def __mro_entries__(cls, bases):
        return (cls,)

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

    @classmethod
    def __class_getitem__(cls, item):
        return cls

    def __or__(self, other):
        return self

    __ror__ = __or__

    @classmethod
    def __mro_entries__(cls, bases):
        return (cls,)


def _create_module(name: str) -> types.ModuleType:
    mod = _dummy_module(name)
    spec = importlib.machinery.ModuleSpec(
        name, loader=importlib.machinery.BuiltinImporter
    )
    if '.' not in name:
        spec.submodule_search_locations = []
        mod.__package__ = name
        mod.__path__ = []
    mod.__spec__ = spec
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
        "duration",
        "dynamics",
        "spanner",
        "tie",
        "layout",
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
            "Flute",
            "KeyboardInstrument",
            "Contrabass",
            "Violoncello",
            "Viola",
            "Vocalist",
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
        "duration": ["Duration"],
        "scale": ["ConcreteScale"],
        "tie": ["Tie"],
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

    def resample_poly(x, up, down, axis=0):
        return _empty_array()

    sig.hamming = hamming
    sig.butter = butter
    sig.lfilter = lfilter
    sig.resample_poly = resample_poly
    mod.signal = sig


def _populate_pretty_midi(mod: types.ModuleType) -> None:
    setattr(mod, "PrettyMIDI", _dummy_class)


def _populate_soundfile(mod: types.ModuleType) -> None:
    setattr(mod, "read", lambda *_a, **_k: ([], 44100))
    def _write(path, data, sr, *a, **k):
        Path(path).touch()
    setattr(mod, "write", _write)


def _populate_numpy(mod: types.ModuleType) -> None:
    import math
    from array import array

    class _Array:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            if isinstance(idx, tuple):
                x, y = idx
                return self.data[x][y]
            return self.data[idx]

        def __len__(self):
            return len(self.data)

        def __setitem__(self, idx, value):
            if isinstance(idx, tuple):
                x, y = idx
                self.data[x][y] = value
            else:
                self.data[idx] = value

        def sum(self, axis=None, keepdims=False):
            if axis is None:
                if self.data and isinstance(self.data[0], list):
                    return sum(sum(row) for row in self.data)
                return sum(self.data)
            if axis == 1:
                res = [sum(row) for row in self.data]
                if keepdims:
                    res = [[r] for r in res]
                return _Array(res) if keepdims else res
            if axis == 0:
                cols = [sum(col) for col in zip(*self.data)]
                return _Array([cols]) if keepdims else cols

        def __itruediv__(self, other):
            if isinstance(other, _Array):
                other = other.data
            if isinstance(other, list):
                if self.data and isinstance(self.data[0], list):
                    for r, div in zip(self.data, other):
                        if isinstance(div, list):
                            div = div[0]
                        for i, v in enumerate(r):
                            r[i] = v / div
                else:
                    for i, v in enumerate(self.data):
                        val = other[i]
                        if isinstance(val, list):
                            val = val[0]
                        self.data[i] = v / val
            else:
                if self.data and isinstance(self.data[0], list):
                    for r in self.data:
                        for i, v in enumerate(r):
                            r[i] = v / other
                else:
                    for i, v in enumerate(self.data):
                        self.data[i] = v / other
            return self

        def __truediv__(self, other):
            copied = [row[:] for row in self.data] if self.data and isinstance(self.data[0], list) else list(self.data)
            out = _Array(copied)
            out.__itruediv__(other)
            return out

        @property
        def shape(self):
            if self.data and isinstance(self.data[0], list):
                return (len(self.data), len(self.data[0]))
            return (len(self.data),)

        @property
        def ndim(self):
            return len(self.shape)

        def astype(self, _dtype):
            return self

    def _zeros(n, dtype=None):
        if isinstance(n, tuple):
            return _Array([[0.0] * n[1] for _ in range(n[0])])
        return _Array([0.0] * n)

    def _ones(n, dtype=None):
        if isinstance(n, tuple):
            return _Array([[1.0] * n[1] for _ in range(n[0])])
        return _Array([1.0] * n)

    def _array(obj, dtype=None):
        if isinstance(obj, list) and obj and isinstance(obj[0], list):
            return _Array([list(row) for row in obj])
        return _Array(list(obj))

    def _linspace(start, stop, num, endpoint=True):
        step = (stop - start) / (num - 1 if endpoint else num)
        return array("f", [start + step * i for i in range(num)])

    def _sin(arr):
        return array("f", [math.sin(x) for x in arr])

    def _apply_elemwise(fn):
        def _inner(x):
            data = x.data if isinstance(x, _Array) else x
            if data and isinstance(data[0], list):
                return _Array([[fn(v) for v in row] for row in data])
            return _Array([fn(v) for v in data])
        return _inner

    mod.pi = math.pi
    mod.float32 = "float32"
    mod.linspace = _linspace
    mod.sin = _sin
    mod.zeros = _zeros
    mod.ones = _ones
    mod.array = _array
    mod.asarray = lambda x, *a, **k: _array(x)
    mod.exp = _apply_elemwise(math.exp)
    mod.log = _apply_elemwise(math.log)

    # minimal numpy.typing with NDArray alias
    typing_mod = types.ModuleType("numpy.typing")
    typing_mod.__spec__ = importlib.machinery.ModuleSpec(
        "numpy.typing", loader=importlib.machinery.BuiltinImporter
    )
    class _NDArray:
        pass
    typing_mod.NDArray = _NDArray
    sys.modules.setdefault("numpy.typing", typing_mod)


SPECIAL_POPULATORS = {
    "music21": _populate_music21,
    "scipy": _populate_scipy,
    "yaml": lambda m: setattr(m, "safe_load", lambda *_a, **_k: {}),
    "pretty_midi": _populate_pretty_midi,
    "soundfile": _populate_soundfile,
    "numpy": _populate_numpy,
    "pandas": lambda m: setattr(m, "DataFrame", _dummy_class),
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
