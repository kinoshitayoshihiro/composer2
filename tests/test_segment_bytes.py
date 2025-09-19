import importlib
import io
import sys
from types import ModuleType
from pathlib import Path

from .torch_stub import _stub_torch

_stub_torch()

try:
    import torch

    _StubTensor = torch.tensor([0]).__class__
    if _StubTensor.__module__.startswith("tests.torch_stub"):
        if not hasattr(_StubTensor, "bool"):
            def _tensor_bool(self):
                return self

            _StubTensor.bool = _tensor_bool  # type: ignore[attr-defined]

        if _StubTensor.__getitem__ is list.__getitem__:
            def _tensor_getitem(self, idx):
                if isinstance(idx, _StubTensor):
                    idx = [bool(v) for v in idx]
                    values = [value for value, flag in zip(self, idx) if flag]
                    return _StubTensor(values)

                value = list.__getitem__(self, idx)
                if isinstance(value, list):
                    return _StubTensor(value)
                return value

            _StubTensor.__getitem__ = _tensor_getitem  # type: ignore[assignment]
except Exception:  # pragma: no cover - defensive
    pass


def _stub_pandas() -> None:
    if importlib.util.find_spec("pandas") is not None:
        return

    from utilities import pd_stub

    pd_stub.install_pandas_stub()


_stub_pandas()
sk_mod = ModuleType("sklearn.metrics")
sk_mod.f1_score = lambda *_a, **_k: 1.0  # type: ignore[assignment]
sys.modules.setdefault("sklearn", ModuleType("sklearn"))
sys.modules["sklearn.metrics"] = sk_mod


def _stub_numpy() -> None:
    if importlib.util.find_spec("numpy") is not None:
        return

    np = ModuleType("numpy")

    class ndarray(list):
        @property
        def size(self) -> int:  # pragma: no cover - simple
            if self and isinstance(self[0], list):
                return len(self) * len(self[0])
            return len(self)

        def sum(self, axis=None):  # pragma: no cover - simple
            if axis == 0 and self and isinstance(self[0], list):
                return [sum(col) for col in zip(*self)]
            return sum(self)

    def ones(shape, dtype=None):
        if isinstance(shape, int):
            return ndarray([1] * shape)
        return ndarray([[1] * shape[1] for _ in range(shape[0])])

    np.ones = ones
    np.ndarray = ndarray
    np.bool_ = bool
    np.float32 = float
    sys.modules["numpy"] = np


_stub_numpy()

tp_mod = ModuleType("scripts.train_phrase")
tp_mod.PhraseLSTM = object
_previous_train_phrase = sys.modules.get("scripts.train_phrase")

pt_mod = ModuleType("models.phrase_transformer")
pt_mod.PhraseTransformer = object
sys.modules.setdefault("models", ModuleType("models"))
sys.modules["models.phrase_transformer"] = pt_mod

_segment_bytes = None
try:
    sys.modules["scripts.train_phrase"] = tp_mod
    from scripts.segment_phrase import segment_bytes as _segment_bytes  # noqa: E402
finally:
    if _previous_train_phrase is not None:
        sys.modules["scripts.train_phrase"] = _previous_train_phrase
    else:
        sys.modules.pop("scripts.train_phrase", None)

segment_bytes = _segment_bytes


class DummyModel:
    def __call__(self, feats: dict, mask: object) -> list[list[float]]:
        import torch

        n = len(feats["pitch_class"][0])
        return {"boundary": torch.tensor([[0.1 * i for i in range(1, n + 1)]])}


def _stub_miditoolkit() -> None:
    mt = ModuleType("miditoolkit")

    class Note:
        def __init__(self, pitch: int, start: float, end: float) -> None:
            self.pitch = pitch
            self.start = start
            self.end = end
            self.velocity = 64

    class Instrument:
        def __init__(self) -> None:
            self.notes = [
                Note(60, 0.0, 0.5),
                Note(62, 0.5, 1.0),
                Note(64, 1.0, 1.5),
            ]

    class MidiFile:
        def __init__(self, file) -> None:  # pragma: no cover - stub
            self.instruments = [Instrument()]

    mt.MidiFile = MidiFile
    sys.modules["miditoolkit"] = mt


def _stub_pretty_midi() -> None:
    """Install a lightweight ``pretty_midi`` stub if the real package is missing."""

    try:  # use the real package when available
        import pretty_midi as _pm  # noqa: F401

        if _pm.__class__.__name__ != "_dummy_module":
            return
    except Exception:  # pragma: no cover - optional dependency
        pass

    pm = ModuleType("pretty_midi")

    class PrettyMIDI:
        def __init__(
            self, _file=None, *, initial_tempo=120
        ) -> None:  # pragma: no cover - stub
            self.instruments = []
            self.initial_tempo = initial_tempo

        def write(self, _f) -> None:  # pragma: no cover - stub
            pass

        def get_piano_roll(self, fs: int = 24):  # pragma: no cover - stub
            import numpy as np

            return np.ones((1, 4))

    class Instrument(list):
        def __init__(
            self, program: int = 0, is_drum: bool = False
        ) -> None:  # pragma: no cover - stub
            super().__init__()
            self.program = program
            self.is_drum = is_drum

    class Note:
        def __init__(
            self, velocity: int, pitch: int, start: float, end: float
        ) -> None:  # pragma: no cover - stub
            self.velocity = velocity
            self.pitch = pitch
            self.start = start
            self.end = end

    pm.PrettyMIDI = PrettyMIDI
    pm.Instrument = Instrument
    pm.Note = Note
    sys.modules["pretty_midi"] = pm


def _stub_streamlit() -> None:
    st = ModuleType("streamlit")
    st.sidebar = ModuleType("sidebar")
    st.sidebar.selectbox = lambda *a, **k: "transformer"
    st.sidebar.slider = lambda *a, **k: 0.5
    st.sidebar.button = lambda *a, **k: False
    st.file_uploader = lambda *a, **k: None
    st.title = lambda *a, **k: None
    st.json = lambda *a, **k: None
    st.line_chart = lambda *a, **k: None
    st.warning = lambda *a, **k: None
    st.session_state = {}

    # Add cache methods for compatibility
    def cache_decorator(func):
        return func

    st.cache = cache_decorator
    st.cache_data = cache_decorator

    sys.modules["streamlit"] = st


_stub_miditoolkit()
_stub_pretty_midi()
_stub_streamlit()

MIDI = bytes.fromhex("4d54686400000006000100010060" "4d54726b0000000400ff2f00")


def test_segment_bytes_schema() -> None:
    model = DummyModel()
    res = segment_bytes(MIDI, model, 0.0)
    assert res and isinstance(res[0][0], int) and isinstance(res[0][1], float)


def test_piano_roll_stub() -> None:

    mod = importlib.import_module("streamlit_app.phrase_gui")
    pm = mod.pretty_midi.PrettyMIDI(io.BytesIO(MIDI))
    roll = pm.get_piano_roll(fs=24).sum(axis=0)
    df = mod.pd.DataFrame({"roll": roll})
    assert hasattr(df, "__iter__") and len(df) == 4
