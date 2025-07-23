import importlib
import io
import sys
from types import ModuleType


def _stub_torch() -> None:
    if importlib.util.find_spec("torch") is not None:
        return

    import numpy as np

    torch = ModuleType("torch")

    class Tensor(np.ndarray):  # type: ignore[misc]
        def __new__(cls, data):
            return np.asarray(data).view(cls)

        def unsqueeze(self, dim: int = 0) -> "Tensor":
            return Tensor(np.expand_dims(self, dim))

        def clamp(self, max: int | None = None) -> "Tensor":
            out = np.minimum(self, max) if max is not None else self
            return Tensor(out)

    torch.Tensor = Tensor
    torch.tensor = lambda data, dtype=None: Tensor(data)
    torch.arange = lambda n, dtype=None: Tensor(np.arange(n))
    torch.ones = lambda *shape, dtype=None: Tensor(np.ones(shape, dtype=float))
    torch.float32 = np.float32
    torch.long = np.int64
    torch.bool = np.bool_

    class NoGrad:
        def __enter__(self) -> None:  # pragma: no cover - stub
            pass

        def __exit__(self, *exc: object) -> None:  # pragma: no cover - stub
            pass

    torch.no_grad = NoGrad
    torch.sigmoid = lambda x: Tensor(1 / (1 + np.exp(-x)))

    nn = ModuleType("torch.nn")
    nn.Module = object
    torch.nn = nn
    utils = ModuleType("torch.utils")
    data = ModuleType("torch.utils.data")
    data.DataLoader = object
    data.Dataset = object
    utils.data = data
    torch.utils = utils
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.utils"] = utils
    sys.modules["torch.utils.data"] = data


_stub_torch()

pd_mod = sys.modules.setdefault("pandas", ModuleType("pandas"))


class _DF(list):
    pass


pd_mod.DataFrame = lambda data: _DF(data.get("roll", []))  # type: ignore[attr-defined]
sk_mod = ModuleType("sklearn.metrics")
sk_mod.f1_score = lambda *_a, **_k: 1.0  # type: ignore[assignment]
sys.modules.setdefault("sklearn", ModuleType("sklearn"))
sys.modules["sklearn.metrics"] = sk_mod

from scripts.segment_phrase import segment_bytes  # noqa: E402


class DummyModel:
    def __call__(self, feats: dict, mask: object) -> list[list[float]]:
        import torch

        n = len(feats["pitch_class"][0])
        return torch.tensor([[0.1 * i for i in range(1, n + 1)]])


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
        return
    except Exception:  # pragma: no cover - optional dependency
        pass

    pm = ModuleType("pretty_midi")

    class PrettyMIDI:
        def __init__(self, _file=None, *, initial_tempo=120) -> None:  # pragma: no cover - stub
            self.instruments = []
            self.initial_tempo = initial_tempo

        def write(self, _f) -> None:  # pragma: no cover - stub
            pass

        def get_piano_roll(self, fs: int = 24):  # pragma: no cover - stub
            import numpy as np

            return np.ones((1, 4))

    class Instrument(list):
        def __init__(self, program: int = 0, is_drum: bool = False) -> None:  # pragma: no cover - stub
            super().__init__()
            self.program = program
            self.is_drum = is_drum

    class Note:
        def __init__(self, velocity: int, pitch: int, start: float, end: float) -> None:  # pragma: no cover - stub
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
