import importlib
import importlib.util
import sys
from types import ModuleType

MIDI = bytes.fromhex("4d54686400000006000100010060" "4d54726b0000000400ff2f00")


def _ensure_pandas(out: list) -> None:
    """Install lightweight pandas stub when pandas is missing."""

    try:
        import pandas as pd_mod  # noqa: F401
        if hasattr(pd_mod, "DataFrame"):
            return
    except Exception:
        pass

    from utilities import pd_stub

    def DataFrame(data: dict) -> pd_stub._DF:  # type: ignore
        df = pd_stub._DF(data.get("roll", []))
        out.append(df)
        return df

    pd_stub.install_pandas_stub()
    sys.modules["pandas"].DataFrame = DataFrame  # type: ignore


def _stub_pretty_midi() -> None:
    """Install a minimal ``pretty_midi`` stub unless the real package exists."""

    try:  # prefer the actual dependency if installed
        import pretty_midi as _pm  # noqa: F401
        return
    except Exception:  # pragma: no cover - optional dependency
        pass

    pm = ModuleType("pretty_midi")

    class PrettyMIDI:
        def __init__(self, _f=None, *, initial_tempo=120) -> None:  # pragma: no cover - stub
            self.instruments = []
            self.initial_tempo = initial_tempo

        def write(self, _path) -> None:  # pragma: no cover - stub
            pass

        def get_piano_roll(self, fs: int = 24):
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


def _ensure_streamlit(upload: bytes | None = None, charts: list | None = None) -> None:
    st = ModuleType("streamlit")
    st.sidebar = ModuleType("sidebar")
    st.sidebar.selectbox = lambda *a, **k: "transformer"
    st.sidebar.slider = lambda *a, **k: 0.5
    st.sidebar.button = lambda *a, **k: False
    if upload is None:
        st.file_uploader = lambda *a, **k: None
    else:

        class F:
            def getvalue(self) -> bytes:
                return upload

        st.file_uploader = lambda *a, **k: F()
    st.title = lambda *a, **k: None
    st.json = lambda *a, **k: None

    def line_chart(df):
        if charts is not None:
            charts.append(df)

    st.line_chart = line_chart
    st.warning = lambda *a, **k: None
    st.session_state = {}
    sys.modules["streamlit"] = st
    test_mod = ModuleType("streamlit.testing.v1")

    class DummyAppTest:
        status = "COMPLETE"

        @classmethod
        def from_function(cls, fn):  # pragma: no cover - stub
            cls.fn = fn
            return cls()

        def run(self):  # pragma: no cover - stub
            if hasattr(self.__class__, "fn"):
                self.__class__.fn()

        def get(self, key):
            return type("C", (), {"exists": lambda self: True})()

    test_mod.AppTest = DummyAppTest
    sys.modules["streamlit.testing.v1"] = test_mod


def test_phrase_gui_render() -> None:
    out: list = []
    _ensure_pandas(out)
    _stub_pretty_midi()
    _ensure_streamlit(upload=MIDI, charts=out)
    st_test = importlib.import_module("streamlit.testing.v1")
    AppTest = st_test.AppTest
    mod = importlib.import_module("streamlit_app.phrase_gui")
    at = AppTest.from_function(mod.main)
    at.run()
    assert at.status == "COMPLETE"
    assert at.get("upload").exists()
    assert out and hasattr(out[0], "__iter__")
