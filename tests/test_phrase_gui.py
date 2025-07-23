import importlib
import importlib.util
import sys
from types import ModuleType

MIDI = bytes.fromhex("4d54686400000006000100010060" "4d54726b0000000400ff2f00")


def _ensure_pandas(out: list) -> None:
    pd = ModuleType("pandas")

    class DF(list):
        pass

    def DataFrame(data: dict) -> DF:  # pragma: no cover - stub
        df = DF(data.get("roll", []))
        out.append(df)
        return df

    pd.DataFrame = DataFrame
    sys.modules["pandas"] = pd


def _stub_pretty_midi() -> None:
    pm = ModuleType("pretty_midi")

    class PrettyMIDI:
        def __init__(self, _f) -> None:  # pragma: no cover - stub
            pass

        def get_piano_roll(self, fs: int = 24):
            import numpy as np

            return np.ones((1, 4))

    pm.PrettyMIDI = PrettyMIDI
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
