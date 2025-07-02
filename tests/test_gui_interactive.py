import pytest

pytest.importorskip("streamlit", reason="GUI deps not installed in CI")

from streamlit_app import gui


def test_setup_interactive(monkeypatch):
    logs = []

    class DummyEngine:
        def __init__(self, model_name: str, bpm: float) -> None:
            self.args = (model_name, bpm)
            self.cbs = []

        def add_callback(self, cb):
            self.cbs.append(cb)

    monkeypatch.setattr(gui, "InteractiveEngine", DummyEngine)
    engine = gui.setup_interactive("m", 120.0, lambda msg: logs.append(msg))
    assert isinstance(engine, DummyEngine)
    assert engine.cbs
    engine.cbs[0]({"note": 60})
    assert logs
