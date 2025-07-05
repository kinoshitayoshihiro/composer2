import pytest
from utilities import interactive_engine

pytest.importorskip("transformers")
pytest.importorskip("torch")
pytest.importorskip("mido")

class DummyModel:
    def __init__(self, *a, **k):
        pass
    def sample(self, seq, top_k, temperature, rhythm_schema=None):
        return [60, 62, 64, 65]
    def encode(self, events):
        return [e.get("pitch", 0) for e in events]


def test_transformer_engine(monkeypatch):
    monkeypatch.setattr(interactive_engine, "BassTransformer", DummyModel)
    eng = interactive_engine.TransformerInteractiveEngine(model_name="x")
    out = []
    eng.add_callback(lambda ev: out.append(ev))
    class Msg:
        type = "note_on"
        note = 60
        velocity = 100
    eng._trigger(Msg())
    assert len(out) > 0
