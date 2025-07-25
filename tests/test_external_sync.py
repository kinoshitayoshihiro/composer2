import pytest

from utilities.realtime_engine import RealtimeEngine

pytest.importorskip("mido")
pytestmark = pytest.mark.no_midi_port


def test_external_sync_no_port(monkeypatch):
    monkeypatch.setattr("mido.get_input_names", lambda: [])
    def fake_load(_p):
        mod = types.ModuleType("dummy")
        return mod, {}

    monkeypatch.setattr("utilities.groove_sampler_rnn.load", fake_load)
    eng = RealtimeEngine("dummy.pt", sync="external")
    assert eng.sync == "internal"
