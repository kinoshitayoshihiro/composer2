import pytest

from utilities.realtime_engine import RealtimeEngine

pytest.importorskip("mido")
pytestmark = pytest.mark.no_midi_port


def test_external_sync_no_port(monkeypatch):
    monkeypatch.setattr("mido.get_input_names", lambda: [])
    monkeypatch.setattr(
        "utilities.groove_sampler_rnn.load",
        lambda p: (__import__("types").SimpleNamespace(), {}),
    )
    eng = RealtimeEngine("dummy.pt", sync="external")
    assert eng.sync == "internal"
