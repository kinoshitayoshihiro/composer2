import types
import pytest
from utilities.live_player import play_live

pytestmark = pytest.mark.stretch


def test_live_stub(monkeypatch):
    sent = []

    class Dummy:
        def send(self, msg):
            sent.append(msg)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr("mido.open_output", lambda name: Dummy())
    monkeypatch.setattr("mido.get_output_names", lambda: ["port"])

    events = [{"instrument": "kick", "offset": 0.0, "velocity": 100, "duration": 0.25}]
    play_live(events, bpm=120)
    assert sent
