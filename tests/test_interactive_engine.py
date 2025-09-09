import pytest
from utilities import interactive_engine

pytest.importorskip("mido")


class Msg:
    type = "note_on"
    note = 60
    velocity = 100


def test_interactive_trigger(monkeypatch):
    events = []

    class DummyGen:
        def generate(self, prompt, bars):
            events.append(prompt)
            return [{"instrument": "bass"}]

    monkeypatch.setattr(interactive_engine, "TransformerBassGenerator", lambda name: DummyGen())
    eng = interactive_engine.InteractiveEngine(model_name="x")
    out = []
    eng.add_callback(lambda ev: out.append(ev))
    eng._trigger(Msg())
    assert out == [{"instrument": "bass"}]
    assert events


async def _dummy_start(
    eng: interactive_engine.InteractiveEngine, monkeypatch
) -> list[str]:
    sent: list[str] = []

    class Port:
        def __iter__(self):
            yield Msg()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def send(self, msg):
            sent.append(msg.type)

    monkeypatch.setattr(
        interactive_engine,
        "mido",
        type(
            "M",
            (),
            {
                "open_input": lambda *_a, **_k: Port(),
                "open_output": lambda *_a, **_k: Port(),
                "Message": lambda *a, **k: type("Msg", (), {"type": "note_on"})(),
            },
        ),
    )
    monkeypatch.setattr(interactive_engine, "_MIDO_ERROR", None)
    await eng.start("in", "out")
    return sent


def test_start_async(monkeypatch):
    class DummyGen:
        def generate(self, prompt, bars):
            return [{"pitch": 60, "velocity": 100, "duration": 0.1}]

    monkeypatch.setattr(interactive_engine, "TransformerBassGenerator", lambda name: DummyGen())
    eng = interactive_engine.InteractiveEngine(model_name="x")

    sent = __import__("asyncio").run(_dummy_start(eng, monkeypatch))
    assert "note_on" in sent
