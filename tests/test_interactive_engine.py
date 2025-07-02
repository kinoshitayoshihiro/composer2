from utilities import interactive_engine


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
