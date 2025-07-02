import json
from utilities import ai_sampler


def test_transformer_bass_generate(monkeypatch):
    calls = {}

    class DummyPipe:
        def __call__(self, prompt, max_new_tokens=64, num_return_sequences=1):
            calls['prompt'] = prompt
            return [{"generated_text": prompt + json.dumps([{"instrument": "bass"}])}]

    monkeypatch.setattr(ai_sampler, "pipeline", lambda *a, **k: DummyPipe())
    monkeypatch.setattr(ai_sampler, "AutoModelForCausalLM", type("M", (), {"from_pretrained": lambda *a, **k: object()}) )
    monkeypatch.setattr(ai_sampler, "AutoTokenizer", type("T", (), {"from_pretrained": lambda *a, **k: object()}) )
    gen = ai_sampler.TransformerBassGenerator(model_name="dummy")
    events = gen.generate([], 1)
    assert events == [{"instrument": "bass"}]
    assert "events" in calls["prompt"]
