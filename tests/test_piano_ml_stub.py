import importlib.util
import pytest


deps_available = all(
    importlib.util.find_spec(mod) is not None
    for mod in ("torch", "transformers", "peft")
)

@pytest.mark.skipif(not deps_available, reason="missing ML deps")
def test_piano_ml_generate(monkeypatch):
    import torch
    from generator import piano_ml_generator as pmg

    class DummyModel(torch.nn.Module):
        def generate(self, input_ids, **kwargs):
            batch, _ = input_ids.shape
            new_tokens = torch.zeros((batch, 4), dtype=torch.long)
            return torch.cat([input_ids, new_tokens], dim=1)

    class DummyBase(torch.nn.Module):
        def __init__(self, *_a, **_k):
            super().__init__()

    monkeypatch.setattr(pmg, "GPT2Config", lambda **_: object())
    monkeypatch.setattr(pmg, "GPT2LMHeadModel", DummyBase)
    monkeypatch.setattr(pmg, "PeftModel", type("P", (), {"from_pretrained": staticmethod(lambda base, path: DummyModel())}))

    gen = pmg.PianoMLGenerator("dummy")
    events = gen.generate(max_bars=1)
    assert isinstance(events, list)
