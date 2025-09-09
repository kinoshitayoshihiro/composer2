import pytest
torch = pytest.importorskip("torch")
from utilities import bass_transformer


def test_token_sample_length(monkeypatch):
    class DummyModel(torch.nn.Module):
        def generate(self, input_ids, **kwargs):
            batch, _ = input_ids.shape
            new_tokens = torch.arange(16).unsqueeze(0).repeat(batch, 1)
            return torch.cat([input_ids, new_tokens], dim=1)

    class DummyTok:
        def add_special_tokens(self, *_a, **_k):
            return 0

        def convert_tokens_to_ids(self, token):
            return 1

        def __len__(self):
            return 10

    monkeypatch.setattr(bass_transformer, "AutoTokenizer", type("T", (), {"from_pretrained": lambda *a, **k: DummyTok()}))
    monkeypatch.setattr(bass_transformer, "AutoModelForCausalLM", type("M", (), {"from_pretrained": lambda *a, **k: DummyModel()}))
    monkeypatch.setattr(bass_transformer, "LoraModel", type("L", (), {"from_pretrained": lambda model, path: model}))

    model = bass_transformer.BassTransformer("dummy")
    out = model.sample([0, 1, 2], top_k=5, temperature=1.0)
    assert len(out) == 16
