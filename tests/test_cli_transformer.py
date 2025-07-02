import json

from modular_composer import cli
from utilities import ai_sampler, user_history


def test_sample_transformer(monkeypatch, tmp_path, capsys):
    class DummyPipe:
        def __call__(self, prompt, max_new_tokens=64, num_return_sequences=1):
            return [{"generated_text": prompt + json.dumps([{"instrument": "bass"}])}]

    monkeypatch.setattr(ai_sampler, "pipeline", lambda *a, **k: DummyPipe())
    monkeypatch.setattr(
        ai_sampler,
        "AutoModelForCausalLM",
        type("M", (), {"from_pretrained": lambda *a, **k: object()}),
    )
    monkeypatch.setattr(
        ai_sampler,
        "AutoTokenizer",
        type("T", (), {"from_pretrained": lambda *a, **k: object()}),
    )
    hist_file = tmp_path / "history.jsonl"
    monkeypatch.setattr(user_history, "_HISTORY_FILE", hist_file)

    model = tmp_path / "dummy.pt"
    model.write_text("x")
    cli.main(["sample", str(model), "--ai-backend", "transformer", "--model-name", "dummy"])
    out = capsys.readouterr().out
    assert json.loads(out)[0]["instrument"] == "bass"
