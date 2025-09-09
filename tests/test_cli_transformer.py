import json
import pytest

pytest.importorskip("torch")

from modular_composer import cli
from utilities import user_history
from generator import bass_generator


def test_sample_transformer(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        bass_generator,
        "sample_transformer_bass",
        lambda *a, **k: [{"pitch": 60, "offset": 0.0, "duration": 0.25, "velocity": 100}],
    )
    hist_file = tmp_path / "history.jsonl"
    monkeypatch.setattr(user_history, "_HISTORY_FILE", hist_file)

    model = tmp_path / "dummy.pt"
    model.write_text("x")
    cli.main(["sample", str(model), "--backend", "transformer", "--model-name", "dummy"])
    out = capsys.readouterr().out
    assert json.loads(out)[0]["pitch"] == 60
