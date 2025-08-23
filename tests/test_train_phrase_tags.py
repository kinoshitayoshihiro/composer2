import csv
import json
import tempfile

import pytest

pytest.importorskip("torch")

from scripts import train_phrase as tp


def test_section_mood_columns(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "train").mkdir(parents=True)
    (corpus / "valid").mkdir()
    sample = {
        "pitch": 60,
        "velocity": 100,
        "duration": 1,
        "pos": 0,
        "boundary": 1,
        "bar": 0,
        "instrument": "",
        "tags": {"section": "A", "mood": "happy"},
    }
    (corpus / "train" / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    (corpus / "valid" / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    csv_dir = tmp_path / "csvs"

    class TD:
        def __enter__(self):
            csv_dir.mkdir()
            return str(csv_dir)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: TD())
    monkeypatch.setattr(tp, "train_model", lambda *a, **k: (0.0, "cpu"))
    tp.main([
        "--data",
        str(corpus),
        "--sample",
        "4",
        "--epochs",
        "0",
        "--out",
        str(tmp_path / "m.ckpt"),
        "--min-f1",
        "-1",
    ])
    train_csv = csv_dir / "train.csv"
    with train_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[0]["section"] == "A"
    assert rows[0]["mood"] == "happy"
