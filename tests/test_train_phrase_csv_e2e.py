import csv
import json
from pathlib import Path
import sys
import types

# stub optional deps
sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda x: {}))
sys.modules.setdefault(
    "models.phrase_transformer", types.SimpleNamespace(PhraseTransformer=object)
)
sys.modules.setdefault(
    "pretty_midi", types.SimpleNamespace(PrettyMIDI=object, program_to_instrument_name=lambda p: "inst")
)

from tools import corpus_to_phrase_csv as c2p
from scripts import train_phrase as tp


def test_csv_emit_and_train_stub(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "train").mkdir(parents=True)
    (corpus / "valid").mkdir()
    sample = {
        "pitch": 60,
        "velocity": 64,
        "duration": 1.0,
        "pos": 0,
        "boundary": 1,
        "bar": 0,
        "instrument": "inst",
    }
    (corpus / "train" / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    (corpus / "valid" / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    train_csv = tmp_path / "train.csv"
    valid_csv = tmp_path / "valid.csv"
    c2p.main([
        "--from-corpus",
        str(corpus),
        "--out-train",
        str(train_csv),
        "--out-valid",
        str(valid_csv),
        "--emit-buckets",
    ])
    with train_csv.open() as f:
        header = csv.DictReader(f).fieldnames or []
    assert "velocity_bucket" in header and "duration_bucket" in header

    def fake_train_model(*args, **kwargs):
        out = Path(args[4] if len(args) > 4 else kwargs.get("out"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("")
        (out.parent / "metrics.json").write_text(
            json.dumps({"best_threshold": 0.5, "by_tag": {}})
        )
        (out.parent / "metrics_by_tag.json").write_text(json.dumps({}))
        (out.parent / "metrics_epoch.csv").write_text("epoch,f1\n0,1.0\n")
        (out.parent / "preds_preview.json").write_text("[]")
        return 0.5, "cpu", {}

    monkeypatch.setattr(tp, "train_model", fake_train_model)
    ckpt = tmp_path / "m.ckpt"
    tp.main([
        str(train_csv),
        str(valid_csv),
        "--epochs",
        "1",
        "--out",
        str(ckpt),
        "--min-f1",
        "-1",
    ])
    assert ckpt.is_file()
    run_cfg = json.loads((ckpt.with_suffix(".run.json")).read_text())
    assert "viz_enabled" in run_cfg
