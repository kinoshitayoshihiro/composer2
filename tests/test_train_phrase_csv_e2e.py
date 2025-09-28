import csv
import json
from pathlib import Path
import sys
import types
from typing import Any, Dict

import pytest


def _yaml_safe_load(_: object) -> Dict[str, object]:
    return {}


def _program_to_instrument_name(_: int) -> str:
    return "inst"


# stub optional deps
yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = _yaml_safe_load  # type: ignore[attr-defined]
sys.modules.setdefault("yaml", yaml_stub)

phrase_stub = types.ModuleType("models.phrase_transformer")
phrase_stub.PhraseTransformer = object  # type: ignore[attr-defined]
sys.modules.setdefault("models.phrase_transformer", phrase_stub)

pretty_midi_stub = types.ModuleType("pretty_midi")
pretty_midi_stub.PrettyMIDI = object  # type: ignore[attr-defined]
pretty_midi_stub.program_to_instrument_name = (  # type: ignore[attr-defined]
    _program_to_instrument_name
)
sys.modules.setdefault("pretty_midi", pretty_midi_stub)

from tools import corpus_to_phrase_csv as c2p  # noqa: E402
from scripts import train_phrase as tp  # noqa: E402


def test_csv_emit_and_train_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus: Path = tmp_path / "corpus"
    (corpus / "train").mkdir(parents=True)
    (corpus / "valid").mkdir()
    sample: Dict[str, object] = {
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
    c2p.main(
        [
            "--from-corpus",
            str(corpus),
            "--out-train",
            str(train_csv),
            "--out-valid",
            str(valid_csv),
            "--emit-buckets",
        ]
    )
    with train_csv.open() as f:
        header = csv.DictReader(f).fieldnames or []
    assert "velocity_bucket" in header and "duration_bucket" in header

    def fake_train_model(
        train_csv_path: Path,
        val_csv_path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[float, str, dict[str, object]]:
        _ = train_csv_path, val_csv_path
        out_value = kwargs.get("out")
        if out_value is None and len(args) > 2:
            out_value = args[2]
        if out_value is None:
            raise AssertionError("missing out path")
        out = Path(out_value)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        (out.parent / "metrics.json").write_text(
            json.dumps({"best_threshold": 0.5, "by_tag": {}}), encoding="utf-8"
        )
        (out.parent / "metrics_by_tag.json").write_text(json.dumps({}), encoding="utf-8")
        (out.parent / "metrics_epoch.csv").write_text("epoch,f1\n0,1.0\n", encoding="utf-8")
        (out.parent / "preds_preview.json").write_text("[]", encoding="utf-8")
        stats: dict[str, object] = {}
        return 0.5, "cpu", stats

    monkeypatch.setattr(tp, "train_model", fake_train_model)
    ckpt = tmp_path / "m.ckpt"
    tp.main(
        [
            str(train_csv),
            str(valid_csv),
            "--epochs",
            "1",
            "--out",
            str(ckpt),
            "--min-f1",
            "-1",
        ]
    )
    assert ckpt.is_file()
    run_cfg = json.loads((ckpt.with_suffix(".run.json")).read_text())
    assert "viz_enabled" in run_cfg
    assert run_cfg["status"] == "ok"
