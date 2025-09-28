import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, NoReturn

import pytest

from scripts import train_phrase as tp  # noqa: E402

pytest.importorskip("torch")


def test_filter_and_reweight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # create corpus with sections A (twice) and B (once)
    corpus = tmp_path / "corpus"
    (corpus / "train").mkdir(parents=True)
    (corpus / "valid").mkdir()
    rows: List[Dict[str, object]] = [
        {
            "pitch": 60,
            "velocity": 64,
            "duration": 1,
            "pos": 0,
            "boundary": 1,
            "bar": i,
            "instrument": "",
            "tags": {"section": sec},
        }
        for i, sec in enumerate(["A", "A", "B"])
    ]
    for split in ["train", "valid"]:
        samples_path = corpus / split / "samples.jsonl"
        with samples_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    captured: Dict[str, Path | str] = {}

    original_train_model = tp.train_model

    def fake_train_model(
        train_csv: Path,
        val_csv: Path,
        *args: object,
        **kwargs: Any,
    ) -> tuple[float, str, Dict[str, Dict[str, int]]]:
        _ = val_csv, args
        captured["train_csv"] = train_csv
        captured["reweight"] = str(kwargs.get("reweight", ""))
        return 0.0, "cpu", {"class_counts": {}, "tag_counts": {}}

    monkeypatch.setattr(tp, "train_model", fake_train_model)
    tp.main(
        [
            "--data",
            str(corpus),
            "--include-tags",
            "section=A",
            "--reweight",
            "tag=section,scheme=inv_freq",
            "--epochs",
            "0",
            "--out",
            str(tmp_path / "m.ckpt"),
            "--min-f1",
            "-1",
        ]
    )
    train_csv = Path(captured["train_csv"])
    with train_csv.open(encoding="utf-8") as f:
        rows_csv = list(csv.DictReader(f))
    assert rows_csv and all(r["section"] == "A" for r in rows_csv)
    assert captured["reweight"] == "tag=section,scheme=inv_freq"

    # CLI 用のスタブはここまで。以降は本物の train_model を使用する。
    monkeypatch.setattr(tp, "train_model", original_train_model)

    # weighted sampling on explicit CSV
    full_train = tmp_path / "train_full.csv"
    tp.write_csv(
        [
            {
                "pitch": 60,
                "velocity": 64,
                "duration": 1,
                "pos": 0,
                "boundary": 1,
                "bar": i,
                "instrument": "",
                "section": sec,
                "mood": "",
            }
            for i, sec in enumerate(["A", "A", "B"])
        ],
        full_train,
    )
    weights: Dict[str, List[float]] = {}

    def fake_wrs(w: Iterable[float], n: int, r: bool):
        values = list(w)
        weights["w"] = values
        torch_mod = tp.torch
        return torch_mod.utils.data.WeightedRandomSampler(values, n, r)

    import torch

    monkeypatch.setattr(tp, "torch", torch)
    monkeypatch.setattr(tp.torch.utils.data, "WeightedRandomSampler", fake_wrs)
    tp.train_model(
        full_train,
        full_train,
        epochs=1,
        arch="lstm",
        out=tmp_path / "m2.ckpt",
        batch_size=1,
        d_model=32,
        max_len=32,
        reweight="tag=section,scheme=inv_freq",
        instrument=None,
        include_tags=None,
        exclude_tags=None,
        viz=False,
    )
    assert weights["w"]

    # run CLI to ensure weights land in run.json
    out_ckpt = tmp_path / "m3.ckpt"
    tp.main(
        [
            str(full_train),
            str(full_train),
            "--epochs",
            "0",
            "--reweight",
            "tag=section,scheme=inv_freq",
            "--out",
            str(out_ckpt),
            "--min-f1",
            "-1",
        ]
    )
    run_data = json.loads(out_ckpt.with_suffix(".run.json").read_text())
    assert run_data["status"] == "ok"
    assert run_data["sampler_weights_summary"]["tag_weights_top10"]


def test_run_json_written_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows: List[Dict[str, object]] = [
        {
            "pitch": 60,
            "velocity": 64,
            "duration": 1,
            "pos": 0,
            "boundary": 1,
            "bar": 0,
            "instrument": "",
            "section": "",
            "mood": "",
        }
    ]
    train_csv = tmp_path / "train.csv"
    valid_csv = tmp_path / "valid.csv"
    tp.write_csv(rows, train_csv)
    tp.write_csv(rows, valid_csv)

    def fail_train_model(*args: object, **kwargs: object) -> NoReturn:
        raise ValueError("explode")

    monkeypatch.setattr(tp, "train_model", fail_train_model)
    out_ckpt = tmp_path / "fail.ckpt"
    exit_code = tp.main(
        [
            str(train_csv),
            str(valid_csv),
            "--epochs",
            "1",
            "--out",
            str(out_ckpt),
            "--min-f1",
            "-1",
        ]
    )
    assert exit_code == 2
    run_data = json.loads(out_ckpt.with_suffix(".run.json").read_text())
    assert run_data["status"] == "error"
    assert run_data.get("error", {}).get("type") == "ValueError"
    assert run_data.get("error", {}).get("message") == "explode"
