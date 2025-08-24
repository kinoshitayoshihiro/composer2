import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from scripts import train_phrase as tp


def test_filter_and_reweight(tmp_path: Path, monkeypatch) -> None:
    # create corpus with sections A (twice) and B (once)
    corpus = tmp_path / "corpus"
    (corpus / "train").mkdir(parents=True)
    (corpus / "valid").mkdir()
    rows = [
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
        with (corpus / split / "samples.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    captured: dict[str, Path | str] = {}

    def fake_train_model(train_csv, val_csv, *a, **kw):
        captured["train_csv"] = train_csv
        captured["reweight"] = kw.get("reweight")
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
    with train_csv.open() as f:
        rows_csv = list(csv.DictReader(f))
    assert rows_csv and all(r["section"] == "A" for r in rows_csv)
    assert captured["reweight"] == "tag=section,scheme=inv_freq"

    # weighted sampling on explicit CSV
    full_train = tmp_path / "train_full.csv"
    tp.write_csv(
        [
            {"pitch": 60, "velocity": 64, "duration": 1, "pos": 0, "boundary": 1, "bar": i, "instrument": "", "section": sec, "mood": ""}
            for i, sec in enumerate(["A", "A", "B"])
        ],
        full_train,
    )
    weights: dict[str, list[float]] = {}

    def fake_wrs(w, n, r):
        weights["w"] = list(w)
        import torch

        return torch.utils.data.WeightedRandomSampler(w, n, r)
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
    assert run_data["sampler_weights_summary"]["tag_weights_top10"]
