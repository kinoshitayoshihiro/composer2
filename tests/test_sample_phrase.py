import csv
import sys
import subprocess
from pathlib import Path
import math
import json

import pytest

torch = pytest.importorskip("torch")
from models.phrase_transformer import PhraseTransformer


def test_sample_phrase(tmp_path):
    ckpt_path = tmp_path / "toy.ckpt"
    model = PhraseTransformer(d_model=16, max_len=8, nhead=2, num_layers=1, pitch_vocab_size=128)
    meta = {
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 2,
        "max_len": 8,
        "duv_mode": "reg",
        "vel_bins": 0,
        "dur_bins": 0,
        "vocab_pitch": 128,
        "vocab": {},
    }
    torch.save({"model": model.state_dict(), "meta": meta}, ckpt_path)

    out_csv = tmp_path / "out.csv"
    out_midi = tmp_path / "out.mid"
    cmd = [
        sys.executable,
        "-m",
        "scripts.sample_phrase",
        "--ckpt",
        str(ckpt_path),
        "--length",
        "8",
        "--out-csv",
        str(out_csv),
        "--out-midi",
        str(out_midi),
        "--seed",
        "0",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    with out_csv.open() as f:
        rows = list(csv.reader(f))
    assert len(rows) - 1 == 8

    try:
        import pretty_midi  # type: ignore
    except Exception:
        assert not out_midi.exists()
    else:
        pm = pretty_midi.PrettyMIDI(str(out_midi))
        assert pm.instruments and len(pm.instruments[0].notes) == 8


def test_sample_logits_topp():
    from scripts.sample_phrase import sample_logits

    logits = torch.tensor([1.0, 0.5, 0.1])
    idx = sample_logits(logits, temperature=1.0, topk=0, topp=0.9)
    assert 0 <= idx < 3
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    mask = cdf <= 0.9
    mask[0] = True
    filtered = sorted_probs * mask
    filtered = filtered / filtered.sum()
    assert float(filtered.sum()) == pytest.approx(1.0)
    allowed = {i for i, m in zip(sorted_idx.tolist(), mask.tolist()) if m}
    for _ in range(1000):
        j = sample_logits(logits, temperature=1.0, topk=0, topp=0.9)
        assert j in allowed


def test_seed_and_bars(tmp_path):
    ckpt_path = tmp_path / "toy.ckpt"
    model = PhraseTransformer(d_model=16, max_len=8, nhead=2, num_layers=1, pitch_vocab_size=128)
    with torch.no_grad():
        if model.head_dur_reg is not None:
            model.head_dur_reg.weight.zero_()
            model.head_dur_reg.bias.fill_(math.log1p(1.0))
        if model.head_vel_reg is not None:
            model.head_vel_reg.weight.zero_()
            model.head_vel_reg.bias.fill_(0.5)
    meta = {
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 2,
        "max_len": 8,
        "duv_mode": "reg",
        "vel_bins": 0,
        "dur_bins": 0,
        "vocab_pitch": 128,
        "vocab": {},
    }
    torch.save({"model": model.state_dict(), "meta": meta}, ckpt_path)

    out_csv1 = tmp_path / "out1.csv"
    out_csv2 = tmp_path / "out2.csv"
    cmd_base = [
        sys.executable,
        "-m",
        "scripts.sample_phrase",
        "--ckpt",
        str(ckpt_path),
        "--length",
        "64",
        "--bars",
        "1",
        "--seed",
        "123",
    ]
    subprocess.run(cmd_base + ["--out-csv", str(out_csv1)], check=True, cwd=Path(__file__).resolve().parents[1])
    subprocess.run(cmd_base + ["--out-csv", str(out_csv2)], check=True, cwd=Path(__file__).resolve().parents[1])

    with out_csv1.open() as f1, out_csv2.open() as f2:
        r1 = list(csv.DictReader(f1))
        r2 = list(csv.DictReader(f2))
    assert len(r1) == len(r2) == 4
    assert r1[:2] == r2[:2]


def test_dur_max_beats(tmp_path):
    ckpt_path = tmp_path / "toy.ckpt"
    model = PhraseTransformer(d_model=16, max_len=8, nhead=2, num_layers=1, pitch_vocab_size=128)
    with torch.no_grad():
        if model.head_dur_reg is not None:
            model.head_dur_reg.weight.zero_()
            model.head_dur_reg.bias.fill_(math.log1p(10.0))
    meta = {
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 2,
        "max_len": 8,
        "duv_mode": "reg",
        "vel_bins": 0,
        "dur_bins": 0,
        "vocab_pitch": 128,
        "vocab": {},
    }
    torch.save({"model": model.state_dict(), "meta": meta}, ckpt_path)
    out_csv = tmp_path / "out.csv"
    out_midi = tmp_path / "out.mid"
    cmd = [
        sys.executable,
        "-m",
        "scripts.sample_phrase",
        "--ckpt",
        str(ckpt_path),
        "--length",
        "4",
        "--out-csv",
        str(out_csv),
        "--out-midi",
        str(out_midi),
        "--dur-max-beats",
        "0.5",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert all(float(r["duration_beats"]) <= 0.5 for r in rows)
    try:
        import pretty_midi  # type: ignore
    except Exception:
        assert not out_midi.exists()
    else:
        pm = pretty_midi.PrettyMIDI(str(out_midi))
        sec_per_beat = 60.0 / 110.0
        assert all(
            (n.end - n.start) <= 0.5 * sec_per_beat + 1e-6 for n in pm.instruments[0].notes
        )


def test_strict_tags_fail(tmp_path):
    train_csv = tmp_path / "train.csv"
    valid_csv = tmp_path / "valid.csv"
    header = ["pitch", "velocity", "duration", "pos", "boundary", "bar", "section"]
    row = {"pitch": 60, "velocity": 100, "duration": 1, "pos": 0, "boundary": 0, "bar": 0, "section": "B"}
    for p in (train_csv, valid_csv):
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerow(row)
    (tmp_path / "tag_vocab.json").write_text(json.dumps({"section": ["A"]}))
    cmd = [
        sys.executable,
        "scripts/train_phrase.py",
        str(train_csv),
        str(valid_csv),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--arch",
        "transformer",
        "--nhead",
        "2",
        "--layers",
        "1",
        "--max-len",
        "8",
        "--strict-tags",
    ]
    res = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1])
    assert res.returncode != 0
